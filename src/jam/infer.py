#!/usr/bin/env python3
"""
Generate audio using JAM model
Reads from filtered test set and generates audio using CFM+DiT model.
"""

import os
import json
import random
import sys
import torch
import torchaudio
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import accelerate
import pyloudnorm as pyln
from safetensors.torch import load_file
from muq import MuQMuLan
import numpy as np

from jam.dataset import enhance_webdataset_config, DiffusionWebDataset
from jam.model.vae import StableAudioOpenVAE, DiffRhythmVAE

# DiffRhythm imports for CFM+DiT model
from jam.model import CFM, DiT

def get_negative_style_prompt(device, file_path):
    vocal_stlye = np.load(file_path)

    vocal_stlye = torch.from_numpy(vocal_stlye).to(device)  # [1, 512]
    vocal_stlye = vocal_stlye.half()

    return vocal_stlye

def normalize_audio(audio, normalize_lufs=True):
    audio = audio - audio.mean(-1, keepdim=True)
    audio = audio / (audio.abs().max(-1, keepdim=True).values + 1e-8)
    if normalize_lufs:
        meter = pyln.Meter(rate=44100)
        target_lufs = -14.0
        loudness = meter.integrated_loudness(audio.transpose(0, 1).numpy())
        normalised = pyln.normalize.loudness(audio.transpose(0, 1).numpy(), loudness, target_lufs)
        normalised = torch.from_numpy(normalised).transpose(0, 1)
    else:
        normalised = audio
    return normalised

class FilteredTestSetDataset(Dataset):
    """Custom dataset for loading from filtered test set JSON"""
    def __init__(self, test_set_path, diffusion_dataset, muq_model, num_samples=None, random_crop_style=False, num_style_secs=30, use_prompt_style=False):
        with open(test_set_path, 'r') as f:
            self.test_samples = json.load(f)
        
        if num_samples is not None:
            self.test_samples = self.test_samples[:num_samples]
            
        self.diffusion_dataset = diffusion_dataset
        self.muq_model = muq_model
        self.random_crop_style = random_crop_style
        self.num_style_secs = num_style_secs
        self.use_prompt_style = use_prompt_style
        if self.use_prompt_style:
            print("Using prompt style instead of audio style.")

    def __len__(self):
        return len(self.test_samples)
    
    def __getitem__(self, idx):
        test_sample = self.test_samples[idx]
        sample_id = test_sample["id"]
        
        # Load LRC data
        lrc_path = test_sample["lrc_path"]
        with open(lrc_path, 'r') as f:
            lrc_data = json.load(f)
        if 'word' not in lrc_data:
            data = {'word': lrc_data}
            lrc_data = data
        
        # Generate style embedding from original audio on-the-fly
        audio_path = test_sample["audio_path"]
        if self.use_prompt_style:
            prompt_path = test_sample["prompt_path"]
            prompt = open(prompt_path, 'r').read()
            if len(prompt) > 300:
                print(f"Sample {sample_id} has prompt length {len(prompt)}")
                prompt = prompt[:300]
            print(prompt)
            style_embedding = self.muq_model(texts=[prompt]).squeeze(0)
        else:
            style_embedding = self.generate_style_embedding(audio_path)
        
        duration = test_sample["duration"]
        
        # Create fake latent with correct length
        # Assuming frame_rate from config (typically 21.5 fps for 44.1kHz)
        frame_rate = 21.5
        num_frames = int(duration * frame_rate)
        fake_latent = torch.randn(128, num_frames)  # 128 is latent dim
        
        # Create sample tuple matching DiffusionWebDataset format
        fake_sample = (
            sample_id,
            fake_latent,     # latent with correct duration
            style_embedding, # style from actual audio
            lrc_data        # actual LRC data
        )
        
        # Process through DiffusionWebDataset's process_sample_safely
        processed_sample = self.diffusion_dataset.process_sample_safely(fake_sample)
        
        # Add metadata
        if processed_sample is not None:
            processed_sample['test_metadata'] = {
                'sample_id': sample_id,
                'audio_path': audio_path,
                'lrc_path': lrc_path,
                'duration': duration,
                'num_frames': num_frames
            }
        
        return processed_sample
    
    def generate_style_embedding(self, audio_path):
        """Generate style embedding using MuQ model on the whole music"""
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 24kHz if needed (MuQ expects 24kHz)
        if sample_rate != 24000:
            resampler = torchaudio.transforms.Resample(sample_rate, 24000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Ensure waveform is 2D (channels, time) - squeeze out channel dim for mono
        waveform = waveform.squeeze(0)  # Now shape is (time,)
        
        # Move to same device as model
        waveform = waveform.to(self.muq_model.device)
        
        # Generate embedding using MuQ model
        with torch.inference_mode():
            # MuQ expects batch dimension and 1D audio, returns (batch, embedding_dim)
            if self.random_crop_style:
                # Randomly crop 30 seconds from the waveform
                total_samples = waveform.shape[0]
                target_samples = 24000 * self.num_style_secs  # 30 seconds at 24kHz
                
                start_idx = random.randint(0, total_samples - target_samples)
                style_embedding = self.muq_model(wavs=waveform.unsqueeze(0)[..., start_idx:start_idx + target_samples])
            else:
                style_embedding = self.muq_model(wavs=waveform.unsqueeze(0)[..., :24000 * self.num_style_secs])
        
        # Keep shape as (embedding_dim,) not scalar
        return style_embedding[0]


def custom_collate_fn_with_metadata(batch, base_collate_fn):
    """Custom collate function that preserves test_metadata"""
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Extract test_metadata before collating
    test_metadata = [item.pop('test_metadata') for item in batch]
    
    # Use base collate function for the rest
    collated = base_collate_fn(batch)
    
    # Add test_metadata back
    if collated is not None:
        collated['test_metadata'] = test_metadata
    
    return collated


def load_model(model_config, checkpoint_path, device):
    """
    Load JAM CFM model from checkpoint (follows infer.py pattern)
    """
    # Build CFM model from config
    dit_config = model_config["dit"].copy()
    # Add text_num_embeds if not specified - should be at least 64 for phoneme tokens
    if "text_num_embeds" not in dit_config:
        dit_config["text_num_embeds"] = 256  # Default value from DiT
    
    cfm = CFM(
        transformer=DiT(**dit_config),
        **model_config["cfm"]
    )
    cfm = cfm.to(device)
    
    # Load checkpoint - use the path from config
    checkpoint = load_file(checkpoint_path)
    cfm.load_state_dict(checkpoint, strict=False)
    
    return cfm.eval()


def generate_latent(model, batch, sample_kwargs, negative_style_prompt_path=None, ignore_style=False):
    """
    Generate latent from batch data (follows infer.py pattern)
    """
    with torch.inference_mode():
        batch_size = len(batch["lrc"])
        text = batch["lrc"]
        style_prompt = batch["prompt"]
        start_time = batch["start_time"]
        duration_abs = batch["duration_abs"]
        duration_rel = batch["duration_rel"]
        
        # Create zero conditioning latent
        # Handle case where model might be wrapped by accelerator
        max_frames = model.max_frames
        cond = torch.zeros(batch_size, max_frames, 64).to(text.device)
        pred_frames = [(0, max_frames)]

        default_sample_kwargs = {
            "cfg_strength": 4,
            "steps": 50,
            "batch_infer_num": 1
        }
        sample_kwargs = {**default_sample_kwargs, **sample_kwargs}
        
        if negative_style_prompt_path is None:
            negative_style_prompt_path = 'public_checkpoints/vocal.npy'
            negative_style_prompt = get_negative_style_prompt(text.device, negative_style_prompt_path)
        elif negative_style_prompt_path == 'zeros':
            negative_style_prompt = torch.zeros(1, 512).to(text.device)
        else:
            negative_style_prompt = get_negative_style_prompt(text.device, negative_style_prompt_path)

        negative_style_prompt = negative_style_prompt.repeat(batch_size, 1)

        latents, _ = model.sample(
            cond=cond,
            text=text,
            style_prompt=negative_style_prompt if ignore_style else style_prompt,
            duration_abs=duration_abs,
            duration_rel=duration_rel,
            negative_style_prompt=negative_style_prompt,
            start_time=start_time,
            latent_pred_segments=pred_frames,
            **sample_kwargs
        )
        
        return latents


def main():
    import warnings
    warnings.filterwarnings("ignore", message="Possible clipped samples in output.", category=UserWarning)

    # Setup accelerator
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    
    # Load config
    cfg_cli = OmegaConf.from_dotlist(sys.argv[1:])
    config_path = cfg_cli.get('config', 'configs/jam_infer.yaml')
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, cfg_cli)
    OmegaConf.resolve(config)

    # Override output directory for evaluation
    output_dir = config.evaluation.output_dir
    # Save config to output directory
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        config_save_path = os.path.join(output_dir, "generation_config.yaml")
        with open(config_save_path, 'w') as f:
            OmegaConf.save(config, f)
        print(f"Config saved to: {config_save_path}")
    
    
    if config.evaluation.checkpoint_path == "":
        print("Please set the checkpoint path in the config file")
        return

    if accelerator.is_main_process:
        print("🎵 JAM Song Generation")
        print(f"Config: {config_path}")
        print(f"Output directory: {output_dir}")
        print(f"Checkpoint: {config.evaluation.checkpoint_path}")
        print(f"VAE type: {config.evaluation.get('vae_type', 'stable_audio')}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Process index: {accelerator.process_index}")

    # Wait for main process to create directory
    accelerator.wait_for_everyone()
    
    # Use filtered test set
    test_set_path = config.evaluation.test_set_path
    if not os.path.exists(test_set_path):
        if accelerator.is_main_process:
            print(f"Test set not found at {test_set_path}")
        return
    
    # Load models
    if accelerator.is_main_process:
        print("Loading models...")
    
    # Load VAE based on configuration
    vae_type = config.evaluation.get('vae_type', 'stable_audio')
    if vae_type == 'diffrhythm':
        if accelerator.is_main_process:
            print("Loading DiffRhythm VAE...")
        vae = DiffRhythmVAE(device=device).to(device)
    else:
        if accelerator.is_main_process:
            print("Loading StableAudio VAE...")
        vae = StableAudioOpenVAE().to(device)
    
    cfm_model = load_model(config.model, config.evaluation.checkpoint_path, device)
    
    # Print model parameter count
    total_params = sum(p.numel() for p in cfm_model.parameters())
    trainable_params = sum(p.numel() for p in cfm_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load MuQ model for style embeddings
    muq_model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(device).eval()
    
    # Print model size only on main process
    if accelerator.is_main_process:
        model_size = sum(p.numel() for p in cfm_model.parameters())
        print(f"Model size: {model_size:,} parameters")
    
    # Setup base dataset
    dataset_cfg = OmegaConf.merge(config.data.train_dataset, config.evaluation.dataset)
    enhance_webdataset_config(dataset_cfg)
    # Override multiple_styles to False since we're generating single style embeddings
    dataset_cfg.multiple_styles = False
    base_dataset = DiffusionWebDataset(**dataset_cfg)
    
    # Create filtered test set dataset
    num_samples = config.evaluation.num_samples
    test_dataset = FilteredTestSetDataset(
        test_set_path=test_set_path,
        diffusion_dataset=base_dataset,
        muq_model=muq_model,
        num_samples=num_samples,
        random_crop_style=config.evaluation.random_crop_style,
        num_style_secs=config.evaluation.num_style_secs,
        use_prompt_style=config.evaluation.use_prompt_style
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        test_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn_with_metadata(batch, base_dataset.custom_collate_fn)
    )
    
    # Prepare models and dataloader with accelerator
    # This automatically distributes the dataloader across GPUs
    cfm_model, vae, dataloader = accelerator.prepare(cfm_model, vae, dataloader)
    
    # Create output directories (each process creates them independently)
    generated_dir = os.path.join(output_dir, "generated_orig")
    generated_trimmed_dir = os.path.join(output_dir, "generated")
    cfm_latents_dir = os.path.join(output_dir, "cfm_latents")
    local_metadata_dir = os.path.join(output_dir, "local_files")
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(generated_trimmed_dir, exist_ok=True)
    os.makedirs(cfm_latents_dir, exist_ok=True)
    os.makedirs(local_metadata_dir, exist_ok=True)
    # Local metadata collection for this process
    local_generation_metadata = []
    
    # Process samples (each GPU processes different samples automatically)
    if accelerator.is_main_process:
        print('Use sampling config: ', config.evaluation.sample_kwargs)
        print(f'Total batches to process: {len(dataloader)}')
    
    # Create process-specific progress bar
    process_desc = f"GPU {accelerator.process_index}/{accelerator.num_processes}"
    
    for i, batch in enumerate(tqdm(dataloader, desc=f"Generating audio [{process_desc}]", 
                                  position=accelerator.process_index, leave=True)):
        if batch is None:
            print(f"Process {accelerator.process_index}: Skipped batch {i} - failed to process")
            continue
            
        # Generate latent
        if config.evaluation.get('sample_kwargs_2', None) is not None:
            sample_kwargs = random.choice([config.evaluation.sample_kwargs, config.evaluation.sample_kwargs_2])
        else:
            sample_kwargs = config.evaluation.sample_kwargs
        latents = generate_latent(accelerator.unwrap_model(cfm_model), batch, sample_kwargs, config.evaluation.negative_style_prompt, config.evaluation.ignore_style)
        
        for j, latents_inf in enumerate(latents):
            for k, latent in enumerate(latents_inf):
                test_metadata = batch['test_metadata'][k]
                sample_id = test_metadata['sample_id']
                original_duration = test_metadata['duration']

                # Save CFM latent directly (shape: seq_len, 64)
                if j == 0:
                    cfm_latent_path = os.path.join(cfm_latents_dir, f"{sample_id}.pt")
                else:
                    cfm_latent_path = os.path.join(cfm_latents_dir, f"{sample_id}_{j}.pt")
                    
                torch.save(latent.cpu(), cfm_latent_path)

                # Decode audio
                latent_for_vae = latent.transpose(0, 1).unsqueeze(0)
                
                # Use chunked decoding if configured (only for DiffRhythm VAE)
                use_chunked = config.evaluation.get('use_chunked_decoding', True)
                if vae_type == 'diffrhythm' and use_chunked:
                    pred_audio = accelerator.unwrap_model(vae).decode(
                        latent_for_vae, 
                        chunked=True, 
                        overlap=config.evaluation.get('chunked_overlap', 32),
                        chunk_size=config.evaluation.get('chunked_size', 128)
                    ).sample.squeeze(0).detach().cpu()
                else:
                    pred_audio = accelerator.unwrap_model(vae).decode(latent_for_vae).sample.squeeze(0).detach().cpu()
                
                pred_audio = normalize_audio(pred_audio)
                
                # Save full version (each GPU saves different files)
                if j == 0:
                    generated_path = os.path.join(generated_dir, f"{sample_id}.mp3")
                else:
                    generated_path = os.path.join(generated_dir, f"{sample_id}_{j}.mp3")
                torchaudio.save(generated_path, pred_audio, 44100, format="mp3")
                
                # Save trimmed version (trim to original duration)
                sample_rate = 44100
                trim_samples = int(original_duration * sample_rate)
                if pred_audio.shape[1] > trim_samples:
                    pred_audio_trimmed = pred_audio[:, :trim_samples]
                else:
                    pred_audio_trimmed = pred_audio
                    
                if j == 0:
                    generated_trimmed_path = os.path.join(generated_trimmed_dir, f"{sample_id}.mp3")
                else:
                    generated_trimmed_path = os.path.join(generated_trimmed_dir, f"{sample_id}_{j}.mp3")
                torchaudio.save(generated_trimmed_path, pred_audio_trimmed, sample_rate, format="mp3")
                
                # Save metadata locally
                sample_metadata = {
                    "sample_id": sample_id,
                    "generation_index": j,
                    "generated_orig_path": generated_path,
                    "generated_audio_path": generated_trimmed_path,
                    "cfm_latent_path": cfm_latent_path,
                    "original_audio_path": test_metadata['audio_path'],
                    "original_duration": original_duration,
                    "generated_duration": pred_audio.shape[1] / sample_rate,
                    "trimmed_duration": pred_audio_trimmed.shape[1] / sample_rate,
                    "lrc_path": test_metadata['lrc_path'],
                    "process_index": accelerator.process_index,  # Track which GPU processed this
                    "sample_kwargs": sample_kwargs,
                    "vae_type": vae_type,
                    "use_chunked_decoding": use_chunked,
                    "latent_type": "cfm_sampled"
                }
                
                # Add batch metadata if available
                for key in ["id", "start_time", "duration_abs", "duration_rel"]:
                    if key in batch:
                        value = batch[key][k]
                        sample_metadata[key] = value.item() if hasattr(value, 'item') else value
                
                local_generation_metadata.append(sample_metadata)

        # Save local metadata after each batch to the same JSONL file (append mode)
        if local_generation_metadata:
            local_metadata_path = os.path.join(local_metadata_dir, f"local_metadata_process_{accelerator.process_index}.jsonl")
            
            # Append new metadata as JSONL (one JSON object per line)
            with open(local_metadata_path, 'a') as f:
                for metadata in local_generation_metadata:
                    json.dump(metadata, f, default=str)
                    f.write('\n')
            
            # Clear local metadata to free memory
            local_generation_metadata.clear()
    
    # Wait for all processes to finish processing
    accelerator.wait_for_everyone()
    
    # Only main process reads and combines all metadata files
    if accelerator.is_main_process:
        print(f"All processes finished. Reading metadata from {accelerator.num_processes} processes...")
        
        # Read all local metadata files and combine them
        all_metadata = []
        for process_idx in range(accelerator.num_processes):
            local_metadata_path = os.path.join(local_metadata_dir, f"local_metadata_process_{process_idx}.jsonl")
            if os.path.exists(local_metadata_path):
                with open(local_metadata_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            metadata = json.loads(line)
                            all_metadata.append(metadata)
        
        # Save combined generation metadata
        generation_path = os.path.join(output_dir, "generation_metadata.json")
        with open(generation_path, 'w') as f:
            json.dump(all_metadata, f, indent=2, default=str)
        # remove local metadata files
        for process_idx in range(accelerator.num_processes):
            local_metadata_path = os.path.join(local_metadata_dir, f"local_metadata_process_{process_idx}.jsonl")
            if os.path.exists(local_metadata_path):
                os.remove(local_metadata_path)
        
        print(f"\n🎉 JAM song generation complete!")
        print(f"- Generated audio (full): {generated_dir}")
        print(f"- Generated audio (trimmed): {generated_trimmed_dir}")
        print(f"- CFM latents: {cfm_latents_dir}")
        print(f"- Metadata: {generation_path}")
        print(f"- Total samples generated: {len(all_metadata)}")
        print(f"- Processed by {accelerator.num_processes} GPUs")
        
        # Print per-process statistics
        process_counts = {}
        for metadata in all_metadata:
            proc_idx = metadata.get('process_index', 'unknown')
            process_counts[proc_idx] = process_counts.get(proc_idx, 0) + 1
        
        print("\nPer-process statistics:")
        for proc_idx, count in sorted(process_counts.items()):
            print(f"  Process {proc_idx}: {count} samples")

if __name__ == "__main__":
    main() 