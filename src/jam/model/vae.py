import torch
import torch.nn as nn
from diffusers.models import AutoencoderOobleck
import torchaudio
from huggingface_hub import hf_hub_download

def vae_gaussian_sample(pre_bottleneck_latents, chunk_dim=1):
    mean, std = pre_bottleneck_latents.chunk(2, dim=chunk_dim)
    x = torch.randn_like(mean) * std + mean
    return x

class VAEBottleneck(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

class StableAudioOpenVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.vae = AutoencoderOobleck.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            subfolder="vae",
            # torch_dtype=torch.float16
        ).eval()
        self.sr = self.vae.config.sampling_rate
        
    @torch.inference_mode()
    def encode_pre_bottleneck(self, x, sr):
        if sr != self.sr:
            x = torchaudio.functional.resample(x, sr, self.sr)
        if x.shape[0] == 1:
            x = torch.cat([x, x], dim=0)
        # x = x.to(dtype=torch.float16)

        latent_dist = self.vae.encode(x.unsqueeze(0)).latent_dist
        pre_bottleneck_latents = torch.cat([latent_dist.mean, latent_dist.std], dim=1)
        return pre_bottleneck_latents.squeeze(0)

    @torch.inference_mode()
    def bottleneck(self, pre_bottleneck_latents):
        return vae_gaussian_sample(pre_bottleneck_latents)

    @torch.inference_mode()
    def decode(self, x):
        return self.vae.decode(x)


class DiffRhythmVAEOutput:
    """Wrapper to make DiffRhythm VAE output compatible with StableAudio interface"""
    def __init__(self, sample):
        self.sample = sample


class DiffRhythmVAE(nn.Module):
    """DiffRhythm VAE implementation based on their actual inference code"""
    
    def __init__(self, device="cuda", repo_id="ASLP-lab/DiffRhythm-vae"):
        super().__init__()
        self.device = device
        self.repo_id = repo_id
        self.vae = None
        self._load_model()
        
    def _load_model(self):
        """Download and load the DiffRhythm VAE model (exact same as their implementation)"""
        vae_ckpt_path = hf_hub_download(
            repo_id=self.repo_id,
            filename="vae_model.pt",
            cache_dir="./pretrained",
        )
        self.vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(self.device)
        self.vae.eval()
    
    @torch.inference_mode()
    def decode(self, latents, chunked=False, overlap=32, chunk_size=128):
        """
        Decode latents to audio with StableAudio-compatible interface
        
        Args:
            latents: Input latents [batch, channels, time]
            chunked: Whether to use chunked decoding
            overlap: Overlap for chunked decoding
            chunk_size: Chunk size for chunked decoding
            
        Returns:
            DiffRhythmVAEOutput with .sample attribute containing decoded audio
        """
        if not chunked:
            # Simple decoding - same as DiffRhythm
            decoded_audio = self.vae.decode_export(latents)
        else:
            # Chunked decoding - exact same implementation as DiffRhythm
            decoded_audio = self._decode_chunked(latents, overlap, chunk_size)
        
        # Return in StableAudio-compatible format
        return DiffRhythmVAEOutput(decoded_audio)
    
    def _decode_chunked(self, latents, overlap=32, chunk_size=128):
        """Chunked decoding implementation (exact copy from DiffRhythm)"""
        downsampling_ratio = 2048
        io_channels = 2
        
        hop_size = chunk_size - overlap
        total_size = latents.shape[2]
        batch_size = latents.shape[0]
        chunks = []
        i = 0
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunk = latents[:, :, i : i + chunk_size]
            chunks.append(chunk)
        if i + chunk_size != total_size:
            # Final chunk
            chunk = latents[:, :, -chunk_size:]
            chunks.append(chunk)
        chunks = torch.stack(chunks)
        num_chunks = chunks.shape[0]
        # samples_per_latent is just the downsampling ratio
        samples_per_latent = downsampling_ratio
        # Create an empty waveform, we will populate it with chunks as decode them
        y_size = total_size * samples_per_latent
        y_final = torch.zeros((batch_size, io_channels, y_size)).to(latents.device)
        for i in range(num_chunks):
            x_chunk = chunks[i, :]
            # decode the chunk
            y_chunk = self.vae.decode_export(x_chunk)
            # figure out where to put the audio along the time domain
            if i == num_chunks - 1:
                # final chunk always goes at the end
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent
            #  remove the edges of the overlaps
            ol = (overlap // 2) * samples_per_latent
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if i > 0:
                # no overlap for the start of the first chunk
                t_start += ol
                chunk_start += ol
            if i < num_chunks - 1:
                # no overlap for the end of the last chunk
                t_end -= ol
                chunk_end -= ol
            # paste the chunked audio into our y_final output audio
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
        return y_final