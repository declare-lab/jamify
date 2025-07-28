import math
import random
import json
import traceback
import torch
import webdataset as wds
from jam.model.vae import vae_gaussian_sample
from . import get_filler
from .tokenizer import create_phoneme_tokenizer
from dp.phonemizer import Phonemizer # type: ignore
from dp.english import english_to_ipa # type: ignore
import braceexpand

def enhance_webdataset_config(config):
    pattern = config.pop("pattern")
    urls = list(braceexpand.braceexpand(pattern))
    config["urls"] = urls
    return None # Implict return None as this is an mutating function

class DiffusionWebDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        urls: str,
        id_list_jsonl: str = None,
        max_frames: int = 2048,
        sampling_rate: int = 44100,
        downsample_rate: int = 2048,
        # precision: str = 'fp16',
        shuffle: bool = True,
        num_samples: int = -1,
        multiple_styles: bool = False,
        filler: str = "pad_right",
        silence_latent_path: str = None,
        resample_by_duration_threshold: float = None,
        ignore_by_duration_threshold: float = None,
        tokenizer_path: str = None,
        lrc_upsample_factor: int = 4,
        return_word_info: bool = False,
        always_crop_from_beginning: bool = False,
        always_use_style_index: int = None,
        phonemizer_checkpoint: str | None = None,
    ):
        """
        DiffusionDataset using WebDataset format.
        Assumes each sample contains latent.pt, style.pt, and json with 'word' field containing phonemes.
        
        Args:
            urls: WebDataset URLs pattern
            id_list_jsonl: Optional path to a jsonl file containing song IDs and durations
            max_frames: Maximum number of frames
            sampling_rate: Audio sampling rate
            downsample_rate: Downsample rate for frame calculation
            precision: Precision for tensors ('fp16', 'bf16', 'fp32')
            shuffle: Whether to shuffle the dataset
            num_samples: Number of samples to use (-1 for all)
            filler: Filling strategy for phonemes ('pad_right', 'average_repeat', 'random_duration')
            silence_latent_path: Optional path to silence latent for padding instead of zeros
            resample_by_duration_threshold: Optional threshold for resampling based on duration
            ignore_by_duration_threshold: Optional threshold for ignoring samples based on duration
            tokenizer_path: Optional path to phoneme tokenizer
            lrc_upsample_factor: Factor by which lrc tensor is longer than latent tensor (default: 4)
            return_word_info: Whether to return word-level information
            always_crop_from_beginning: Whether to always crop from the beginning
            always_use_style_index: Optional fixed style index to use
            phonemizer_checkpoint: Optional path to phonemizer checkpoint for computing phonemes when not available in word data
        """
        self.urls = urls
        self.max_frames = max_frames
        self.sampling_rate = sampling_rate
        self.downsample_rate = downsample_rate
        self.max_secs = max_frames / (sampling_rate / downsample_rate)
        self.shuffle = shuffle
        self.num_samples = num_samples
        self.multiple_styles = multiple_styles
        self.lrc_upsample_factor = lrc_upsample_factor
        self.phoneme_tokenizer = create_phoneme_tokenizer(tokenizer_path)
        self.filler = get_filler(filler)
        self.resample_by_duration_threshold = resample_by_duration_threshold
        if self.resample_by_duration_threshold is not None:
            print(f"Resample by duration threshold is set to {self.resample_by_duration_threshold}, this will be used to resample the dataset based on the duration threshold")
        self.ignore_by_duration_threshold = ignore_by_duration_threshold
        if self.ignore_by_duration_threshold is not None:
            print(f"Ignore by duration threshold is set to {self.ignore_by_duration_threshold}, this will be used to ignore the samples based on the duration threshold")
        self.return_word_info = return_word_info
        self.always_crop_from_beginning = always_crop_from_beginning
        self.always_use_style_index = always_use_style_index
        self.pad_phoneme_token_id = 62
        self.no_phoneme_token_id = 63

        # Load silence latent if provided
        self.silence_latent_path = silence_latent_path
        self.silence_latent = torch.load(silence_latent_path, weights_only=True)
        
        # Initialize phonemizer if checkpoint is provided
        self.phonemizer = None
        if phonemizer_checkpoint is not None:
            ph = Phonemizer.from_checkpoint(phonemizer_checkpoint, device="cpu")
            self.phonemizer = lambda text: english_to_ipa(text, lambda x: ph(x, lang="en_us"))
            print(f"Phonemizer loaded from {phonemizer_checkpoint}")
        
        # Load ID list if provided
        self.id_to_duration = {}
        if id_list_jsonl:
            with open(id_list_jsonl, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    self.id_to_duration[str(item["id"])] = item["duration"]

    def __iter__(self):
        # Create fresh pipeline for each iterator
        dataset = wds.WebDataset(self.urls, nodesplitter=lambda urls: urls, handler=wds.warn_and_continue, resampled=True, shardshuffle=False)
        
        if self.shuffle:
            dataset = dataset.shuffle(1000)
            
        # Process each sample
        dataset = dataset.decode(wds.handle_extension('pt', wds.torch_loads)).to_tuple('__key__', 'latent.pt', 'style.pt', 'json')
        
        # Filter out samples with empty words
        # dataset = dataset.select(lambda sample: len(sample[3].get("word", [])) > 0)
        
        # Filter by ID list if provided
        if self.id_to_duration:
            dataset = dataset.select(lambda sample: sample[0] in self.id_to_duration)

        if self.ignore_by_duration_threshold is not None:
            dataset = dataset.select(lambda sample: self.id_to_duration[sample[0]] > self.ignore_by_duration_threshold)

        if self.resample_by_duration_threshold is not None:
            # Resample based on duration threshold
            dataset = dataset.select(lambda sample: 
                random.random() < (self.id_to_duration[sample[0]] / self.resample_by_duration_threshold) ** 1.2
            )
        
        dataset = dataset.map(self.process_sample_safely)
        # Filter out None values from failed processing
        dataset = dataset.select(lambda x: x is not None)
        
        return iter(dataset)
    
    def process_sample_safely(self, sample):
        try:
            return self.process_sample(sample)
        except Exception as e:
            song_id = sample[0] if len(sample) > 0 else "unknown"
            print(f"Error processing sample {song_id}: {str(e)}")
            print(traceback.format_exc())
            return None  # Return None for failed samples

    def process_sample(self, sample):
        song_id, latent, style, json_data = sample
        
        # Use json['word'] for word-level phonemes
        # words = json_data.get("word", [])
        # if not words:
        #     raise ValueError("No words found in json data")
        
        words = json_data["word"]
        if not words:
            # print(f"No words found in json data for {song_id}, {words}")
            pass
        
        # Process latent and prompt
        if self.multiple_styles:
            if self.always_use_style_index is not None:
                style = style[self.always_use_style_index]
            else:
                style = style[random.randint(0, style.shape[0] - 1)]
        
        # Random cropping logic - calculate start and end frames
        max_start_frame = max(0, latent.shape[-1] - self.max_frames)
        if self.always_crop_from_beginning:
            start_frame = 0
        else:
            start_frame = random.randint(0, max_start_frame)
        end_frame = min(start_frame + self.max_frames, latent.shape[-1])
        
        # Calculate corresponding time boundaries
        crop_start_time = start_frame * self.downsample_rate / self.sampling_rate
        crop_end_time = end_frame * self.downsample_rate / self.sampling_rate
        normalized_start_time = start_frame / latent.shape[-1]
        normalized_duration_abs = math.log1p(crop_end_time - crop_start_time) / math.log1p(500)
        normalized_duration_rel = (crop_end_time - crop_start_time) / self.max_secs

        # Trim latent to exact frames right away
        latent = vae_gaussian_sample(latent.transpose(0, 1)).transpose(0, 1)
        latent = latent[:, start_frame:end_frame]
        
        
        # Filter words that overlap with our time segment
        selected_words = [w for w in words if w["end"] > crop_start_time and w["start"] < crop_end_time]
        
        if not selected_words:
            pass
            # print(f"No words found in the selected time segment for {song_id} from {crop_start_time}s to {crop_end_time}s")
        
        # Create lrc tensor using word-level filling approach
        # Make lrc tensor longer by lrc_upsample_factor
        lrc_frames = self.max_frames * self.lrc_upsample_factor
        lrc = torch.full((lrc_frames,), self.no_phoneme_token_id, dtype=torch.long)
        
        # Adjusted downsampling rate for lrc to make it longer
        lrc_downsample_rate = self.downsample_rate / self.lrc_upsample_factor
        word_info = []
        
        for word in selected_words:
            word_start = word["start"] - crop_start_time
            word_end = word["end"] - crop_start_time
            
            # Get phoneme from word data or compute it using phonemizer
            phoneme = word.get("phoneme", None)
            if phoneme is None:
                assert self.phonemizer is not None, "Phonemizer is not loaded, please provide a phonemizer checkpoint"
                # Fallback: compute phoneme from word text using phonemizer
                word_text = word["word"]
                phoneme = self.phonemizer(word_text)
            
            # Convert to frame indices (now relative to 0) with adjusted rate for lrc
            start_frame_idx = int(word_start * self.sampling_rate / lrc_downsample_rate)
            end_frame_idx = int(word_end * self.sampling_rate / lrc_downsample_rate)
            
            # Simple clamp to frame boundaries
            start_frame_idx = max(0, start_frame_idx)
            end_frame_idx = min(lrc_frames, end_frame_idx)
            frame_length = end_frame_idx - start_frame_idx

            if frame_length <= 0:
                continue
            
            # Convert phoneme to token IDs
            if phoneme:
                if not phoneme.endswith('_'):
                    phoneme += '_'
                tokens = self.phoneme_tokenizer(phoneme, language="en_us")[1:-1]
            else:
                tokens = []

            if self.return_word_info:
                word_info.append({
                    "start_time": word_start,
                    "end_time": word_end,
                    "phoneme": phoneme,
                    "word": word["word"],
                    "start_frame_idx": start_frame_idx,
                    "end_frame_idx": end_frame_idx,
                    "frame_length": frame_length,
                    "tokens": tokens
                })
            
            
            # Use filler to distribute tokens across the frame span
            if tokens:
                filled_tokens = self.filler(tokens, frame_length, blank_id=self.pad_phoneme_token_id)
                lrc[start_frame_idx:end_frame_idx] = torch.tensor(filled_tokens, dtype=torch.long)
        
        result = {
            'prompt': style,
            'lrc': lrc,
            'latent': latent,
            'start_time': normalized_start_time,
            'duration_abs': normalized_duration_abs,
            'duration_rel': normalized_duration_rel
        }
        if self.return_word_info:
            result['word_info'] = word_info
        return result

    def custom_collate_fn(self, batch):
        latent_list = [item['latent'] for item in batch]
        prompt_list = [item['prompt'] for item in batch]
        lrc_list = [item['lrc'] for item in batch]
        start_time_list = [item['start_time'] for item in batch]
        duration_abs_list = [item['duration_abs'] for item in batch]
        duration_rel_list = [item['duration_rel'] for item in batch]
        word_info_list = [item['word_info'] for item in batch] if self.return_word_info else None

        latent_lengths = torch.LongTensor([self.max_frames for latent in latent_list])

        padded_latent_list = []
        for latent in latent_list:
            pad_length = self.max_frames - latent.shape[-1]
            if pad_length > 0:
                silence_padding = self.silence_latent.repeat(pad_length, 1).transpose(0, 1)
                padded_latent = torch.cat([latent, silence_padding], dim=-1)
            else:
                padded_latent = latent
            padded_latent_list.append(padded_latent)

        padded_start_time_list = []
        for start_time in start_time_list:
            padded_start_time = start_time
            padded_start_time_list.append(padded_start_time)

        prompt_tensor = torch.stack(prompt_list)
        lrc_tensor = torch.stack(lrc_list)
        latent_tensor = torch.stack(padded_latent_list)
        start_time_tensor = torch.tensor(padded_start_time_list)
        duration_abs_tensor = torch.tensor(duration_abs_list)
        duration_rel_tensor = torch.tensor(duration_rel_list)

        result = {'prompt': prompt_tensor, 'lrc': lrc_tensor, 'latent': latent_tensor, \
                "latent_lengths": latent_lengths, \
                "start_time": start_time_tensor, "duration_abs": duration_abs_tensor, "duration_rel": duration_rel_tensor}
        if self.return_word_info:
            result['word_info'] = word_info_list
        return result
