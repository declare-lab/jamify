import torch
from dp.preprocessing.text import Preprocessor, SequenceTokenizer


def create_phoneme_tokenizer(ckpt_path: str) -> SequenceTokenizer:
    with torch.serialization.safe_globals({"dp.preprocessing.text.Preprocessor": Preprocessor}):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    base_tok: SequenceTokenizer = ckpt["preprocessor"].phoneme_tokenizer

    special = {'_','<end>'} | {f'<{lang}>' for lang in base_tok.languages}
    base_symbols = [s for s in base_tok.idx_to_token.values() if s not in special]

    extra_symbols = list(".,?!:;")
    new_symbols = base_symbols + [s for s in extra_symbols if s not in base_symbols]

    punct_tok = SequenceTokenizer(
        symbols=new_symbols,
        languages=base_tok.languages,
        char_repeats=1,
        lowercase=False,
        append_start_end=True,
    )

    return punct_tok
