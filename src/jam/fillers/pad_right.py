from .. import register_filler

@register_filler('pad_right')
def pad_right(phoneme_ids, length, blank_id=0):
    out = [blank_id]*length
    # if len(phoneme_ids) > length:
    #     print(f"pad_right: {len(phoneme_ids)} > {length}")
    for i,p in enumerate(phoneme_ids):
        if i >= length:
            break
        out[i] = p
    return out
