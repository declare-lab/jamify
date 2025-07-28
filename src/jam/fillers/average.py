from .. import register_filler

@register_filler('average_repeat')
def average_repeat(phoneme_ids, length, blank_id=0, rng=None):
    if not phoneme_ids or length <=0:
        return [blank_id]*length
    n = len(phoneme_ids)
    base = length // n
    extra = length % n
    out=[]
    for i,p in enumerate(phoneme_ids):
        rep = base + (1 if i < extra else 0)
        out.extend([p]*rep)
    return out[:length]

@register_filler('average_sparse')
def average_sparse(phoneme_ids, length, blank_id=0, rng=None):
    if not phoneme_ids or length <= 0:
        return [blank_id]*length
    n = len(phoneme_ids)
    if n >= length:
        # print(f"Warning: phoneme_ids length {n} is greater than length {length}, using first {length} phonemes")
        return phoneme_ids[:length]
    
    span = length // n
    
    out = [blank_id] * length
    
    for i, p in enumerate(phoneme_ids):
        pos = i * span
        if pos < length:
            out[pos] = p
    
    return out