import random
from .. import register_filler

@register_filler('random_duration')
def random_duration(phoneme_ids, length, blank_id=0, rng=None):
    if rng is None:
        rng = random
    if not phoneme_ids or length <= 0:
        return [blank_id]*length
    n = len(phoneme_ids)
    if n >= length:
        return phoneme_ids[:length]
    durations = [1]*n
    remaining = length - n
    if remaining > 0:
        weights = [rng.random() for _ in range(n)]
        total = sum(weights)
        alloc = [int(round(w/total*remaining)) for w in weights]
        diff = remaining - sum(alloc)
        for i in range(abs(diff)):
            alloc[i % n] += 1 if diff > 0 else -1
        for i in range(n):
            durations[i] += alloc[i]
    out=[]
    for p,d in zip(phoneme_ids,durations):
        out.extend([p]*d)
    return out[:length]
