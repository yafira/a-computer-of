import json, numpy as np
from collections import defaultdict, Counter
cats = defaultdict(list)
for line in open("electronic-txt/semantic-categories-2026-04-10.jsonl"):
    o = json.loads(line)
    if o['label']:
        cats[o['text'].lower()].extend(o['label'])
counted = {w: Counter(t) for w, t in cats.items()}
cols = ['cold','emotive','round','outlandish','warm','youthful','sharp','organic','academic','tangible']
words = list(counted)
M = np.array([[counted[w][c] for c in cols] for w in words], float)
Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
def vec(**k):
    v = np.zeros(len(cols))
    for a,b in k.items(): v[cols.index(a)] = b
    return v/np.linalg.norm(v)
# slot targets from the notebook (interface uses sharp=1 in the unfiltered version)
slots = {
 "material": vec(organic=2, tangible=2),
 "structure": vec(round=2, warm=1),
 "power": vec(warm=3, organic=1),
 "interface": vec(tangible=2, sharp=1),
 "interface_soft": vec(tangible=2),
 "location": vec(warm=2, round=2),
 "inhabitant": vec(emotive=3, warm=1),
 "promise": vec(emotive=2, youthful=1),
}
pools, used = {}, set()
for name, v in slots.items():
    sims = Mn @ v
    idx = np.argsort(-sims)[:40]
    pools[name] = [words[i] for i in idx]
    used.update(pools[name])
tags = {w: [[t,n] for t,n in counted[w].most_common()] for w in sorted(used)}
json.dump({"pools": pools, "tags": tags}, open("pools.json","w"), separators=(',',':'))
print(len(used), "words")
for k,v in pools.items(): print(k, v[:12])
