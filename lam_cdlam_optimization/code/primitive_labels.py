"""CD-LAM-style 12-way coarse action-primitive labels for EgoDex verbs.

Faithful to CD-LAM Appendix B: the canonical label space is 12 primitives, and —
deliberately — several OPPOSITE verb pairs are merged into ONE primitive
(pick–place, insert–remove, stack–unstack, scoop–dump, wash–rinse), while
open/close and turn on/turn off stay separate. Verbs are clustered into the
canonical space by semantic similarity (synonyms/near-synonyms merged); verbs
with no reliable primitive stay UNLABELED and join no contrastive pair.

This merging is the point of the repro: a contrastive loss trained on these
labels PULLS several opposite-action pairs TOGETHER by construction, which is
the mechanism behind CD-LAM's residual opposite-action confusion — the failure
mode our direction losses target. Do not "fix" the merging here.

Judgment calls (documented, conservative):
  - assemble/disassemble -> insert_remove: EgoDex furniture-bench assembly is
    repeated part insertion/removal; dropping them would lose ~18% of episodes.
  - screw/unscrew, charge/uncharge (plug/unplug), thread, slot, sleeve ->
    insert_remove: fastener/connector insertion motions.
  - clean/wipe/dry -> wash_rinse: surface-cleaning near-synonyms.
  - push/pull, fold/unfold, tie/untie, zip/unzip, wrap/unwrap etc. have NO
    canonical primitive in CD-LAM's 12 -> unlabeled. This is faithful, and it
    is exactly the fine-grained gap our method targets.

Coverage on the val-split verb map (263 episodes): ~73% labeled.
Self-check: python code/primitive_labels.py
"""

from __future__ import annotations

PRIMITIVES = [
    "pick_place", "insert_remove", "stack_unstack", "scoop_dump",
    "open", "close", "turn_on", "turn_off", "wash_rinse", "cut", "stir", "pour",
]

VERB_TO_PRIMITIVE: dict[str, str] = {
    # pick–place (merged opposite pair)
    "pick": "pick_place", "pick up": "pick_place", "put": "pick_place",
    "put down": "pick_place", "place": "pick_place", "grab": "pick_place",
    "grasp": "pick_place", "take": "pick_place", "collect": "pick_place",
    "gather": "pick_place", "add": "pick_place",
    # insert–remove (merged opposite pair)
    "insert": "insert_remove", "remove": "insert_remove",
    "assemble": "insert_remove", "disassemble": "insert_remove",
    "screw": "insert_remove", "unscrew": "insert_remove",
    "charge": "insert_remove", "uncharge": "insert_remove",
    "thread": "insert_remove", "slot": "insert_remove", "sleeve": "insert_remove",
    "plug": "insert_remove", "unplug": "insert_remove",
    # stack–unstack (merged opposite pair)
    "stack": "stack_unstack", "unstack": "stack_unstack",
    # scoop–dump (merged opposite pair)
    "scoop": "scoop_dump", "dump": "scoop_dump",
    # open / close (SEPARATE primitives in CD-LAM)
    "open": "open", "close": "close",
    # turn on / turn off (SEPARATE; not present in val vocab, kept for train vocab)
    "turn on": "turn_on", "switch on": "turn_on",
    "turn off": "turn_off", "switch off": "turn_off",
    # wash–rinse (merged near-synonym cluster)
    "wash": "wash_rinse", "rinse": "wash_rinse", "clean": "wash_rinse",
    "wipe": "wash_rinse", "dry": "wash_rinse",
    # cut / stir / pour (singletons; cut & stir absent from val vocab, kept for train)
    "cut": "cut", "chop": "cut", "slice": "cut",
    "stir": "stir", "mix": "stir", "whisk": "stir",
    "pour": "pour", "dispense": "pour",
}


def _norm_verb(v: object) -> str:
    return str(v).strip().lower().replace("_", " ")


def primitive_of(verb: object) -> str | None:
    """Map a raw verb string to its canonical primitive, or None (unlabeled)."""
    if verb is None:
        return None
    v = _norm_verb(verb)
    if v in ("unknown", "none", ""):
        return None
    return VERB_TO_PRIMITIVE.get(v)


if __name__ == "__main__":
    import sys
    from collections import Counter
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset import EgoDexDataset

    ds = EgoDexDataset(
        data_root="/home/xuan/embodied-ai/data/egodex/test_240p",
        img_h=240, img_w=320, downsample_factors=(1,), split="val",
        val_ratio=0.1, seed=42, load_verbs=True, load_actions=False,
    )
    verbs = [v for v in ds._verb_map.values() if v is not None]
    prim = Counter()
    unlabeled = Counter()
    for v in verbs:
        p = primitive_of(v)
        if p is None:
            unlabeled[_norm_verb(v)] += 1
        else:
            prim[p] += 1
    n = len(verbs)
    labeled = sum(prim.values())
    print(f"episodes with verb: {n} | labeled: {labeled} ({labeled/n:.1%}) | primitives used: {len(prim)}/12")
    for p in PRIMITIVES:
        print(f"  {p:15s} {prim.get(p, 0):4d}")
    print("top unlabeled verbs:", unlabeled.most_common(12))
