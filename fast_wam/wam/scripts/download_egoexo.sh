#!/usr/bin/env bash
# Download the Ego-Exo4D subset the multi-view LAM needs.
#
#   bash fast_wam/wam/scripts/download_egoexo.sh /path/to/egoexo4d
#
# PREREQUISITE (one-off, manual): sign the license at
#   https://ego4ddataset.com/egoexo-license/
# You will be emailed AWS access keys. Configure them once:
#   aws configure --profile ego4d      # or: export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
#
# Sizes (from the official docs) -- only the first three are downloaded here:
#   metadata                0.05 GB
#   annotations            10.5  GB   <- 3D hand pose lives here
#   downscaled_takes/448  438.6  GB   (FULL; the scenario filter cuts it to ~150-200 GB)
#   takes (full res)   10,553    GB   NOT downloaded
#   take_vrs           12,301    GB   NOT downloaded
#   take_point_cloud    6,164    GB   NOT downloaded
set -euo pipefail

OUT="${1:?usage: download_egoexo.sh /path/to/egoexo4d}"
mkdir -p "$OUT"
command -v egoexo >/dev/null || { echo "egoexo CLI missing: pip install ego4d"; exit 1; }

echo "===================================================================="
echo "STEP 1/2  metadata + annotations (~11 GB)"
echo "===================================================================="
egoexo -o "$OUT" --parts metadata annotations --yes

echo
echo "===================================================================="
echo "STEP 2/2  downscaled 448px video"
echo "===================================================================="
echo "Full set is 438 GB. Check free space before continuing:"
df -h "$OUT" | tail -1
echo
echo "Option A (recommended) -- filter to manipulation scenarios first."
echo "  Inspect what scenarios exist:"
echo "    python - <<'PY'"
echo "    import json,collections"
echo "    d=json.load(open('$OUT/takes.json'))"
echo "    c=collections.Counter(t.get('parent_task_name') or t.get('task_name') for t in d)"
echo "    [print(f'{n:5d}  {k}') for k,n in c.most_common()]"
echo "    PY"
echo "  Then download only those take uids:"
echo "    egoexo -o $OUT --parts downscaled_takes/448 --uids <uid1> <uid2> ... --yes"
echo
echo "Option B -- everything (438 GB):"
echo "    egoexo -o $OUT --parts downscaled_takes/448 --yes"
echo
read -r -p "Download the FULL 448px set now? [y/N] " ans
if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
  egoexo -o "$OUT" --parts downscaled_takes/448 --yes
else
  echo "Skipped. Use Option A above, then re-run the prepare pipeline."
fi

echo
echo "Next:"
echo "  python fast_wam/wam/prepare/build_index.py --egoexo_root $OUT --inspect"
