#!/usr/bin/env bash
# Pre-transcode EgoDex 1080p mpeg4 -> 240p all-intra h264 for fast random-access
# decoding during LAM training. Center-crops to 4:3 (matching the dataset's
# _center_crop_resize) then scales to 320x240, with GOP=1 (every frame a keyframe).
# HDF5 pose files are symlinked alongside so the dataset finds labels unchanged.
#
#   bash transcode_240p.sh           # run transcode (parallel)
#   bash transcode_240p.sh --check   # verify count + a few frame-count matches
set -uo pipefail

SRC=/home/xuan/embodied-ai/data/egodex/test
DST=/home/xuan/embodied-ai/data/egodex/test_240p
JOBS=24

if [[ "${1:-}" == "--check" ]]; then
  echo "src mp4: $(find $SRC -name '*.mp4' | wc -l)   dst mp4: $(find $DST -name '*.mp4' 2>/dev/null | wc -l)"
  echo "dst size: $(du -sh $DST 2>/dev/null | cut -f1)"
  echo "frame-count match (src vs dst) on 5 random videos:"
  for rel in $(cd $SRC && find . -name '*.mp4' | shuf | head -5); do
    s=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$SRC/$rel")
    d=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$DST/$rel" 2>/dev/null)
    echo "  $rel : src=$s dst=$d $([ "$s" == "$d" ] && echo OK || echo MISMATCH)"
  done
  exit 0
fi

transcode_one() {
  src="$1"
  rel="${src#"$2"/}"                  # task/idx.mp4
  dst="$3/$rel"
  mkdir -p "$(dirname "$dst")"
  if [[ ! -f "$dst" ]]; then
    ffmpeg -y -loglevel error -i "$src" \
      -vf "crop=w='if(gte(iw/ih,4/3),ih*4/3,iw)':h='if(gte(iw/ih,4/3),ih,iw*3/4)',scale=320:240" \
      -c:v libx264 -preset veryfast -crf 20 -g 1 -an -sn -vsync 0 "$dst" \
      </dev/null 2>/dev/null || { echo "FAIL $rel"; return; }
  fi
  # symlink the sibling HDF5 so the dataset finds pose labels
  h5="${src%.mp4}.hdf5"
  [[ -f "$h5" ]] && ln -sf "$h5" "${dst%.mp4}.hdf5"
}
export -f transcode_one

echo "Transcoding $(find $SRC -name '*.mp4' | wc -l) videos -> $DST  (JOBS=$JOBS)"
find "$SRC" -name '*.mp4' | xargs -P "$JOBS" -I {} bash -c 'transcode_one "$@"' _ {} "$SRC" "$DST"
echo "Done. dst mp4: $(find $DST -name '*.mp4' | wc -l)   size: $(du -sh $DST | cut -f1)"
