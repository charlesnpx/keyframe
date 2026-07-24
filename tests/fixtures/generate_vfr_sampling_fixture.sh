#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output_path=${1:-"$script_dir/vfr-sampling.mp4"}

ffmpeg \
  -hide_banner \
  -loglevel error \
  -y \
  -f lavfi \
  -i "testsrc2=size=64x48:rate=30:duration=2" \
  -vf "select='eq(n,0)+eq(n,2)+eq(n,7)+eq(n,15)+eq(n,16)+eq(n,30)+eq(n,45)+eq(n,59)'" \
  -fps_mode vfr \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -map_metadata -1 \
  "$output_path"
