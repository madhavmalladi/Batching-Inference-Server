#!/usr/bin/env bash
# Downloads the sample images the model tests use, into worker/testdata/:
#   bus.jpg     810x1080 portrait: a bus and several people
#   zidane.jpg  1280x720 landscape: two people
# Both are the standard Ultralytics demo photos. Different shapes on purpose: the
# tests batch them together to check each image is un-letterboxed with its own numbers.
set -euo pipefail

cd "$(dirname "$0")/.."            # worker/
mkdir -p testdata
for name in bus zidane; do
  # -L follows redirects, -f fails on HTTP errors instead of saving an error page.
  curl -fsSL -o "testdata/$name.jpg" "https://ultralytics.com/images/$name.jpg"
  echo "wrote $(pwd)/testdata/$name.jpg ($(wc -c < "testdata/$name.jpg") bytes)"
done
