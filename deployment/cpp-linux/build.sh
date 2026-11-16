#!/usr/bin/env bash
# Build the exported Edge Impulse C++ library into a Linux executable, exactly as the reference
# guide does ("Deploying the Impulse"): example-standalone-inferencing-linux + APP_CUSTOM=1 make.
#
#   (in WSL)  bash deployment/cpp-linux/build.sh
#
# Expects the unpacked C++ library in deployment/cpp-linux/work/model/ (created by
#   python scripts/ei_pipeline.py --skip-upload --skip-retrain --targets cpp ).
# Builds in the WSL home folder (much faster than /mnt/d) and leaves ./build/custom there.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$HERE/work/model"
APP="${EI_APP_DIR:-$HOME/ei-standalone-linux}"

if [ ! -d "$MODEL/edge-impulse-sdk" ]; then
  echo "Missing $MODEL/edge-impulse-sdk"
  echo "Run first: python scripts/ei_pipeline.py --skip-upload --skip-retrain --targets cpp"
  exit 1
fi

if [ ! -d "$APP/.git" ]; then
  git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/edgeimpulse/example-standalone-inferencing-linux "$APP"
fi

# The guide: "Every time it is necessary to test another model ... delete these three folders
# and replace them with the new ones."
for d in edge-impulse-sdk model-parameters tflite-model; do
  rm -rf "${APP:?}/$d"
  if [ -d "$MODEL/$d" ]; then cp -r "$MODEL/$d" "$APP/$d"; fi
done

cd "$APP"
APP_CUSTOM=1 make clean >/dev/null 2>&1 || true
if ! APP_CUSTOM=1 make -j"$(nproc)"; then
  echo ">>> Build failed with the default flags; retrying with C17 (GCC 15 compiles C as C23 by default)"
  APP_CUSTOM=1 make clean >/dev/null 2>&1 || true
  APP_CUSTOM=1 make -j"$(nproc)" CC="gcc -std=gnu17"
fi
echo "Built: $APP/build/custom"
