#!/usr/bin/env bash
# Score every test clip with the compiled C++ impulse (no Edge Impulse compute needed).
#
#   (in WSL)  bash deployment/cpp-linux/eval.sh [model_dir] [out.csv]
#
# model_dir defaults to deployment/cpp-linux/work/model (the exported C++ library).
# Builds a copy of example-standalone-inferencing-linux whose custom app is batch_eval.cpp,
# then runs it over data/prepared/testing and data/mic/testing.
# Summarise with: python scripts/eval_cpp_results.py <out.csv>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODEL="$(realpath "${1:-$HERE/work/model}")"
OUT="$(realpath -m "${2:-$ROOT/results/cpp_eval.csv}")"
BASE="${EI_APP_DIR:-$HOME/ei-standalone-linux}"
APP="$HOME/ei-eval-$(echo "$MODEL" | md5sum | cut -c1-8)"

if [ ! -d "$BASE/.git" ]; then
  echo "Run deployment/cpp-linux/build.sh first (it clones the standalone app)"; exit 1
fi
if [ ! -d "$APP" ]; then cp -a "$BASE" "$APP"; fi
for d in edge-impulse-sdk model-parameters tflite-model; do
  rm -rf "${APP:?}/$d"
  if [ -d "$MODEL/$d" ]; then cp -r "$MODEL/$d" "$APP/$d"; fi
done
cp "$HERE/batch_eval.cpp" "$APP/source/custom.cpp"

cd "$APP"
APP_CUSTOM=1 make clean >/dev/null 2>&1 || true
if ! APP_CUSTOM=1 make -j"$(nproc)" >/dev/null 2>"$APP/build.err"; then
  APP_CUSTOM=1 make clean >/dev/null 2>&1 || true
  APP_CUSTOM=1 make -j"$(nproc)" CC="gcc -std=gnu17" >/dev/null 2>"$APP/build.err" || { tail -20 "$APP/build.err"; exit 1; }
fi

LIST="$APP/files.txt"
find "$ROOT/data/prepared/testing" "$ROOT/data/mic/testing" -name '*.wav' | sort > "$LIST"
mkdir -p "$(dirname "$OUT")"
./build/custom "@$LIST" > "$OUT"
echo "$(($(wc -l < "$OUT") - 1)) windows from $(wc -l < "$LIST") clips -> $OUT"
