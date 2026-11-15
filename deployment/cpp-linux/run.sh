#!/usr/bin/env bash
# Run the compiled impulse on one window of raw features (same as the guide's
# `./build/custom features-class7.txt` step).
#
#   (in WSL)  bash deployment/cpp-linux/run.sh deployment/cpp-linux/features-abnormal.txt
set -euo pipefail
APP="${EI_APP_DIR:-$HOME/ei-standalone-linux}"
FEATURES="$(realpath "${1:?usage: run.sh <features.txt>}")"
cd "$APP"
./build/custom "$FEATURES"
