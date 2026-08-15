#!/usr/bin/env bash
# Recreate the public-Drive corpus under the ignored data/raw directory.
set -euo pipefail

mkdir -p data/raw
python scripts/download_public_drive_folder.py \
  --folder-id "1YmYjw444rIYjriDqzZ-fcrHe2Io-F28S" \
  --output data/raw
