#!/bin/sh
# Reassemble the split encoder. GitHub's API rejects single files above ~35 MB
# (HTTP 422), so the 43 MB encoder is stored as .part0/.part1. CI runs this
# after checkout; run it once after cloning to build locally.
set -e
cd "$(dirname "$0")/.."
cat assets/models/encoder_int8.onnx.part0 assets/models/encoder_int8.onnx.part1 > assets/models/encoder_int8.onnx
