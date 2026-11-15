# STT Test

Minimal on-device test app for the final v0 streaming STT model (ONNX). It runs
the model on the phone and measures **real latency and RTF** — the numbers
that matter for the "fastest small open streaming STT on phone CPU" goal.

This is a test tool, not a product. UI is deliberately plain: sample clips,
mic recording, big readable numbers, and a COPY RESULT button.

## What it does

- Loads `encoder_int8.onnx` + `joint_step_int8.onnx` (v0 final
  checkpoint, 35 epochs) via ONNX Runtime, 1 intra-op thread.
- Log-mel frontend in pure Dart, verified sample-for-sample against the
  Python (torchaudio) pipeline — see `test/mel_frontend_test.dart`.
- Greedy RNN-T decode mirroring `bench_onnx.py` (same joint-step
  recompute semantics, same final-chunk latency definition).
- Inputs: 3 bundled English sample clips (repeatable tests) or mic
  recording (up to 10 s, 16 kHz mono WAV).
- Shows: transcript, final-chunk latency in ms (with frontend / encoder /
  joint breakdown), RTF, model load time. COPY RESULT copies a short
  plain-text summary (transcript + latency + RTF + load time + input +
  versions) for pasting into chat.

## Known limitations (v1)

- **English only for transcription.** The app always passes
  `lang_ids=[0]` (the export itself now switches adapters per lang id —
  v1.0.1's export baked in English only). Latency is
  language-independent, so the timing numbers are still valid.
- **Offline decode.** The exported encoder is the full-sequence encoder,
  not the streaming `forward_chunk` — the app transcribes whole clips.
  The reported latency is the final-chunk figure from the benchmark
  (last 80 ms chunk + finalize), not a live streaming measurement.
- **Int8 model.** WER cost vs fp32 is zero on the English val subset
  (0.4826 both); size 43 + 13 MB instead of 166 + 19 MB.
- Inference runs synchronously on the UI thread (so the measured
  latencies are pure inference time, with no isolate overhead). The UI
  freezes briefly during a run — acceptable for a test tool.

## Build

```sh
flutter pub get
flutter analyze
flutter test        # mel + tokenizer tests; e2e test skips without STT_FP32_DIR
flutter build apk --debug --split-per-abi --target-platform android-arm64
```

CI (`.github/workflows/build-apk.yml`) runs analyze + tests and builds a
debug APK on every push to `main`, uploaded as an artifact.

## Local end-to-end verification

`test/engine_e2e_test.dart` loads the fp32 models and checks the Dart
decode against the Python reference transcript. It needs the fp32 models
and the Linux ORT .so:

```sh
STT_FP32_DIR=$HOME/workspace/stt-project/eval_s1/onnx/s4 \
LD_LIBRARY_PATH=$HOME/.pub-cache/hosted/pub.dev/onnxruntime_plus-1.5.0/linux \
flutter test test/engine_e2e_test.dart
```

(The shipped int8 model is MatMul-only quantized — it has no ConvInteger
nodes, which is what lets it load on ARM phones. The e2e test uses the
fp32 models, not the int8 ones.)

## Versions

- App 1.0.2+3, package `com.aorigoon.stt_test_app`
- Flutter 3.47.9, Java 17 (pinned in CI, same as local)
- onnxruntime_plus 1.5.0 (ORT 1.22.0 on Android), record 7.2.0
