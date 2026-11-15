// End-to-end decode test: loads the fp32 ONNX models via the app's
// SttEngine and transcribes sample1.wav. The expected transcript comes from
// the Python reference for the v0 final checkpoint
// (eval_s1/onnx/s4/samples_s4.json).
//
// Why fp32: the test predates the MatMul-only int8 quantization (the
// shipped int8 model has no ConvInteger nodes, so it loads on ARM phones).
// The fp32 run still verifies the whole Dart decode path (encode + greedy
// loop + tokenizer) end to end.
//
// Run with the fp32 model dir (skipped if unset):
//   STT_FP32_DIR=$HOME/workspace/stt-project/eval_s1/onnx/s4 \
//   LD_LIBRARY_PATH=$HOME/.pub-cache/hosted/pub.dev/onnxruntime_plus-1.5.0/linux \
//   flutter test test/engine_e2e_test.dart
// (CI does NOT run this test — the debug APK build is the CI gate.)

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:stt_test_app/stt_engine.dart';
import 'package:stt_test_app/wav_io.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('e2e fp32 decode matches Python reference', () async {
    final fp32Dir = Platform.environment['STT_FP32_DIR'];
    if (fp32Dir == null || fp32Dir.isEmpty) {
      markTestSkipped('STT_FP32_DIR not set (local verification only)');
      return;
    }
    final encBytes = File('$fp32Dir/encoder.onnx').readAsBytesSync();
    final jointBytes = File('$fp32Dir/joint_step.onnx').readAsBytesSync();
    final vocabStr = File('assets/models/vocab.json').readAsStringSync();

    final engine = SttEngine();
    await engine.loadFromBytes(encBytes, jointBytes, vocabStr);
    expect(engine.isLoaded, isTrue);
    expect(engine.vocabSize, 3354);

    final bytes = File('test/fixtures/sample1.wav').readAsBytesSync();
    final wav = parseWav(bytes);
    final result = engine.transcribe(wav.samples, 'sample1.wav');

    // ignore: avoid_print
    print('transcript: "${result.transcript}"');
    // ignore: avoid_print
    print('latency: ${result.latencyMs.toStringAsFixed(1)} ms, '
        'RTF: ${result.rtf.toStringAsFixed(2)}, '
        'load: ${result.modelLoadMs.toStringAsFixed(0)} ms');
    // ignore: avoid_print
    print('copy text:\n${result.toCopyText('1.0.2', 'v0 final test')}');

    // Python reference for this clip (eval_s1/onnx/s4/samples_s4.json).
    expect(result.transcript, 'stuping out of his heiding place');
    expect(result.latencyMs, greaterThan(0));
    expect(result.rtf, greaterThan(0));
    engine.dispose();
  }, timeout: const Timeout(Duration(minutes: 5)));
}
