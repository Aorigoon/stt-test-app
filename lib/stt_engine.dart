// On-device STT engine: ONNX Runtime (onnxruntime_plus, FFI) + greedy decode.
// Mirrors eval_s1/onnx/bench_onnx.py exactly:
//   - encoder.onnx: mel (1, T, 80) + mel_lengths + lang_ids -> enc frames
//   - joint_step.onnx: (enc_frame, token, h, c) -> (logits, h_out, c_out);
//     the greedy loop re-feeds the last token from its saved PRE-token state
//     (deterministic recompute, same as the Python loop).
//   - 1 intra-op thread (fastest in the VM benchmark), graph opt level all.
//   - lang_ids is always 0 (English): the export baked in the English adapter.

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime_plus/onnxruntime_plus.dart';

import 'mel_frontend.dart';
import 'tokenizer.dart';

/// One completed test run: transcript + measured timings.
class SttResult {
  /// Transcribed text (may be empty if nothing was recognised).
  final String transcript;

  /// What was transcribed: e.g. "sample1.wav" or "Mic recording".
  final String inputLabel;

  /// Audio duration in seconds.
  final double durationSec;

  /// Moonshine-style final-chunk latency in ms (last 80 ms chunk + finalize).
  final double latencyMs;

  /// Breakdown of [latencyMs]: frontend / encoder / joint, in ms.
  final double frontendMs;
  final double encoderMs;
  final double jointMs;

  /// Real-time factor of the full offline decode (total time / audio time).
  final double rtf;

  /// Model load time in ms (both sessions + vocab).
  final double modelLoadMs;

  const SttResult({
    required this.transcript,
    required this.inputLabel,
    required this.durationSec,
    required this.latencyMs,
    required this.frontendMs,
    required this.encoderMs,
    required this.jointMs,
    required this.rtf,
    required this.modelLoadMs,
  });

  /// Short plain-text summary, one line per value — for the COPY RESULT button.
  String toCopyText(String appVersion, String modelVersion) {
    String ms(double v) => '${v.toStringAsFixed(1)} ms';
    return 'STT Test $appVersion — model $modelVersion\n'
        'Input: $inputLabel (${durationSec.toStringAsFixed(1)}s)\n'
        'Transcript: ${transcript.isEmpty ? '(empty)' : transcript}\n'
        'Latency: ${ms(latencyMs)}\n'
        'RTF: ${rtf.toStringAsFixed(2)}\n'
        'Model load: ${ms(modelLoadMs)}';
  }
}

class SttEngine {
  static const int dModel = 320;
  static const int predictorHidden = 320;
  static const int predictorLayers = 2;
  static const int maxSymbolsPerFrame = 10;
  static const int blankId = 0;
  static const int chunkSamples = 1280; // 80 ms at 16 kHz

  static const String encoderAsset = 'assets/models/encoder_int8.onnx';
  static const String jointAsset = 'assets/models/joint_step_int8.onnx';
  static const String vocabAsset = 'assets/models/vocab.json';

  OrtSession? _encSession;
  OrtSession? _jointSession;
  CharTokenizer? _tokenizer;
  double modelLoadMs = 0.0;

  bool get isLoaded =>
      _encSession != null && _jointSession != null && _tokenizer != null;

  int get vocabSize => _tokenizer?.vocabSize ?? 0;

  /// Load both ONNX sessions + vocab from the bundled assets.
  /// Times the whole load (asset read + session creation).
  Future<void> load() async {
    final sw = Stopwatch()..start();
    final encBytes = await rootBundle.load(encoderAsset);
    final jointBytes = await rootBundle.load(jointAsset);
    final vocabStr = await rootBundle.loadString(vocabAsset);
    _initSessions(
      encBytes.buffer
          .asUint8List(encBytes.offsetInBytes, encBytes.lengthInBytes),
      jointBytes.buffer
          .asUint8List(jointBytes.offsetInBytes, jointBytes.lengthInBytes),
      vocabStr,
    );
    modelLoadMs = sw.elapsedMicroseconds / 1000.0;
  }

  /// Load sessions from raw bytes (used by the local e2e test with fp32
  /// models, since the Linux test ORT build lacks the ConvInteger kernel
  /// the int8 model needs — see test/engine_e2e_test.dart).
  Future<void> loadFromBytes(
      Uint8List enc, Uint8List joint, String vocabStr) async {
    final sw = Stopwatch()..start();
    _initSessions(enc, joint, vocabStr);
    modelLoadMs = sw.elapsedMicroseconds / 1000.0;
  }

  void _initSessions(Uint8List enc, Uint8List joint, String vocabStr) {
    OrtEnv.instance.init();
    _tokenizer = CharTokenizer.fromJson(vocabStr);

    final options = OrtSessionOptions();
    options.setIntraOpNumThreads(1);
    options.setInterOpNumThreads(1);
    options.setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);
    _encSession = OrtSession.fromBuffer(enc, options);
    _jointSession = OrtSession.fromBuffer(joint, options);
    options.release();
  }

  void dispose() {
    _encSession?.release();
    _jointSession?.release();
    _encSession = null;
    _jointSession = null;
  }

  // ---- low-level inference ----

  /// Encode mel frames (flat, row-major (T, 80)) -> encoder frames.
  /// Returns (flat enc frames (T', 320), T').
  (Float32List, int) _encode(
      Float32List mel, int nFrames, OrtRunOptions runOptions) {
    final melT =
        OrtValueTensor.createTensorWithDataList(mel, [1, nFrames, 80]);
    final lenT = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList([nFrames]), [1]);
    final langT = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList([0]), [1]);
    try {
      final outputs = _encSession!.run(runOptions, {
        'mel': melT,
        'mel_lengths': lenT,
        'lang_ids': langT,
      }, [
        'enc',
        'out_lengths'
      ]);
      try {
        final encVal = (outputs[0] as OrtValueTensor).value;
        final lenVal = (outputs[1] as OrtValueTensor).value as List;
        final nEnc = (lenVal[0] as num).toInt();
        final enc = _flattenToFloat32(encVal, nEnc * dModel);
        return (enc, nEnc);
      } finally {
        for (final o in outputs) {
          o?.release();
        }
      }
    } finally {
      melT.release();
      lenT.release();
      langT.release();
    }
  }

  /// Greedy loop over encoder frames (flat, (T', 320)). Returns token ids.
  List<int> _greedyLoop(
      Float32List enc, int nEnc, OrtRunOptions runOptions) {
    final emissions = <int>[];
    var hPre = Float32List(predictorLayers * predictorHidden);
    var cPre = Float32List(predictorLayers * predictorHidden);
    var token = blankId;
    final encFrame = Float32List(dModel);
    for (int f = 0; f < nEnc; f++) {
      encFrame.setRange(0, dModel, enc, f * dModel);
      for (int s = 0; s < maxSymbolsPerFrame; s++) {
        final frameT = OrtValueTensor.createTensorWithDataList(
            encFrame, [1, dModel]);
        final tokenT = OrtValueTensor.createTensorWithDataList(
            Int64List.fromList([token]), [1]);
        final hT = OrtValueTensor.createTensorWithDataList(
            hPre, [predictorLayers, 1, predictorHidden]);
        final cT = OrtValueTensor.createTensorWithDataList(
            cPre, [predictorLayers, 1, predictorHidden]);
        try {
          final outputs = _jointSession!.run(runOptions, {
            'enc_frame': frameT,
            'token': tokenT,
            'h': hT,
            'c': cT,
          }, [
            'logits',
            'h_out',
            'c_out'
          ]);
          try {
            final logitsVal = (outputs[0] as OrtValueTensor).value as List;
            final logits = logitsVal[0] as List;
            var best = 0;
            var bestVal = (logits[0] as num).toDouble();
            for (int i = 1; i < logits.length; i++) {
              final v = (logits[i] as num).toDouble();
              if (v > bestVal) {
                bestVal = v;
                best = i;
              }
            }
            if (best == blankId) break;
            emissions.add(best);
            token = best;
            hPre = _flattenToFloat32((outputs[1] as OrtValueTensor).value,
                predictorLayers * predictorHidden);
            cPre = _flattenToFloat32((outputs[2] as OrtValueTensor).value,
                predictorLayers * predictorHidden);
          } finally {
            for (final o in outputs) {
              o?.release();
            }
          }
        } finally {
          frameT.release();
          tokenT.release();
          hT.release();
          cT.release();
        }
      }
    }
    return emissions;
  }

  /// Flatten a nested `List<num>` (any depth) into a Float32List.
  static Float32List _flattenToFloat32(dynamic nested, int expectedLength) {
    final out = Float32List(expectedLength);
    var idx = 0;
    void walk(dynamic node) {
      if (node is List) {
        for (final e in node) {
          walk(e);
        }
      } else if (node is num) {
        if (idx < expectedLength) out[idx++] = node.toDouble();
      }
    }

    walk(nested);
    return out;
  }

  // ---- full transcription + timing ----

  /// Transcribe 16 kHz mono PCM. Runs two passes:
  ///  1. full offline decode -> transcript + RTF.
  ///  2. final-chunk latency pass (mirrors bench_onnx.py: feed all but the
  ///     last 80 ms chunk untimed, then time last chunk + finalize).
  /// This is synchronous (blocks the calling thread) so the measured
  /// latencies are pure inference time, with no isolate overhead.
  SttResult transcribe(Float32List pcm, String inputLabel) {
    if (!isLoaded) throw StateError('Model not loaded');
    final durationSec = pcm.length / MelFrontend.sampleRate;
    final frontend = MelFrontend();
    final runOptions = OrtRunOptions();
    try {
      // --- pass 1: full offline decode (transcript + RTF) ---
      final t0 = Stopwatch()..start();
      final mel = frontend.computeMel(pcm);
      final nFrames = mel.length ~/ MelFrontend.nMels;
      String transcript = '';
      if (nFrames > 0) {
        final (enc, nEnc) = _encode(mel, nFrames, runOptions);
        final ids = _greedyLoop(enc, nEnc, runOptions);
        transcript = _tokenizer!.decode(ids);
      }
      final fullMs = t0.elapsedMicroseconds / 1000.0;
      final rtf = durationSec > 0 ? (fullMs / 1000.0) / durationSec : 0.0;

      // --- pass 2: final-chunk latency (Moonshine definition) ---
      final streaming = StreamingMelFrontend();
      final chunks = <Float32List>[];
      for (int i = 0; i < pcm.length; i += chunkSamples) {
        chunks.add(pcm.sublist(i, math.min(i + chunkSamples, pcm.length)));
      }
      for (int i = 0; i < chunks.length - 1; i++) {
        streaming.feed(chunks[i]); // untimed
      }
      double feMs = 0, encMs = 0, jointMs = 0;
      final sw = Stopwatch()..start();
      if (chunks.isNotEmpty) {
        var t1 = Stopwatch()..start();
        final lastMel = streaming.feed(chunks.last);
        feMs += t1.elapsedMicroseconds / 1000.0;
        if (lastMel.isNotEmpty) {
          t1 = Stopwatch()..start();
          final (encLast, nEncLast) =
              _encode(lastMel, lastMel.length ~/ MelFrontend.nMels, runOptions);
          encMs += t1.elapsedMicroseconds / 1000.0;
          t1 = Stopwatch()..start();
          _greedyLoop(encLast, nEncLast, runOptions);
          jointMs += t1.elapsedMicroseconds / 1000.0;
        }
      }
      var t1 = Stopwatch()..start();
      final tail = streaming.flush();
      feMs += t1.elapsedMicroseconds / 1000.0;
      if (tail.isNotEmpty) {
        t1 = Stopwatch()..start();
        final (encTail, nEncTail) =
            _encode(tail, tail.length ~/ MelFrontend.nMels, runOptions);
        encMs += t1.elapsedMicroseconds / 1000.0;
        t1 = Stopwatch()..start();
        _greedyLoop(encTail, nEncTail, runOptions);
        jointMs += t1.elapsedMicroseconds / 1000.0;
      }
      final latencyMs = sw.elapsedMicroseconds / 1000.0;

      return SttResult(
        transcript: transcript,
        inputLabel: inputLabel,
        durationSec: durationSec,
        latencyMs: latencyMs,
        frontendMs: feMs,
        encoderMs: encMs,
        jointMs: jointMs,
        rtf: rtf,
        modelLoadMs: modelLoadMs,
      );
    } finally {
      runOptions.release();
    }
  }
}
