// Verifies the Dart log-Mel frontend against the Python (torchaudio)
// reference computed by /tmp/stt_ref_dump.py and stored in
// test/fixtures/mel_ref.json (231 frames for sample1.wav).
// If this test fails, transcripts on the phone will be garbage.

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:stt_test_app/mel_frontend.dart';
import 'package:stt_test_app/tokenizer.dart';
import 'package:stt_test_app/wav_io.dart';

void main() {
  test('wav parsing: sample1 is 16 kHz mono', () {
    final bytes = File('test/fixtures/sample1.wav').readAsBytesSync();
    final wav = parseWav(bytes);
    expect(wav.sampleRate, 16000);
    // 2.33 s * 16000 = 37280 samples
    expect(wav.samples.length, 37280);
    // Samples are in [-1, 1] and non-silent.
    var maxAbs = 0.0;
    for (final s in wav.samples) {
      if (s.abs() > maxAbs) maxAbs = s.abs();
    }
    expect(maxAbs, greaterThan(0.1));
    expect(maxAbs, lessThanOrEqualTo(1.0));
  });

  test('mel matches Python reference', () {
    final bytes = File('test/fixtures/sample1.wav').readAsBytesSync();
    final wav = parseWav(bytes);
    final frontend = MelFrontend();
    final sw = Stopwatch()..start();
    final mel = frontend.computeMel(wav.samples);
    final elapsedMs = sw.elapsedMicroseconds / 1000.0;
    // ignore: avoid_print
    print('Dart mel for 2.33 s clip: ${elapsedMs.toStringAsFixed(0)} ms');

    final refJson = jsonDecode(
      File('test/fixtures/mel_ref.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    final nFrames = refJson['n_frames'] as int;
    final refMel = (refJson['mel'] as List).cast<List>();

    expect(mel.length ~/ MelFrontend.nMels, nFrames);
    double maxDiff = 0.0;
    int worstT = 0, worstJ = 0;
    for (int t = 0; t < nFrames; t++) {
      for (int j = 0; j < MelFrontend.nMels; j++) {
        final d = (mel[t * MelFrontend.nMels + j] - (refMel[t][j] as num))
            .abs()
            .toDouble();
        if (d > maxDiff) {
          maxDiff = d;
          worstT = t;
          worstJ = j;
        }
      }
    }
    // ignore: avoid_print
    print('max mel abs diff: $maxDiff (frame $worstT, bin $worstJ)');
    expect(maxDiff, lessThan(1e-3));
  });

  test('streaming frontend matches offline framing', () {
    final bytes = File('test/fixtures/sample1.wav').readAsBytesSync();
    final wav = parseWav(bytes);
    final offline = MelFrontend().computeMel(wav.samples);

    // Feed in 80 ms chunks (1280 samples) like the app does.
    final streaming = StreamingMelFrontend();
    final fed = <double>[];
    for (int i = 0; i < wav.samples.length; i += 1280) {
      final end =
          (i + 1280 < wav.samples.length) ? i + 1280 : wav.samples.length;
      final chunk = wav.samples.sublist(i, end);
      fed.addAll(streaming.feed(chunk));
    }
    final tail = streaming.flush();
    // feed() frames must equal the offline frames exactly; flush adds the
    // final zero-padded frame(s) on top.
    expect(fed.length, offline.length);
    double maxDiff = 0.0;
    for (int i = 0; i < offline.length; i++) {
      final d = (fed[i] - offline[i]).abs();
      if (d > maxDiff) maxDiff = d;
    }
    expect(maxDiff, 0.0);
    expect(tail.length, MelFrontend.nMels); // 1 padded frame for this clip
  });

  test('tokenizer decode skips blank/unk/pad', () {
    final tok = CharTokenizer.fromJson(
      '{"blank": 0, "unk": 1, "pad": 2, "tokens": '
      '["<blank>", "<unk>", "<pad>", "h", "i", " "]}',
    );
    expect(tok.decode([3, 4, 0, 1, 2, 5, 4]), 'hi i');
    expect(tok.vocabSize, 6);
  });
}
