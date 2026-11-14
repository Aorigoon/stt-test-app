// Log-Mel frontend, pure Dart — must match src/frontend.py EXACTLY.
//   16 kHz, 25 ms window (400 samples), 10 ms hop (160 samples),
//   n_fft 400, 80 HTK mel bins, f_min 0, f_max 8000, center=False,
//   power 2.0, log(mel + 1e-6), no CMVN.
// Verified against the Python pipeline by test/mel_frontend_test.dart
// (max abs diff vs torchaudio reference on a val clip).
//
// The DFT is computed directly (201 bins x 400 samples) with precomputed
// cos/sin tables. It is slower than an FFT but obviously correct; the
// frontend time is reported separately in the app's timing breakdown.

import 'dart:math' as math;
import 'dart:typed_data';

class MelFrontend {
  static const int sampleRate = 16000;
  static const int nFft = 400;
  static const int winLength = 400;
  static const int hopLength = 160;
  static const int nMels = 80;
  static const double fMin = 0.0;
  static const double fMax = 8000.0;
  static const int nFreqBins = nFft ~/ 2 + 1; // 201

  /// Number of complete frames for [nSamples] (center=False).
  static int numFrames(int nSamples) {
    final n = (nSamples - winLength) ~/ hopLength + 1;
    return n < 0 ? 0 : n;
  }

  // Hann window (periodic), length 400: w[n] = 0.5 - 0.5*cos(2*pi*n/400).
  final Float64List _window = Float64List(winLength);
  // DFT tables: _cos[k * 400 + n] = cos(2*pi*k*n/400), k = 0..200.
  final Float64List _cosTable = Float64List(nFreqBins * nFft);
  final Float64List _sinTable = Float64List(nFreqBins * nFft);
  // Mel filterbank, row-major [freq bin][mel]: _fb[k * 80 + j].
  final Float64List _fb = Float64List(nFreqBins * nMels);

  MelFrontend() {
    for (int n = 0; n < winLength; n++) {
      _window[n] = 0.5 - 0.5 * math.cos(2 * math.pi * n / winLength);
    }
    for (int k = 0; k < nFreqBins; k++) {
      for (int n = 0; n < nFft; n++) {
        final angle = 2 * math.pi * k * n / nFft;
        _cosTable[k * nFft + n] = math.cos(angle);
        _sinTable[k * nFft + n] = math.sin(angle);
      }
    }
    _buildMelFilterbank();
  }

  double _hzToMel(double f) => 2595.0 * (math.log(1.0 + f / 700.0) / math.ln10);
  double _melToHz(double m) => 700.0 * (math.pow(10.0, m / 2595.0) - 1.0);

  // Triangular HTK mel filterbank, librosa/torchaudio convention:
  // 82 mel points linspace in mel space -> Hz; each band j is the triangle
  // (hz[j], hz[j+1], hz[j+2]); weights = max(0, min(rising, falling)).
  // No area normalization (torchaudio norm=None).
  void _buildMelFilterbank() {
    final melMin = _hzToMel(fMin);
    final melMax = _hzToMel(fMax);
    final hzPoints = Float64List(nMels + 2);
    for (int i = 0; i < nMels + 2; i++) {
      final m = melMin + (melMax - melMin) * i / (nMels + 1);
      hzPoints[i] = _melToHz(m);
    }
    for (int k = 0; k < nFreqBins; k++) {
      final freq = k * sampleRate / nFft;
      for (int j = 0; j < nMels; j++) {
        final lower = (freq - hzPoints[j]) / (hzPoints[j + 1] - hzPoints[j]);
        final upper =
            (hzPoints[j + 2] - freq) / (hzPoints[j + 2] - hzPoints[j + 1]);
        var w = lower < upper ? lower : upper;
        if (w < 0) w = 0;
        _fb[k * nMels + j] = w;
      }
    }
  }

  /// Offline log-mel: [pcm] (16 kHz mono float32) -> (T, 80) row-major.
  /// T = numFrames(pcm.length). Returns empty list if pcm is too short.
  Float32List computeMel(Float32List pcm) {
    final nFrames = numFrames(pcm.length);
    if (nFrames <= 0) return Float32List(0);
    final out = Float32List(nFrames * nMels);
    final windowed = Float64List(winLength);
    final power = Float64List(nFreqBins);
    for (int f = 0; f < nFrames; f++) {
      final start = f * hopLength;
      for (int n = 0; n < winLength; n++) {
        windowed[n] = pcm[start + n] * _window[n];
      }
      for (int k = 0; k < nFreqBins; k++) {
        double re = 0.0, im = 0.0;
        final base = k * nFft;
        for (int n = 0; n < nFft; n++) {
          final x = windowed[n];
          re += x * _cosTable[base + n];
          im -= x * _sinTable[base + n];
        }
        power[k] = re * re + im * im;
      }
      for (int j = 0; j < nMels; j++) {
        double s = 0.0;
        for (int k = 0; k < nFreqBins; k++) {
          s += power[k] * _fb[k * nMels + j];
        }
        out[f * nMels + j] = math.log(s + 1e-6);
      }
    }
    return out;
  }
}

/// Chunk-wise log-mel with identical framing to [MelFrontend].
/// Mirrors Python StreamingMelFrontend: feed PCM chunks, frames are emitted
/// as soon as their 25 ms window is fully buffered; flush() emits the final
/// zero-padded frame(s).
class StreamingMelFrontend {
  final MelFrontend _fe = MelFrontend();
  final List<double> _buf = <double>[];

  void reset() => _buf.clear();

  /// Feed a PCM chunk. Returns new log-mel frames, (T, 80) row-major.
  Float32List feed(Float32List pcm) {
    if (pcm.isEmpty) return Float32List(0);
    _buf.addAll(pcm);
    final nFrames = MelFrontend.numFrames(_buf.length);
    if (nFrames <= 0) return Float32List(0);
    final segLen = (nFrames - 1) * MelFrontend.hopLength + MelFrontend.winLength;
    final seg = Float32List(segLen);
    for (int i = 0; i < segLen; i++) {
      seg[i] = _buf[i];
    }
    final mel = _fe.computeMel(seg);
    // Samples from nFrames*hop onward belong to future frames.
    _buf.removeRange(0, nFrames * MelFrontend.hopLength);
    return mel;
  }

  /// Emit final frame(s) for leftover samples (zero-padded).
  Float32List flush() {
    if (_buf.isEmpty) return Float32List(0);
    final n = _buf.length;
    final int total;
    if (n < MelFrontend.winLength) {
      total = MelFrontend.winLength;
    } else {
      total = MelFrontend.winLength +
          ((n - MelFrontend.winLength) ~/ MelFrontend.hopLength + 1) *
              MelFrontend.hopLength;
    }
    final padded = Float32List(total);
    for (int i = 0; i < n; i++) {
      padded[i] = _buf[i];
    }
    _buf.clear();
    return _fe.computeMel(padded);
  }
}
