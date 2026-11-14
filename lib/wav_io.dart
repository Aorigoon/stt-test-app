// Minimal WAV parsing + resampling. Pure Dart (no flutter imports).

import 'dart:typed_data';

class WavData {
  /// Mono samples in [-1, 1].
  final Float32List samples;

  /// Sample rate of [samples] (after any resampling: always 16000 here).
  final int sampleRate;

  /// Original file sample rate (before resampling).
  final int fileSampleRate;

  WavData(this.samples, this.sampleRate, this.fileSampleRate);
}

/// Parse a WAV file (RIFF, PCM 16-bit / 8-bit / float32, mono or stereo).
/// Stereo is downmixed by averaging channels. Throws [FormatException] on
/// unsupported or malformed input.
WavData parseWav(Uint8List bytes) {
  final data = ByteData.sublistView(bytes);
  String tag(int off) => String.fromCharCodes(bytes.sublist(off, off + 4));
  if (bytes.length < 44 || tag(0) != 'RIFF' || tag(8) != 'WAVE') {
    throw const FormatException('Not a RIFF/WAVE file');
  }
  int fmtAudio = -1, fmtChannels = 0, fmtRate = 0, fmtBits = 0;
  int dataOff = -1, dataLen = 0;
  int off = 12;
  while (off + 8 <= bytes.length) {
    final id = tag(off);
    final size = data.getUint32(off + 4, Endian.little);
    final body = off + 8;
    if (id == 'fmt ') {
      fmtAudio = data.getUint16(body, Endian.little);
      fmtChannels = data.getUint16(body + 2, Endian.little);
      fmtRate = data.getUint32(body + 4, Endian.little);
      fmtBits = data.getUint16(body + 14, Endian.little);
    } else if (id == 'data') {
      dataOff = body;
      dataLen = size;
      break;
    }
    off = body + size + (size.isOdd ? 1 : 0); // chunks are word-aligned
  }
  if (fmtAudio < 0 || dataOff < 0) {
    throw const FormatException('WAV missing fmt or data chunk');
  }
  if (fmtChannels < 1) {
    throw const FormatException('WAV has no channels');
  }
  final bytesPerSample = fmtBits ~/ 8;
  if (bytesPerSample <= 0) {
    throw FormatException('Unsupported WAV bit depth: $fmtBits');
  }
  final frameCount = dataLen ~/ (bytesPerSample * fmtChannels);
  final out = Float32List(frameCount);
  for (int i = 0; i < frameCount; i++) {
    double sum = 0.0;
    for (int ch = 0; ch < fmtChannels; ch++) {
      final p = dataOff + (i * fmtChannels + ch) * bytesPerSample;
      double v;
      if (fmtAudio == 3 && fmtBits == 32) {
        v = data.getFloat32(p, Endian.little);
      } else if (fmtAudio == 1 && fmtBits == 16) {
        v = data.getInt16(p, Endian.little) / 32768.0;
      } else if (fmtAudio == 1 && fmtBits == 8) {
        v = (data.getUint8(p) - 128) / 128.0;
      } else if (fmtAudio == 1 && fmtBits == 32) {
        v = data.getInt32(p, Endian.little) / 2147483648.0;
      } else {
        throw FormatException(
            'Unsupported WAV format: audioFormat=$fmtAudio bits=$fmtBits');
      }
      sum += v;
    }
    out[i] = sum / fmtChannels;
  }
  return WavData(out, fmtRate, fmtRate);
}

/// Linear-interpolation resample to 16 kHz (no-op when already 16 kHz).
Float32List resampleTo16k(Float32List samples, int sampleRate) {
  if (sampleRate == 16000) return samples;
  if (samples.isEmpty) return samples;
  final ratio = sampleRate / 16000.0;
  final outLen = (samples.length / ratio).floor();
  final out = Float32List(outLen);
  for (int i = 0; i < outLen; i++) {
    final pos = i * ratio;
    final idx = pos.floor();
    final frac = pos - idx;
    final a = samples[idx];
    final b = idx + 1 < samples.length ? samples[idx + 1] : a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}
