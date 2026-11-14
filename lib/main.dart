// STT Test — minimal on-device test app for the s1 STT ONNX model.
// Measures REAL latency / RTF on the phone. Test tool, not a product.
// Text buttons only (no icons). English-only transcription in v1:
// the exported encoder baked in the English adapter (see README.md).

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import 'stt_engine.dart';
import 'wav_io.dart';

const String appVersion = '1.0.0';
const String modelVersion = 's1 (11.5 epochs, int8 ONNX)';

class SampleClip {
  final String asset;
  final String label;
  final String expected;

  const SampleClip(this.asset, this.label, this.expected);
}

const List<SampleClip> samples = [
  SampleClip('assets/samples/sample1.wav', 'Sample 1 (2.3s)',
      'STEPPING OUT OF HIS HIDING PLACE'),
  SampleClip('assets/samples/sample2.wav', 'Sample 2 (2.8s)',
      'LEFT TO GOD AND NATURE AND THEMSELVES'),
  SampleClip('assets/samples/sample3.wav', 'Sample 3 (3.0s)',
      'AN INDIAN TRAIL LED UP THE RIVER FROM'),
];

void main() {
  runApp(const SttTestApp());
}

class SttTestApp extends StatelessWidget {
  const SttTestApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'STT Test',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const TestPage(),
    );
  }
}

class TestPage extends StatefulWidget {
  const TestPage({super.key});

  @override
  State<TestPage> createState() => _TestPageState();
}

class _TestPageState extends State<TestPage> {
  final SttEngine _engine = SttEngine();
  final AudioRecorder _recorder = AudioRecorder();

  bool _loading = true;
  String? _loadError;
  bool _busy = false;
  bool _recording = false;
  int _recordSeconds = 0;
  Timer? _recordTimer;
  SttResult? _result;
  String? _expected;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _recordTimer?.cancel();
    _recorder.dispose();
    _engine.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    try {
      await _engine.load();
    } catch (e) {
      _loadError = e.toString();
    }
    if (mounted) setState(() => _loading = false);
  }

  Future<void> _runSample(SampleClip sample) async {
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
      _expected = sample.expected;
    });
    // Let the spinner paint before the (synchronous) inference blocks.
    await Future<void>.delayed(const Duration(milliseconds: 100));
    try {
      final bytes = await rootBundle.load(sample.asset);
      final wav = parseWav(bytes.buffer.asUint8List());
      final pcm = resampleTo16k(wav.samples, wav.sampleRate);
      final result = _engine.transcribe(pcm, sample.label);
      if (mounted) setState(() => _result = result);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _toggleRecord() async {
    if (_recording) {
      await _stopRecording();
      return;
    }
    try {
      if (!await _recorder.hasPermission()) {
        setState(() => _error = 'Microphone permission denied.');
        return;
      }
      final dir = await getTemporaryDirectory();
      final path = '${dir.path}/mic_recording.wav';
      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          numChannels: 1,
        ),
        path: path,
      );
      setState(() {
        _recording = true;
        _recordSeconds = 0;
        _error = null;
        _result = null;
        _expected = null;
      });
      _recordTimer = Timer.periodic(const Duration(seconds: 1), (t) {
        if (!mounted) return;
        setState(() => _recordSeconds = t.tick);
        if (t.tick >= 10) _stopRecording(); // auto-stop at 10 s
      });
    } catch (e) {
      setState(() => _error = e.toString());
    }
  }

  Future<void> _stopRecording() async {
    _recordTimer?.cancel();
    _recordTimer = null;
    String? path;
    try {
      path = await _recorder.stop();
    } catch (e) {
      setState(() {
        _recording = false;
        _error = e.toString();
      });
      return;
    }
    setState(() => _recording = false);
    if (path == null) {
      setState(() => _error = 'Recording produced no file.');
      return;
    }
    setState(() => _busy = true);
    await Future<void>.delayed(const Duration(milliseconds: 100));
    try {
      final bytes = await File(path).readAsBytes();
      final wav = parseWav(bytes);
      final pcm = resampleTo16k(wav.samples, wav.sampleRate);
      final result = _engine.transcribe(pcm, 'Mic recording');
      if (mounted) setState(() => _result = result);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _copyResult() async {
    final result = _result;
    if (result == null) return;
    await Clipboard.setData(
      ClipboardData(text: result.toCopyText(appVersion, modelVersion)),
    );
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Result copied to clipboard')),
    );
  }

  Widget _metric(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 140,
            child: Text(label,
                style: const TextStyle(fontWeight: FontWeight.w600)),
          ),
          Expanded(
            child: SelectableText(value),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final result = _result;
    return Scaffold(
      appBar: AppBar(title: const Text('STT Test')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Model: $modelVersion',
                        style: const TextStyle(fontWeight: FontWeight.w600)),
                    const SizedBox(height: 4),
                    if (_loading)
                      const Row(children: [
                        SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                        SizedBox(width: 8),
                        Text('Loading model...'),
                      ])
                    else if (_loadError != null)
                      Text('Load failed: $_loadError',
                          style: const TextStyle(color: Colors.red))
                    else
                      Text(
                          'Loaded in ${_engine.modelLoadMs.toStringAsFixed(0)} ms'
                          ' (vocab ${_engine.vocabSize} tokens, 1 thread)'),
                    const SizedBox(height: 4),
                    const Text(
                      'Test build — English only for now',
                      style: TextStyle(fontStyle: FontStyle.italic),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            const Text('Samples (repeatable tests)',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            for (final sample in samples)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: OutlinedButton(
                  onPressed: (_loading || _busy || _recording)
                      ? null
                      : () => _runSample(sample),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    child: Column(
                      children: [
                        Text(sample.label),
                        const SizedBox(height: 2),
                        Text(
                          'Expected: ${sample.expected}',
                          style: const TextStyle(
                              fontSize: 11, fontWeight: FontWeight.w400),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            const SizedBox(height: 8),
            FilledButton(
              onPressed: (_loading || _busy) ? null : _toggleRecord,
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: Text(_recording
                    ? 'STOP RECORDING ($_recordSeconds s / 10 s max)'
                    : 'RECORD FROM MIC'),
              ),
            ),
            const SizedBox(height: 16),
            if (_busy)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 12),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(width: 12),
                    Text('Transcribing...'),
                  ],
                ),
              ),
            if (_error != null)
              Card(
                color: Colors.red.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Text('Error: $_error',
                      style: const TextStyle(color: Colors.red)),
                ),
              ),
            if (result != null)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Result',
                          style: TextStyle(
                              fontSize: 16, fontWeight: FontWeight.w600)),
                      const SizedBox(height: 8),
                      _metric('Input', result.inputLabel),
                      _metric('Duration',
                          '${result.durationSec.toStringAsFixed(2)} s'),
                      _metric(
                          'Transcript',
                          result.transcript.isEmpty
                              ? '(empty)'
                              : result.transcript),
                      if (_expected != null)
                        _metric('Expected', _expected!),
                      _metric('Latency (final chunk)',
                          '${result.latencyMs.toStringAsFixed(1)} ms'),
                      _metric('  frontend',
                          '${result.frontendMs.toStringAsFixed(1)} ms'),
                      _metric('  encoder',
                          '${result.encoderMs.toStringAsFixed(1)} ms'),
                      _metric('  joint',
                          '${result.jointMs.toStringAsFixed(1)} ms'),
                      _metric('RTF', result.rtf.toStringAsFixed(2)),
                      _metric('Model load',
                          '${result.modelLoadMs.toStringAsFixed(0)} ms'),
                      const SizedBox(height: 12),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton(
                          onPressed: _copyResult,
                          child: const Text('COPY RESULT'),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
