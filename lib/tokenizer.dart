// Character-level tokenizer. Mirrors src/tokenizer.py.
// Vocab JSON: {"blank": 0, "unk": 1, "pad": 2, "tokens": [...]}.
// decode() joins token ids, dropping blank/pad/unk (ids 0, 1, 2).

import 'dart:convert';

class CharTokenizer {
  static const int blankId = 0;
  static const int unkId = 1;
  static const int padId = 2;

  final List<String> tokens;

  CharTokenizer(this.tokens) {
    if (tokens.length < 3 ||
        tokens[blankId] != '<blank>' ||
        tokens[unkId] != '<unk>' ||
        tokens[padId] != '<pad>') {
      throw const FormatException('tokens[0:3] must be [<blank>, <unk>, <pad>]');
    }
  }

  factory CharTokenizer.fromJson(String jsonStr) {
    final data = jsonDecode(jsonStr) as Map<String, dynamic>;
    final tokens = (data['tokens'] as List).cast<String>();
    return CharTokenizer(tokens);
  }

  int get vocabSize => tokens.length;

  String decode(List<int> ids) {
    final sb = StringBuffer();
    for (final id in ids) {
      if (id == blankId || id == padId || id == unkId) continue;
      if (id < 0 || id >= tokens.length) continue;
      sb.write(tokens[id]);
    }
    return sb.toString();
  }
}
