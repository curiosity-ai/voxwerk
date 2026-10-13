using System.Collections.Concurrent;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Voxwerk.Text;

/// <summary>
/// Pure managed port of the Qwen2 byte-level BPE tokenizer (<c>vocab.json</c> + <c>merges.txt</c>),
/// equivalent to Hugging Face's <c>Qwen2TokenizerFast</c>: NFC normalization, added/special tokens
/// matched verbatim, the Qwen2 split regex, GPT-2 byte-to-unicode mapping, then ranked BPE merges.
/// </summary>
public sealed class Qwen2Tokenizer
{
    private const string SplitPattern =
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    private static readonly Regex Splitter = new(SplitPattern, RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private readonly Dictionary<string, int> _vocab;
    private readonly string[] _idToToken;
    private readonly Dictionary<(string, string), int> _ranks;
    private readonly (string Text, int Id)[] _addedTokens; // longest first
    private readonly char[] _byteToChar = new char[256];
    private readonly Dictionary<char, byte> _charToByte = new();
    private readonly ConcurrentDictionary<string, int[]> _cache = new(StringComparer.Ordinal);

    public Qwen2Tokenizer(IReadOnlyDictionary<string, int> vocab, IEnumerable<string> merges, IEnumerable<(string Text, int Id)> addedTokens)
    {
        _vocab = new Dictionary<string, int>(vocab, StringComparer.Ordinal);
        var added = addedTokens.ToList();
        int maxId = Math.Max(_vocab.Values.DefaultIfEmpty(0).Max(), added.Select(a => a.Id).DefaultIfEmpty(0).Max());
        _idToToken = new string[maxId + 1];
        foreach (var (t, id) in _vocab) _idToToken[id] = t;
        foreach (var (t, id) in added)
        {
            _idToToken[id] = t;
            _vocab[t] = id;
        }
        _addedTokens = added.OrderByDescending(a => a.Text.Length).ToArray();
        _ranks = new Dictionary<(string, string), int>();
        int rank = 0;
        foreach (var line in merges)
        {
            if (line.Length == 0 || line.StartsWith("#version", StringComparison.Ordinal)) continue;
            int sp = line.IndexOf(' ');
            if (sp <= 0) continue;
            _ranks.TryAdd((line[..sp], line[(sp + 1)..]), rank++);
        }
        BuildByteMap();
    }

    /// <summary>Loads <c>vocab.json</c>, <c>merges.txt</c> and the added tokens of <c>tokenizer_config.json</c> from a model directory.</summary>
    public static Qwen2Tokenizer FromDirectory(string dir)
    {
        var vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(Path.Combine(dir, "vocab.json")))
                    ?? throw new InvalidDataException("vocab.json is empty");
        var merges = File.ReadAllLines(Path.Combine(dir, "merges.txt"), Encoding.UTF8);
        var added = new List<(string, int)>();
        string cfgPath = Path.Combine(dir, "tokenizer_config.json");
        if (File.Exists(cfgPath))
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(cfgPath));
            if (doc.RootElement.TryGetProperty("added_tokens_decoder", out var dec))
                foreach (var p in dec.EnumerateObject())
                    added.Add((p.Value.GetProperty("content").GetString()!, int.Parse(p.Name, CultureInfo.InvariantCulture)));
        }
        return new Qwen2Tokenizer(vocab, merges, added);
    }

    public int VocabSize => _idToToken.Length;

    public int TokenToId(string token) => _vocab.TryGetValue(token, out int id) ? id : throw new KeyNotFoundException(token);

    /// <summary>Encodes text to token ids (no BOS/EOS are added - Qwen2 has none by default).</summary>
    public int[] Encode(string text)
    {
        var ids = new List<int>();
        int pos = 0;
        while (pos < text.Length)
        {
            // Find the earliest added token occurrence (longest wins at equal position).
            int bestAt = -1, bestId = -1, bestLen = 0;
            foreach (var (t, id) in _addedTokens)
            {
                int at = text.IndexOf(t, pos, StringComparison.Ordinal);
                if (at >= 0 && (bestAt < 0 || at < bestAt || (at == bestAt && t.Length > bestLen)))
                {
                    bestAt = at;
                    bestId = id;
                    bestLen = t.Length;
                }
            }
            int end = bestAt < 0 ? text.Length : bestAt;
            if (end > pos) EncodeOrdinary(text.Substring(pos, end - pos), ids);
            if (bestAt < 0) break;
            ids.Add(bestId);
            pos = bestAt + bestLen;
        }
        return ids.ToArray();
    }

    /// <summary>Decodes ids back to text (added tokens are emitted verbatim).</summary>
    public string Decode(IEnumerable<int> ids)
    {
        var bytes = new List<byte>();
        var sb = new StringBuilder();
        foreach (int id in ids)
        {
            string tok = _idToToken[id];
            if (_addedTokens.Any(a => a.Id == id))
            {
                sb.Append(Encoding.UTF8.GetString(bytes.ToArray()));
                bytes.Clear();
                sb.Append(tok);
                continue;
            }
            foreach (char ch in tok) bytes.Add(_charToByte[ch]);
        }
        sb.Append(Encoding.UTF8.GetString(bytes.ToArray()));
        return sb.ToString();
    }

    private void EncodeOrdinary(string text, List<int> ids)
    {
        text = text.Normalize(NormalizationForm.FormC);
        foreach (string piece in Split(text))
        {
            if (!_cache.TryGetValue(piece, out var pieceIds))
            {
                pieceIds = Bpe(ByteEncode(piece));
                if (_cache.Count < 100_000) _cache[piece] = pieceIds;
            }
            ids.AddRange(pieceIds);
        }
    }

    /// <summary>Applies the split regex. .NET classifies UTF-16 code units rather than code points, so
    /// supplementary characters are matched through a same-category BMP stand-in and mapped back.</summary>
    internal static List<string> Split(string text)
    {
        string matchText = text;
        List<int>? offsets = null;
        if (text.AsSpan().IndexOfAnyInRange('\uD800', '\uDFFF') >= 0)
        {
            var projected = new StringBuilder(text.Length);
            offsets = new List<int>(text.Length + 1);
            int src = 0;
            foreach (Rune r in text.EnumerateRunes())
            {
                offsets.Add(src);
                projected.Append(r.IsBmp ? (char)r.Value : CategoryRepresentative(Rune.GetUnicodeCategory(r)));
                src += r.Utf16SequenceLength;
            }
            offsets.Add(text.Length);
            matchText = projected.ToString();
        }
        var pieces = new List<string>();
        int position = 0;
        void Add(int s, int len)
        {
            int a = offsets == null ? s : offsets[s];
            int b = offsets == null ? s + len : offsets[s + len];
            pieces.Add(text.Substring(a, b - a));
        }
        foreach (var m in Splitter.EnumerateMatches(matchText))
        {
            if (m.Index > position) Add(position, m.Index - position);
            if (m.Length > 0) Add(m.Index, m.Length);
            position = m.Index + m.Length;
        }
        if (position < matchText.Length) Add(position, matchText.Length - position);
        return pieces;
    }

    private static char CategoryRepresentative(UnicodeCategory category) => category switch
    {
        UnicodeCategory.UppercaseLetter => 'Ω',
        UnicodeCategory.LowercaseLetter => 'α',
        UnicodeCategory.TitlecaseLetter => 'ǅ',
        UnicodeCategory.ModifierLetter => 'ʰ',
        UnicodeCategory.OtherLetter => 'ꀀ',
        UnicodeCategory.NonSpacingMark => '́',
        UnicodeCategory.SpacingCombiningMark => 'ः',
        UnicodeCategory.EnclosingMark => '҈',
        UnicodeCategory.DecimalDigitNumber => '٠',
        UnicodeCategory.LetterNumber => 'Ⅻ',
        UnicodeCategory.OtherNumber => '²',
        UnicodeCategory.ConnectorPunctuation => '‿',
        UnicodeCategory.DashPunctuation => '‐',
        UnicodeCategory.OpenPunctuation => '❨',
        UnicodeCategory.ClosePunctuation => '❩',
        UnicodeCategory.InitialQuotePunctuation => '‘',
        UnicodeCategory.FinalQuotePunctuation => '’',
        UnicodeCategory.OtherPunctuation => '※',
        UnicodeCategory.MathSymbol => '∑',
        UnicodeCategory.CurrencySymbol => '¢',
        UnicodeCategory.ModifierSymbol => '˄',
        UnicodeCategory.OtherSymbol => '©',
        UnicodeCategory.SpaceSeparator => ' ',
        _ => '',
    };

    private string ByteEncode(string piece)
    {
        var bytes = Encoding.UTF8.GetBytes(piece);
        var chars = new char[bytes.Length];
        for (int i = 0; i < bytes.Length; i++) chars[i] = _byteToChar[bytes[i]];
        return new string(chars);
    }

    private int[] Bpe(string word)
    {
        if (_vocab.TryGetValue(word, out int direct)) return [direct];
        var parts = new List<string>(word.Length);
        foreach (char c in word) parts.Add(c.ToString());
        while (parts.Count > 1)
        {
            int best = int.MaxValue, at = -1;
            for (int i = 0; i < parts.Count - 1; i++)
                if (_ranks.TryGetValue((parts[i], parts[i + 1]), out int r) && r < best)
                {
                    best = r;
                    at = i;
                }
            if (at < 0) break;
            // Merge every occurrence of the best pair, left to right (the reference BPE semantics).
            string a = parts[at], b = parts[at + 1];
            var merged = new List<string>(parts.Count);
            for (int i = 0; i < parts.Count;)
            {
                if (i < parts.Count - 1 && parts[i] == a && parts[i + 1] == b)
                {
                    merged.Add(a + b);
                    i += 2;
                }
                else merged.Add(parts[i++]);
            }
            parts = merged;
        }
        var ids = new int[parts.Count];
        for (int i = 0; i < parts.Count; i++)
            ids[i] = _vocab.TryGetValue(parts[i], out int id) ? id : throw new InvalidDataException($"BPE produced unknown token '{parts[i]}'");
        return ids;
    }

    private void BuildByteMap()
    {
        // GPT-2 bytes_to_unicode: printable latin-1 bytes map to themselves, the rest to 256+.
        var bs = new List<int>();
        for (int b = '!'; b <= '~'; b++) bs.Add(b);
        for (int b = '¡'; b <= '¬'; b++) bs.Add(b);
        for (int b = '®'; b <= 'ÿ'; b++) bs.Add(b);
        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n++);
            }
        }
        for (int i = 0; i < bs.Count; i++)
        {
            _byteToChar[bs[i]] = (char)cs[i];
            _charToByte[(char)cs[i]] = (byte)bs[i];
        }
    }
}
