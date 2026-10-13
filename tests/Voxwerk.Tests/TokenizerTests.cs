using System.Text.Json;
using Voxwerk.Text;
using Xunit.Abstractions;

namespace Voxwerk.Tests;

public class TokenizerTests(ITestOutputHelper output)
{
    [Fact]
    public void MatchesReferenceTokenizer()
    {
        if (!Fixtures.Available(output)) return;
        var tok = Qwen2Tokenizer.FromDirectory(Fixtures.ModelDir!);
        var cases = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(Path.Combine(Fixtures.ReferenceDir!, "tok_cases.json")))!;
        foreach (var (name, text) in cases)
        {
            var expected = Fixtures.Ref("tok_" + name).Ints;
            var actual = tok.Encode(text);
            output.WriteLine($"{name}: {actual.Length} tokens");
            Assert.Equal(expected, actual);
            Assert.Equal(text.Normalize(System.Text.NormalizationForm.FormC), tok.Decode(actual));
        }
    }
}
