using System.Text.Json;
using Voxwerk.IO;
using Voxwerk.Numerics;
using Voxwerk.Talker;
using Voxwerk.Text;
using Xunit.Abstractions;

namespace Voxwerk.Tests;

/// <summary>
/// Parity for the VoiceDesign and CustomVoice checkpoints against <c>scripts/dump_reference.py --mode design|custom</c>.
/// Each key K reads <c>VOXWERK_{K}_MODEL_DIR</c> and <c>VOXWERK_{K}_REFERENCE_DIR</c>, e.g. DESIGN (1.7B VoiceDesign),
/// CUSTOM (0.6B CustomVoice) and CUSTOM17 (1.7B CustomVoice).
/// </summary>
public unsafe class ModeParityTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData("DESIGN")]
    [InlineData("CUSTOM")]
    [InlineData("CUSTOM17")]
    public void PromptLayersLogitsAndCodesMatch(string key)
    {
        string? modelDir = Environment.GetEnvironmentVariable($"VOXWERK_{key}_MODEL_DIR");
        string? refDir = Environment.GetEnvironmentVariable($"VOXWERK_{key}_REFERENCE_DIR");
        if (modelDir == null || refDir == null || !Directory.Exists(modelDir) || !Directory.Exists(refDir))
        {
            output.WriteLine($"SKIPPED: set VOXWERK_{key}_MODEL_DIR and VOXWERK_{key}_REFERENCE_DIR.");
            return;
        }
        Npy Ref(string name) => Npy.Load(Path.Combine(refDir, name + ".npy"));

        var cfg = Qwen3TtsConfig.Load(Path.Combine(modelDir, "config.json"));
        using var st = new SafeTensorsFile(Path.Combine(modelDir, "model.safetensors"));
        using var model = new TalkerModel(st, cfg);
        var tok = Qwen2Tokenizer.FromDirectory(modelDir);
        var gen = new TalkerGenerator(model, tok);
        var par = new Parallelism(0);

        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(refDir, "cases.json")));
        bool instructSupported = cfg.TtsModelType == "voice_design" || cfg.TtsModelSize != "0b6";
        foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            string name = c.GetProperty("name").GetString()!;
            string? speaker = c.TryGetProperty("speaker", out var s) ? s.GetString() : null;
            string? instruct = c.TryGetProperty("instruct", out var i) && instructSupported ? i.GetString() : null;
            var prompt = gen.BuildPrompt(c.GetProperty("text").GetString()!, c.GetProperty("language").GetString()!,
                null, speaker, instruct, null, null, nonStreaming: true, par);
            var refIn = Ref($"{name}_prefill_in");
            Assert.Equal(refIn.Shape[0], prompt.Length);
            Fixtures.AssertClose(output, $"{name} prefill input", refIn.Data, prompt.Prefill, 1e-5);

            var stack = model.Model;
            stack.LayerObserver = (layer, h, tokens) =>
            {
                if (layer % 9 == 0 || layer == stack.LayerCount - 1)
                    Fixtures.AssertClose(output, $"{name} layer {layer}", Ref($"{name}_layer{layer}").Data, new ReadOnlySpan<float>(h, tokens * stack.Hidden), 2e-4);
            };
            try
            {
                using var cache = stack.CreateCache(prompt.Length);
                var x = (float[])prompt.Prefill.Clone();
                var y = new float[x.Length];
                fixed (float* xp = x, yp = y) stack.Forward(xp, prompt.Length, cache, yp, par);
            }
            finally
            {
                stack.LayerObserver = null;
            }

            var logits = new List<float[]>();
            var cpLogits = new List<float[]>();
            gen.Trace = (n, t) =>
            {
                if (n == "talker_logits") logits.Add(t);
                if (n == "cp_logits") cpLogits.Add(t);
            };
            var expected = Ref($"{name}_codes");
            var frames = gen.Generate(prompt, GenerationOptions.Greedy with { MaxNewTokens = expected.Shape[0] + 1 }, par);
            gen.Trace = null;
            var refLogits = Ref($"{name}_logits");
            int V = refLogits.Shape[1];
            Fixtures.AssertClose(output, $"{name} talker logits step 0", refLogits.Data.AsSpan(0, V), logits[0], 1e-3);
            Fixtures.AssertClose(output, $"{name} talker logits step 3", refLogits.Data.AsSpan(3 * V, V), logits[3], 1e-3);
            var refCp = Ref($"{name}_cp_logits");
            int cv = refCp.Shape[1];
            Fixtures.AssertClose(output, $"{name} code predictor logits 0", refCp.Data.AsSpan(0, cv), cpLogits[0], 1e-3);
            Fixtures.AssertClose(output, $"{name} code predictor logits 20", refCp.Data.AsSpan(20 * cv, cv), cpLogits[20], 1e-3);
            var exp = expected.Ints;
            int mismatches = 0;
            for (int f = 0; f < Math.Min(frames.Count, expected.Shape[0]); f++)
                for (int g = 0; g < 16; g++)
                    if (frames[f][g] != exp[f * 16 + g]) mismatches++;
            output.WriteLine($"{name}: frames {frames.Count} (reference {expected.Shape[0]}), code mismatches {mismatches} / {expected.Shape[0] * 16}");
            Assert.Equal(expected.Shape[0], frames.Count);
            Assert.Equal(0, mismatches);
        }
    }
}
