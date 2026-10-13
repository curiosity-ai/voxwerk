using System.Text.Json;
using Voxwerk.IO;
using Voxwerk.Numerics;
using Voxwerk.Talker;
using Voxwerk.Text;
using Xunit.Abstractions;

namespace Voxwerk.Tests;

public sealed class TalkerFixture : IDisposable
{
    public TalkerModel? Model { get; }
    public Qwen2Tokenizer? Tokenizer { get; }
    public string Text { get; } = "";
    public string RefText { get; } = "";

    public TalkerFixture()
    {
        if (Fixtures.ModelDir == null || Fixtures.ReferenceDir == null) return;
        var cfg = Qwen3TtsConfig.Load(Path.Combine(Fixtures.ModelDir, "config.json"));
        using var st = new SafeTensorsFile(Path.Combine(Fixtures.ModelDir, "model.safetensors"));
        Model = new TalkerModel(st, cfg);
        Tokenizer = Qwen2Tokenizer.FromDirectory(Fixtures.ModelDir);
        var cases = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(Path.Combine(Fixtures.ReferenceDir, "tok_cases.json")))!;
        Text = cases["assistant"]["<|im_start|>assistant\n".Length..^"<|im_end|>\n<|im_start|>assistant\n".Length];
        RefText = cases["ref"]["<|im_start|>assistant\n".Length..^"<|im_end|>\n".Length];
    }

    public void Dispose() => Model?.Dispose();
}

public unsafe class TalkerParityTests(ITestOutputHelper output, TalkerFixture fx) : IClassFixture<TalkerFixture>
{
    private VoiceClonePrompt Voice(bool icl)
    {
        var codes = Fixtures.Ref("ref_codes");
        int frames = codes.Shape[0];
        var c = codes.Ints;
        return new VoiceClonePrompt
        {
            SpeakerEmbedding = Fixtures.Ref("spk_embedding").Data,
            ReferenceCodes = icl ? Enumerable.Range(0, frames).Select(f => c.AsSpan(f * 16, 16).ToArray()).ToArray() : null,
            ReferenceText = icl ? fx.RefText : null,
        };
    }

    [Fact]
    public void IclPrefillLayersAndGreedyCodesMatch()
    {
        if (!Fixtures.Available(output)) return;
        var par = new Parallelism(0);
        var gen = new TalkerGenerator(fx.Model!, fx.Tokenizer!);
        var prompt = gen.BuildPrompt(fx.Text, "English", Voice(icl: true), nonStreaming: false, par);
        var refIn = Fixtures.Ref("talker_prefill_in");
        Assert.Equal(refIn.Shape[0], prompt.Length);
        Fixtures.AssertClose(output, "prefill input", refIn.Data, prompt.Prefill, 1e-5);

        // Layer-by-layer through the 28-layer talker.
        var stack = fx.Model!.Model;
        var errors = new List<string>();
        stack.LayerObserver = (layer, h, tokens) =>
        {
            var expected = Fixtures.Ref($"talker_prefill_layer{layer}").Data;
            Fixtures.AssertClose(output, $"talker layer {layer}", expected, new ReadOnlySpan<float>(h, tokens * stack.Hidden), 2e-4);
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

        // Greedy generation: logits per step and the exact codec frames.
        var logits = new List<float[]>();
        var cpLogits = new List<float[]>();
        gen.Trace = (name, t) =>
        {
            if (name == "talker_logits") logits.Add(t);
            if (name == "cp_logits") cpLogits.Add(t);
        };
        var expectedCodes = Fixtures.Ref("gen_codes");
        var frames = gen.Generate(prompt, GenerationOptions.Greedy with { MaxNewTokens = expectedCodes.Shape[0] + 1 }, par);
        var refLogits = Fixtures.Ref("talker_logits");
        int V = refLogits.Shape[1];
        for (int i = 0; i < Math.Min(4, logits.Count); i++)
            Fixtures.AssertClose(output, $"talker logits step {i}", refLogits.Data.AsSpan(i * V, V), logits[i], 1e-3);
        var refCp = Fixtures.Ref("cp_logits");
        int cv = refCp.Shape[1];
        for (int i = 0; i < Math.Min(refCp.Shape[0], cpLogits.Count); i += 7)
            Fixtures.AssertClose(output, $"code predictor logits {i}", refCp.Data.AsSpan(i * cv, cv), cpLogits[i], 1e-3);

        var exp = expectedCodes.Ints;
        output.WriteLine($"frames: {frames.Count} (reference {expectedCodes.Shape[0]})");
        int mismatches = 0;
        for (int f = 0; f < Math.Min(frames.Count, expectedCodes.Shape[0]); f++)
            for (int g = 0; g < 16; g++)
                if (frames[f][g] != exp[f * 16 + g]) mismatches++;
        output.WriteLine($"code mismatches: {mismatches} / {expectedCodes.Shape[0] * 16}");
        Assert.Equal(expectedCodes.Shape[0], frames.Count);
        Assert.Equal(0, mismatches);
    }

    [Fact]
    public void XVectorPromptAndGreedyCodesMatch()
    {
        if (!Fixtures.Available(output)) return;
        var par = new Parallelism(0);
        var gen = new TalkerGenerator(fx.Model!, fx.Tokenizer!);
        var prompt = gen.BuildPrompt(fx.Text, "English", Voice(icl: false), nonStreaming: false, par);
        var refIn = Fixtures.Ref("xvec_prefill_in");
        Assert.Equal(refIn.Shape[0], prompt.Length);
        Fixtures.AssertClose(output, "x-vector prefill input", refIn.Data, prompt.Prefill, 1e-5);
        var expectedCodes = Fixtures.Ref("xvec_gen_codes");
        var frames = gen.Generate(prompt, GenerationOptions.Greedy with { MaxNewTokens = expectedCodes.Shape[0] + 1 }, par);
        Assert.Equal(expectedCodes.Shape[0], frames.Count);
        Assert.Equal(expectedCodes.Ints, frames.SelectMany(f => f).ToArray());
    }
}
