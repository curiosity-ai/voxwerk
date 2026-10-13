using Voxwerk.Codec;
using Voxwerk.IO;
using Voxwerk.Numerics;
using Xunit.Abstractions;

namespace Voxwerk.Tests;

public sealed class CodecFixture : IDisposable
{
    public SpeechDecoder? Decoder { get; }
    public SpeechEncoder? Encoder { get; }

    public CodecFixture()
    {
        if (Fixtures.ModelDir == null || Fixtures.ReferenceDir == null) return;
        using var st = new SafeTensorsFile(Path.Combine(Fixtures.ModelDir, "speech_tokenizer", "model.safetensors"));
        Decoder = new SpeechDecoder(st);
        Encoder = new SpeechEncoder(st);
    }

    public void Dispose()
    {
        Decoder?.Dispose();
        Encoder?.Dispose();
    }
}

public unsafe class CodecParityTests(ITestOutputHelper output, CodecFixture fx) : IClassFixture<CodecFixture>
{
    /// <summary>Reference hooks capture PyTorch's [C, T] layout; ours is channels-last [T, C].</summary>
    internal static float[] Transpose(Npy n)
    {
        int c = n.Shape[0], t = n.Shape[1];
        var r = new float[n.Data.Length];
        for (int i = 0; i < c; i++)
            for (int j = 0; j < t; j++) r[j * c + i] = n.Data[i * t + j];
        return r;
    }

    [Fact]
    public void DecoderStagesMatch()
    {
        if (!Fixtures.Available(output)) return;
        var codesNpy = Fixtures.Ref("dec_codes");
        int frames = codesNpy.Shape[0];
        var flat = codesNpy.Ints;
        var codes = Enumerable.Range(0, frames).Select(f => flat.AsSpan(f * 16, 16).ToArray()).ToList();
        var par = new Parallelism(0);
        fx.Decoder!.Trace = (name, act) =>
        {
            var reference = Fixtures.Ref(name);
            var expected = name == "dec_pre_transformer" ? reference.Data : Transpose(reference);
            Fixtures.AssertClose(output, name, expected, act.AsSpan(), 1e-4);
        };
        float[] wav;
        try
        {
            wav = fx.Decoder.Decode(codes, par);
        }
        finally
        {
            fx.Decoder.Trace = null;
        }
        var refWav = Fixtures.Ref("dec_wav").Data;
        Fixtures.AssertClose(output, "waveform", refWav, wav, 1e-4, maxAbs: 1e-3);
    }

    [Fact]
    public void StreamingVocoderIsBitIdentical()
    {
        if (!Fixtures.Available(output)) return;
        var flat = Fixtures.Ref("dec_codes").Ints;
        var codes = Enumerable.Range(0, 40).Select(f => flat.AsSpan(f * 16, 16).ToArray()).ToList();
        var par = new Parallelism(0);
        var dec = fx.Decoder!;
        try
        {
            dec.Streaming = false;
            var full = dec.Decode(codes, par);
            dec.Streaming = true;
            dec.StreamingSegment = 7;
            var streamed = dec.Decode(codes, par);
            Assert.Equal(full.Length, streamed.Length);
            for (int i = 0; i < full.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(full[i]) == BitConverter.SingleToInt32Bits(streamed[i]), $"sample {i}: {full[i]} vs {streamed[i]}");
        }
        finally
        {
            dec.Streaming = true;
            dec.StreamingSegment = 32;
        }
    }

    [Fact]
    public void EncoderMatches()
    {
        if (!Fixtures.Available(output)) return;
        var wav = Fixtures.Ref("ref_wav").Data;
        var par = new Parallelism(0);
        fx.Encoder!.Trace = (name, act) =>
        {
            var reference = Fixtures.Ref(name);
            var expected = name == "enc_transformer" ? reference.Data : Transpose(reference);
            Fixtures.AssertClose(output, name, expected, act.AsSpan(), 1e-4);
        };
        int[][] codes;
        try
        {
            codes = fx.Encoder.Encode(wav, par);
        }
        finally
        {
            fx.Encoder.Trace = null;
        }
        var expectedCodes = Fixtures.Ref("ref_codes");
        Assert.Equal(expectedCodes.Shape[0], codes.Length);
        var exp = expectedCodes.Ints;
        int mismatches = 0;
        for (int f = 0; f < codes.Length; f++)
            for (int g = 0; g < 16; g++)
                if (codes[f][g] != exp[f * 16 + g]) mismatches++;
        output.WriteLine($"encoder code mismatches: {mismatches} / {codes.Length * 16}");
        Assert.Equal(0, mismatches);
    }
}
