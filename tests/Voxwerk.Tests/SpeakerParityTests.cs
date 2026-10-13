using Voxwerk.Audio;
using Voxwerk.IO;
using Voxwerk.Numerics;
using Voxwerk.Speaker;
using Xunit.Abstractions;

namespace Voxwerk.Tests;

public class SpeakerParityTests(ITestOutputHelper output)
{
    [Fact]
    public void MelAndSpeakerEmbeddingMatch()
    {
        if (!Fixtures.Available(output)) return;
        var wav = Fixtures.Ref("ref_wav").Data;
        var melRef = Fixtures.Ref("spk_mel");
        var mel = new MelSpectrogram().Compute(wav, out int frames);
        Assert.Equal(melRef.Shape[0], frames);
        Fixtures.AssertClose(output, "log-mel", melRef.Data, mel, 1e-5);

        using var st = new SafeTensorsFile(Path.Combine(Fixtures.ModelDir!, "model.safetensors"));
        using var enc = new SpeakerEncoder(st);
        enc.Trace = (name, act) =>
        {
            var r = Fixtures.Ref(name);
            var expected = r.Shape.Length == 2 && r.Shape[1] != 1 ? CodecParityTests.Transpose(r) : r.Data;
            Fixtures.AssertClose(output, name, expected, act.AsSpan(), 1e-4);
        };
        // Feed the reference mel so the encoder is checked independently of the STFT.
        var emb = enc.Forward(melRef.Data, frames, new Parallelism(0));
        Fixtures.AssertClose(output, "x-vector (ref mel)", Fixtures.Ref("spk_embedding").Data, emb, 1e-4);
        enc.Trace = null;
        var emb2 = enc.Forward(mel, frames, new Parallelism(0));
        Fixtures.AssertClose(output, "x-vector (our mel)", Fixtures.Ref("spk_embedding").Data, emb2, 1e-4);
    }
}
