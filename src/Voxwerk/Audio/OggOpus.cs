using System.Buffers.Binary;
using System.Text;
using Concentus;
using Concentus.Enums;
using Concentus.Structs;

namespace Voxwerk.Audio;

/// <summary>
/// Minimal Ogg Opus (RFC 7845) encoder / decoder for mono audio, built on the pure-managed
/// <see href="https://github.com/lostromb/concentus">Concentus</see> Opus codec (no native dependencies).
/// Files are standard <c>.opus</c> files playable by any Opus-capable player.
/// </summary>
public static class OggOpus
{
    private const int FrameMs = 20;
    private const int PacketsPerPage = 50;   // ~1 s of audio per Ogg page

    /// <summary>Encodes mono PCM (any Opus rate: 8/12/16/24/48 kHz) to an Ogg Opus byte array.</summary>
    /// <param name="pcm">Mono samples in [-1, 1].</param>
    /// <param name="sampleRate">Sample rate of <paramref name="pcm"/>.</param>
    /// <param name="bitrate">Target bitrate in bits per second (24 kbps is transparent for speech cloning).</param>
    /// <param name="tags">Optional Vorbis comments (e.g. title, artist).</param>
    public static byte[] Encode(ReadOnlySpan<float> pcm, int sampleRate = 24000, int bitrate = 24000, IReadOnlyDictionary<string, string>? tags = null)
    {
        var enc = OpusCodecFactory.CreateEncoder(sampleRate, 1, OpusApplication.OPUS_APPLICATION_AUDIO);
        enc.Bitrate = bitrate;
        enc.Complexity = 10;
        // CELT-only: Concentus' SILK path does not round-trip reliably, and CELT at 24 kbps is transparent for voice cloning.
        enc.ForceMode = OpusMode.MODE_CELT_ONLY;
        int frame = sampleRate * FrameMs / 1000;
        int scale = 48000 / sampleRate;                 // granule positions are in 48 kHz units
        int preSkip48 = enc.Lookahead * scale;
        uint serial = (uint)Random.Shared.Next();
        var output = new MemoryStream();
        int pageSeq = 0;

        // Header packets: OpusHead and OpusTags, each on its own page.
        var head = new byte[19];
        Encoding.ASCII.GetBytes("OpusHead").CopyTo(head, 0);
        head[8] = 1;                                    // version
        head[9] = 1;                                    // channels
        BinaryPrimitives.WriteUInt16LittleEndian(head.AsSpan(10), (ushort)preSkip48);
        BinaryPrimitives.WriteUInt32LittleEndian(head.AsSpan(12), (uint)sampleRate);
        WritePage(output, [head], 0, serial, pageSeq++, bos: true, eos: false);
        WritePage(output, [OpusTags(tags)], 0, serial, pageSeq++, bos: false, eos: false);

        var packets = new List<byte[]>();
        var pcmFrame = new float[frame];
        var packet = new byte[4000];
        long encodedSamples = 0;
        int total = pcm.Length;
        int frames = (total + enc.Lookahead + frame - 1) / frame;   // flush the encoder lookahead
        for (int f = 0; f < frames; f++)
        {
            int start = f * frame;
            pcmFrame.AsSpan().Clear();
            if (start < total) pcm.Slice(start, Math.Min(frame, total - start)).CopyTo(pcmFrame);
            int len = enc.Encode(pcmFrame, frame, packet, packet.Length);
            packets.Add(packet.AsSpan(0, len).ToArray());
            encodedSamples += frame;
            bool last = f == frames - 1;
            if (packets.Count == PacketsPerPage || last)
            {
                // The final granule position trims the padding: pre-skip + real samples.
                long granule = last ? preSkip48 + (long)total * scale : encodedSamples * scale;
                WritePage(output, packets, granule, serial, pageSeq++, bos: false, eos: last);
                packets.Clear();
            }
        }
        return output.ToArray();
    }

    /// <summary>Decodes an Ogg Opus stream (mono or stereo, mixed to mono) to PCM at <paramref name="sampleRate"/>.</summary>
    public static float[] Decode(Stream stream, int sampleRate = 24000)
    {
        var packets = ReadPackets(stream, out long lastGranule);
        if (packets.Count < 2 || !Encoding.ASCII.GetString(packets[0], 0, Math.Min(8, packets[0].Length)).Equals("OpusHead"))
            throw new InvalidDataException("Not an Ogg Opus stream.");
        var head = packets[0];
        int channels = head[9];
        int preSkip48 = BinaryPrimitives.ReadUInt16LittleEndian(head.AsSpan(10));
        int scale = 48000 / sampleRate;
        var dec = OpusCodecFactory.CreateDecoder(sampleRate, channels);
        var outPcm = new List<float>();
        var buf = new float[sampleRate * 120 / 1000 * channels];   // max Opus frame (120 ms)
        for (int i = 2; i < packets.Count; i++)
        {
            // Concentus needs the packet's exact duration as frame size (not the 120 ms maximum).
            int frameSize = OpusPacketInfo.GetNumSamples(packets[i], sampleRate);
            if (frameSize <= 0) continue;
            int n = dec.Decode(packets[i], buf.AsSpan(0, frameSize * channels), frameSize, false);
            for (int s = 0; s < n; s++)
            {
                float v = 0;
                for (int c = 0; c < channels; c++) v += buf[s * channels + c];
                outPcm.Add(v / channels);
            }
        }
        int skip = Math.Min(outPcm.Count, preSkip48 / scale);
        int length = outPcm.Count - skip;
        if (lastGranule > 0) length = (int)Math.Clamp((lastGranule - preSkip48) / scale, 0, length);
        return outPcm.GetRange(skip, length).ToArray();
    }

    public static float[] Decode(byte[] data, int sampleRate = 24000) => Decode(new MemoryStream(data), sampleRate);

    private static byte[] OpusTags(IReadOnlyDictionary<string, string>? tags)
    {
        var ms = new MemoryStream();
        ms.Write(Encoding.ASCII.GetBytes("OpusTags"));
        var vendor = Encoding.UTF8.GetBytes("Voxwerk (Concentus)");
        Span<byte> u32 = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(u32, (uint)vendor.Length);
        ms.Write(u32);
        ms.Write(vendor);
        var list = tags?.ToList() ?? [];
        BinaryPrimitives.WriteUInt32LittleEndian(u32, (uint)list.Count);
        ms.Write(u32);
        foreach (var (k, v) in list)
        {
            var c = Encoding.UTF8.GetBytes($"{k.ToUpperInvariant()}={v}");
            BinaryPrimitives.WriteUInt32LittleEndian(u32, (uint)c.Length);
            ms.Write(u32);
            ms.Write(c);
        }
        return ms.ToArray();
    }

    private static void WritePage(Stream s, IReadOnlyList<byte[]> packets, long granule, uint serial, int seq, bool bos, bool eos)
    {
        var lacing = new List<byte>();
        foreach (var p in packets)
        {
            int len = p.Length;
            while (len >= 255)
            {
                lacing.Add(255);
                len -= 255;
            }
            lacing.Add((byte)len);
        }
        if (lacing.Count > 255) throw new InvalidOperationException("Too many segments for one Ogg page.");
        int bodyLen = packets.Sum(p => p.Length);
        var page = new byte[27 + lacing.Count + bodyLen];
        Encoding.ASCII.GetBytes("OggS").CopyTo(page, 0);
        page[4] = 0;
        page[5] = (byte)((bos ? 0x02 : 0) | (eos ? 0x04 : 0));
        BinaryPrimitives.WriteInt64LittleEndian(page.AsSpan(6), granule);
        BinaryPrimitives.WriteUInt32LittleEndian(page.AsSpan(14), serial);
        BinaryPrimitives.WriteUInt32LittleEndian(page.AsSpan(18), (uint)seq);
        page[26] = (byte)lacing.Count;
        lacing.CopyTo(page, 27);
        int at = 27 + lacing.Count;
        foreach (var p in packets)
        {
            p.CopyTo(page, at);
            at += p.Length;
        }
        BinaryPrimitives.WriteUInt32LittleEndian(page.AsSpan(22), Crc(page));
        s.Write(page);
    }

    private static List<byte[]> ReadPackets(Stream s, out long lastGranule)
    {
        var packets = new List<byte[]>();
        var current = new MemoryStream();
        var header = new byte[27];
        lastGranule = 0;
        while (ReadExactly(s, header))
        {
            if (header[0] != 'O' || header[1] != 'g' || header[2] != 'g' || header[3] != 'S') throw new InvalidDataException("Bad Ogg page.");
            long granule = BinaryPrimitives.ReadInt64LittleEndian(header.AsSpan(6));
            if (granule > 0) lastGranule = granule;
            var lacing = new byte[header[26]];
            if (!ReadExactly(s, lacing)) break;
            foreach (byte l in lacing)
            {
                var seg = new byte[l];
                if (!ReadExactly(s, seg)) throw new EndOfStreamException();
                current.Write(seg);
                if (l < 255)
                {
                    packets.Add(current.ToArray());
                    current.SetLength(0);
                }
            }
        }
        return packets;
    }

    private static bool ReadExactly(Stream s, byte[] buf)
    {
        int read = 0;
        while (read < buf.Length)
        {
            int n = s.Read(buf, read, buf.Length - read);
            if (n <= 0) return read == 0 ? false : throw new EndOfStreamException();
            read += n;
        }
        return true;
    }

    private static readonly uint[] CrcTable = BuildCrcTable();

    private static uint[] BuildCrcTable()
    {
        var t = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint r = i << 24;
            for (int j = 0; j < 8; j++) r = (r & 0x80000000) != 0 ? (r << 1) ^ 0x04C11DB7 : r << 1;
            t[i] = r;
        }
        return t;
    }

    private static uint Crc(byte[] data)
    {
        uint crc = 0;
        foreach (byte b in data) crc = (crc << 8) ^ CrcTable[((crc >> 24) & 0xFF) ^ b];
        return crc;
    }
}
