"""Dump layer-by-layer reference tensors from the official qwen-tts PyTorch implementation.

Usage:
    python dump_reference.py <model_dir> <ref_wav> <out_dir> [max_new_tokens]

Runs a deterministic (greedy) voice-clone generation and stores every intermediate the C# port is
compared against as .npy files in <out_dir>. The Voxwerk parity tests read them through the
VOXWERK_REFERENCE_DIR environment variable.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

torch.set_num_threads(1)
model_dir, ref_wav, out = sys.argv[1], sys.argv[2], sys.argv[3]
max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 24
os.makedirs(out, exist_ok=True)

TEXT = "Hello there! This is a quick test of the text to speech system."
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


def save(name, t):
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy() if t.is_floating_point() else t.detach().cpu().numpy().astype(np.int64)
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(t))


tts = Qwen3TTSModel.from_pretrained(model_dir, device_map="cpu", dtype=torch.float32)
model = tts.model
talker = model.talker
st = model.speech_tokenizer.model

# ---- text tokenizer -------------------------------------------------------------------------
tok_cases = {
    "assistant": tts._build_assistant_text(TEXT),
    "ref": tts._build_ref_text(REF_TEXT),
    "misc": "Numbers 1234567, émigré naïve café — 你好，世界！ It's  two  spaces\nnew line\t tab 😀 ok?",
}
for k, v in tok_cases.items():
    ids = tts._tokenize_texts([v])[0][0]
    save(f"tok_{k}", ids)
with open(os.path.join(out, "tok_cases.json"), "w", encoding="utf-8") as f:
    json.dump(tok_cases, f, ensure_ascii=False)

# ---- reference audio: mel + speaker embedding ------------------------------------------------
wav, sr = sf.read(ref_wav, dtype="float32")
assert sr == 24000
save("ref_wav", wav)
mel = mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000,
                      hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
save("spk_mel", mel[0])
se = model.speaker_encoder
acts = {}
hooks = []
for i, blk in enumerate(se.blocks):
    hooks.append(blk.register_forward_hook(lambda m, a, o, i=i: acts.__setitem__(f"spk_block{i}", o[0])))
hooks.append(se.mfa.register_forward_hook(lambda m, a, o: acts.__setitem__("spk_mfa", o[0])))
hooks.append(se.asp.register_forward_hook(lambda m, a, o: acts.__setitem__("spk_asp", o[0])))
with torch.inference_mode():
    spk = se(mel)[0]
for h in hooks:
    h.remove()
for k, v in acts.items():
    save(k, v)
save("spk_embedding", spk)

# ---- speech tokenizer encoder ---------------------------------------------------------------
enc = st.encoder
acts = {}
hooks = [
    enc.encoder.register_forward_hook(lambda m, a, o: acts.__setitem__("enc_seanet", o[0])),
    enc.encoder_transformer.register_forward_hook(lambda m, a, o: acts.__setitem__("enc_transformer", o[0][0])),
    enc.downsample.register_forward_hook(lambda m, a, o: acts.__setitem__("enc_downsample", o[0])),
]
for i, layer in enumerate(enc.encoder.layers):
    hooks.append(layer.register_forward_hook(lambda m, a, o, i=i: acts.__setitem__(f"enc_seanet_l{i}", o[0])))
with torch.inference_mode():
    codes = model.speech_tokenizer.encode(wav, sr=24000).audio_codes[0]
for h in hooks:
    h.remove()
for k, v in acts.items():
    save(k, v)
save("ref_codes", codes)

# ---- talker prefill / generation ------------------------------------------------------------
rec = {"prefill_in": None, "layers": [], "logits": [], "cp_logits": [], "cp_calls": 0, "talker_in": []}


def talker_model_hook(m, args, kwargs, o):
    emb = kwargs["inputs_embeds"]
    rec["talker_in"].append(emb[0])
    if emb.shape[1] > 1 and rec["prefill_in"] is None:
        rec["prefill_in"] = emb[0]
        rec["layers"] = [h[0] for h in o.hidden_states] if o.hidden_states is not None else []


hooks = [talker.model.register_forward_hook(talker_model_hook, with_kwargs=True)]
pre_layer_outs = []
for i, layer in enumerate(talker.model.layers):
    hooks.append(layer.register_forward_hook(
        lambda m, a, o, i=i: pre_layer_outs.append((i, o[0][0])) if o[0].shape[1] > 1 else None))
hooks.append(talker.codec_head.register_forward_hook(lambda m, a, o: rec["logits"].append(o[0, -1])))


def cp_hook(m, a, o):
    if rec["cp_calls"] < 2 * 15:
        rec["cp_logits"].append(o[0, -1])
    rec["cp_calls"] += 1


for head in talker.code_predictor.lm_head:
    hooks.append(head.register_forward_hook(cp_hook))

with torch.inference_mode():
    prompt = tts.create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=REF_TEXT, x_vector_only_mode=False)
    vc = tts._prompt_items_to_voice_clone_prompt(prompt)
    input_ids = tts._tokenize_texts([tts._build_assistant_text(TEXT)])
    ref_ids = [tts._tokenize_texts([tts._build_ref_text(REF_TEXT)])[0]]
    codes_list, _ = model.generate(input_ids=input_ids, ref_ids=ref_ids, voice_clone_prompt=vc, languages=["English"],
                                   non_streaming_mode=False, max_new_tokens=max_new, do_sample=False,
                                   subtalker_dosample=False, repetition_penalty=1.05)
for h in hooks:
    h.remove()

save("talker_prefill_in", rec["prefill_in"])
for i, t in pre_layer_outs:
    save(f"talker_prefill_layer{i}", t)
save("talker_logits", torch.stack(rec["logits"]))
save("cp_logits", torch.stack(rec["cp_logits"]))
save("talker_step_in", torch.cat(rec["talker_in"][1:], dim=0) if len(rec["talker_in"]) > 1 else torch.zeros(0))
gen_codes = codes_list[0]
save("gen_codes", gen_codes)

# ---- x-vector-only prompt (no ICL) -----------------------------------------------------------
rec2 = []
h = talker.model.register_forward_hook(
    lambda m, a, k, o: rec2.append(k["inputs_embeds"][0]) if k["inputs_embeds"].shape[1] > 1 else None, with_kwargs=True)
with torch.inference_mode():
    prompt = tts.create_voice_clone_prompt(ref_audio=(wav, sr), x_vector_only_mode=True)
    vc2 = tts._prompt_items_to_voice_clone_prompt(prompt)
    codes_x, _ = model.generate(input_ids=input_ids, ref_ids=[None], voice_clone_prompt=vc2, languages=["English"],
                                non_streaming_mode=False, max_new_tokens=max_new, do_sample=False,
                                subtalker_dosample=False, repetition_penalty=1.05)
h.remove()
save("xvec_prefill_in", rec2[0])
save("xvec_gen_codes", codes_x[0])

# ---- speech tokenizer decoder ---------------------------------------------------------------
dec = st.decoder
full_codes = torch.cat([codes, gen_codes], dim=0)
save("dec_codes", full_codes)
acts = {}
hooks = [
    dec.quantizer.register_forward_hook(lambda m, a, o: None),
    dec.pre_conv.register_forward_hook(lambda m, a, o: acts.__setitem__("dec_pre_conv", o[0])),
    dec.pre_transformer.register_forward_hook(lambda m, a, o: acts.__setitem__("dec_pre_transformer", o.last_hidden_state[0])),
]
for i, ups in enumerate(dec.upsample):
    hooks.append(ups.register_forward_hook(lambda m, a, o: None))
    hooks.append(ups[1].register_forward_hook(lambda m, a, o, i=i: acts.__setitem__(f"dec_upsample{i}", o[0])))
for i, blk in enumerate(dec.decoder):
    hooks.append(blk.register_forward_hook(lambda m, a, o, i=i: acts.__setitem__(f"dec_decoder{i}", o[0])))
with torch.inference_mode():
    q = dec.quantizer.decode(full_codes.T.unsqueeze(0))
    save("dec_quantized", q[0])
    wavs, fs = model.speech_tokenizer.decode([{"audio_codes": full_codes}])
for h in hooks:
    h.remove()
for k, v in acts.items():
    save(k, v)
save("dec_wav", wavs[0])
ref_len = codes.shape[0]
cut = int(ref_len / full_codes.shape[0] * wavs[0].shape[0])
sf.write(os.path.join(out, "py_greedy.wav"), wavs[0][cut:], fs)
print("done", {"ref_codes": tuple(codes.shape), "gen_codes": tuple(gen_codes.shape),
               "prefill": tuple(rec["prefill_in"].shape), "xvec_codes": tuple(codes_x[0].shape)})
