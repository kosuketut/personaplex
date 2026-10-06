"""build_prefix / build_example が upstream の LMGen.step_system_prompts と同じ入力列を作ることを確認する。"""
import numpy as np
import torch

from japersonaplex.sequence import TEXT_PAD, align_text, build_example, build_prefix
from moshi.models import loaders
from moshi.models.lm import LMGen, LMModel, _delay_sequence

SILENCE_FRAMES = 6


def tiny_lm() -> LMModel:
    kwargs = dict(loaders._lm_kwargs)
    kwargs.update(dim=64, num_heads=4, num_layers=2, depformer_dim=32, depformer_num_heads=4,
                  depformer_num_layers=2, depformer_dim_feedforward=64, dep_q=16, hidden_scale=2)
    torch.manual_seed(0)
    return LMModel(device="cpu", dtype=torch.float32, **kwargs).eval()


def test_prefix_matches_lmgen():
    rng = np.random.default_rng(0)
    lm = tiny_lm()
    voice = rng.integers(0, 2048, size=(8, 9))
    prompt_ids = [11, 222, 3333, 44, 5]
    dialogue_len = 7
    text = rng.integers(4, 32000, size=dialogue_len)
    agent = rng.integers(0, 2048, size=(8, dialogue_len))
    user = rng.integers(0, 2048, size=(8, dialogue_len))

    recorded = []
    original = lm.forward_codes
    lm.forward_codes = lambda codes: (recorded.append(codes.clone()), original(codes))[1]

    gen = LMGen(lm, device="cpu", audio_silence_frame_cnt=SILENCE_FRAMES, text_prompt_tokens=prompt_ids,
                sample_rate=24000, frame_rate=12.5)
    gen.voice_prompt_audio = object()
    gen._encode_voice_prompt_frames = lambda mimi: [torch.from_numpy(voice[None, :, t:t + 1]) for t in range(voice.shape[1])]
    with gen.streaming(1):
        gen.step_system_prompts(mimi=None)
        for t in range(dialogue_len):
            gen.step(input_tokens=torch.from_numpy(user[None, :, t:t + 1]),
                     moshi_tokens=torch.from_numpy(agent[None, :, t:t + 1]),
                     text_token=int(text[t]))
    lmgen_inputs = torch.cat(recorded, dim=2)  # [1, 17, S]

    prefix = build_prefix(voice, prompt_ids, SILENCE_FRAMES)
    example = build_example(prefix, text, agent, user)
    codes = torch.from_numpy(example.codes)[None]
    initial = lm._get_initial_token()
    delayed = torch.cat([initial, _delay_sequence(lm.delays, codes, initial)], dim=2)[:, :, :-1]

    assert example.loss_start == voice.shape[1] + 2 * SILENCE_FRAMES + len(prompt_ids) - 1
    assert lmgen_inputs.shape == delayed.shape, (lmgen_inputs.shape, delayed.shape)
    assert torch.equal(lmgen_inputs, delayed)


class FakeTokenizer:
    def unk_id(self):
        return 0

    def encode(self, word, out_type=str):
        return ["▁"] + list(word)

    def piece_to_id(self, piece):
        return 100 + ord(piece) % 1000


def test_align_text():
    words = [{"word": "はい", "start": 0.16, "end": 0.4}, {"word": "そう", "start": 0.3, "end": 0.6},
             {"word": "ね", "start": 1.0, "end": 1.1}]
    text = align_text(words, FakeTokenizer(), 16)
    ids = lambda w: [FakeTokenizer().piece_to_id(c) for c in w]
    assert text[1] == 0 and list(text[2:4]) == ids("はい")
    assert list(text[4:6]) == ids("そう")  # ぶつかるので後ろへ
    assert text[11] == 0 and text[12] == ids("ね")[0]
    assert (text[13:] == TEXT_PAD).all()


def test_flip_number_matching():
    from japersonaplex.scoring import mentions, normalize_numbers

    assert normalize_numbers("二千八百五十円と十七時と百円") == "2850円と17時と100円"
    assert mentions("十七時", "営業時間は17時までです")
    assert mentions("十七時", "十時から十七時まで")
    assert not mentions("八時", "18時までです")
    assert not mentions("五百円", "千五百円です")
    assert mentions("千百円", "コートは1,100円です")
    assert not mentions("五百円", "コーヒーは2,500円です")
    assert mentions("二千円", "2千円です")
    assert mentions("二十時", "２０時までです")
    assert normalize_numbers("2千5百円") == "2500円"


def test_encode_word_real_tokenizer():
    """語頭の「▁」付きでしか語彙に無い片と、語彙に無い文字の扱い。"""
    import glob

    import pytest
    import sentencepiece

    from japersonaplex.sequence import TEXT_EPAD, encode_word

    paths = glob.glob("checkpoints/llm-jp-moshi-v1-pp16/tokenizer_spm_32k_3.model")
    if not paths:
        pytest.skip("tokenizer not available")
    sp = sentencepiece.SentencePieceProcessor(paths[0])
    for word in ("ちなみに", "こうして", "従って", "ところが", "よって"):
        ids = encode_word(sp, word)
        assert ids and TEXT_EPAD not in ids, (word, ids)
        assert sp.decode(ids) == word, (word, sp.decode(ids))
    assert encode_word(sp, "こんにちは") == [sp.piece_to_id("こんにち"), sp.piece_to_id("は")]
    assert TEXT_EPAD not in encode_word(sp, "OKです")
