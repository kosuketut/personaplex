"""学習用の 17 ストリーム系列（テキスト 1 + agent 音声 8 + user 音声 8）の組み立て。

プロンプト区間は推論時の `LMGen.step_system_prompts` と同じ並びにする:
    声プロンプト -> 無音 -> テキストプロンプト -> 無音
区間中、agent テキストは PAD（テキストプロンプト区間のみ役割文）、user 音声は 440 Hz 正弦波。
`LMGen` は最初の 1 フレームを初期トークンで上書きして捨てるので、学習系列でも先頭 1 フレームを落とす。
ここで作るのは遅延をかける前の系列で、遅延は `LMModel.forward_train` が適用する。
"""
from dataclasses import dataclass

import numpy as np

from moshi.models.lm import SILENCE_TOKENS, SINE_TOKENS

TEXT_PAD = 3
TEXT_EPAD = 0
N_AUDIO = 8
FRAME_RATE = 12.5


def wrap_with_system_tags(text: str) -> str:
    """moshi.server / moshi.offline と同じ規則。"""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def _frames(text: np.ndarray, agent: np.ndarray, user: np.ndarray) -> np.ndarray:
    return np.concatenate([text[None], agent, user], axis=0).astype(np.int64)


def _const(tokens: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(tokens[:, None], n, axis=1)


def build_prefix(voice_codes: np.ndarray | None, prompt_ids: list[int] | None, silence_frames: int) -> np.ndarray:
    """プロンプト区間 [17, Tp] を返す（先頭フレームはまだ落としていない）。"""
    parts = []
    if voice_codes is not None:
        n = voice_codes.shape[1]
        parts.append(_frames(np.full(n, TEXT_PAD), voice_codes, _const(SINE_TOKENS, n)))
    silence = _frames(np.full(silence_frames, TEXT_PAD), _const(SILENCE_TOKENS, silence_frames),
                      _const(SINE_TOKENS, silence_frames))
    parts.append(silence)
    if prompt_ids:
        n = len(prompt_ids)
        parts.append(_frames(np.asarray(prompt_ids), _const(SILENCE_TOKENS, n), _const(SINE_TOKENS, n)))
    parts.append(silence)
    return np.concatenate(parts, axis=1)


@dataclass
class Example:
    codes: np.ndarray  # [17, T]
    loss_start: int  # このフレーム以降が損失対象（プロンプト区間はマスク）


def build_example(prefix: np.ndarray, text: np.ndarray, agent: np.ndarray, user: np.ndarray) -> Example:
    dialogue = _frames(text, agent, user)
    codes = np.concatenate([prefix, dialogue], axis=1)[:, 1:]
    return Example(codes=codes, loss_start=prefix.shape[1] - 1)


def encode_word(tokenizer, word: str) -> list[int]:
    """単語をトークン化する。ベースモデルの学習規約に合わせ、語頭の「▁」は付けない。

    「▁ちなみに」のように「▁」付きでしか語彙に無い片は、「▁」を外すと未知語になるので元の片を使う。
    語彙に無い文字（未知語）は捨てる。未知語の ID は 0 で、テキストストリームの EPAD と同じ値なので、
    残すと「単語の直前」という誤った正解になる。
    """
    unk = tokenizer.unk_id()
    ids = []
    for i, piece in enumerate(tokenizer.encode(word, out_type=str)):
        piece_id = tokenizer.piece_to_id(piece)
        if i == 0 and piece.startswith("▁"):
            stripped = piece[1:]
            if not stripped:
                continue
            if tokenizer.piece_to_id(stripped) != unk:
                piece_id = tokenizer.piece_to_id(stripped)
        if piece_id != unk:
            ids.append(piece_id)
    return ids


def align_text(words: list[dict], tokenizer, n_frames: int) -> np.ndarray:
    """時刻付き単語列をテキストストリーム [T] に配置する。

    各単語のトークンを開始時刻のフレームから 1 フレーム 1 トークンで置き、直前が PAD なら EPAD にする。
    前の単語とぶつかる場合は後ろへずらす。末尾に収まらないトークンは黙って切らずエラーにする。
    """
    text = np.full(n_frames, TEXT_PAD, dtype=np.int64)
    cursor = 0
    for word in sorted(words, key=lambda w: w["start"]):
        ids = encode_word(tokenizer, word["word"])
        if not ids:
            continue
        start = max(int(word["start"] * FRAME_RATE), cursor)
        if start + len(ids) > n_frames:
            raise ValueError(f"word {word['word']!r} does not fit: frame {start}+{len(ids)} > {n_frames}")
        if start > 0 and text[start - 1] == TEXT_PAD:
            text[start - 1] = TEXT_EPAD
        text[start:start + len(ids)] = ids
        cursor = start + len(ids)
    return text
