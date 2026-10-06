"""前処理済み npz（codes [17, T] と loss_start）のデータセットと損失。"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .sequence import N_AUDIO, TEXT_PAD


class PromptedDialogues(Dataset):
    def __init__(self, directory: str | Path, max_frames: int):
        self.files = sorted(Path(directory).glob("*.npz"))
        assert self.files, f"no npz under {directory}"
        self.max_frames = max_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        codes = torch.from_numpy(data["codes"].astype(np.int64))[:, : self.max_frames]
        return codes, int(data["loss_start"])


def collate(batch, pad_id: int):
    """pad_id（LMModel.zero_token_id）で右詰めする。pad 位置は forward_train が損失から外す。"""
    length = max(codes.shape[1] for codes, _ in batch)
    codes = torch.full((len(batch), batch[0][0].shape[0], length), pad_id, dtype=torch.long)
    loss_mask = torch.zeros(len(batch), length, dtype=torch.bool)
    for i, (c, loss_start) in enumerate(batch):
        codes[i, :, : c.shape[1]] = c
        loss_mask[i, loss_start: c.shape[1]] = True
    return codes, loss_mask


def compute_loss(lm, out, codes, loss_mask, text_pad_weight=0.3, acoustic_weight=0.02, user_weight=0.0):
    """プロンプト区間を除いた損失。

    total = text + agent semantic + acoustic_weight * sum(agent acoustic)
            + user_weight * (user semantic + acoustic_weight * sum(user acoustic))
    テキストは PAD の重みを text_pad_weight に下げた重み付き平均。
    """
    text_target = codes[:, 0]
    text_mask = out.text_mask[:, 0] & loss_mask
    ce = F.cross_entropy(out.text_logits[:, 0][text_mask].float(), text_target[text_mask], reduction="none")
    weight = torch.where(text_target[text_mask] == TEXT_PAD, text_pad_weight, 1.0)
    parts = {"text": (ce * weight).sum() / weight.sum()}
    # キーの集合は常に同じにする（分散学習で rank ごとにキーが違うと集約が合わなくなる）。
    # 非 PAD が 1 つも無いバッチでは 0 になるので、ログ上の平均は少し低めに出る
    not_pad = text_target[text_mask] != TEXT_PAD
    parts["text_nonpad"] = ce[not_pad].mean().detach() if not_pad.any() else ce.new_zeros(())

    n_streams = lm.dep_q if user_weight > 0 else N_AUDIO
    audio = []
    for k in range(n_streams):
        mask = out.mask[:, k] & loss_mask
        audio.append(F.cross_entropy(out.logits[:, k][mask].float(), codes[:, lm.audio_offset + k][mask]))
    parts["agent_semantic"] = audio[0]
    parts["agent_acoustic"] = torch.stack(audio[1:N_AUDIO]).sum()
    total = parts["text"] + parts["agent_semantic"] + acoustic_weight * parts["agent_acoustic"]
    if user_weight > 0:
        parts["user_semantic"] = audio[N_AUDIO]
        parts["user_acoustic"] = torch.stack(audio[N_AUDIO + 1:]).sum()
        total = total + user_weight * (parts["user_semantic"] + acoustic_weight * parts["user_acoustic"])
    parts["total"] = total
    return parts
