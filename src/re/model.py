"""
BioBERT 关系分类模型（论文3.4.1节）

架构：BioBERT + 类型化实体标记 + 实体起始位置拼接分类头

分类头：取 E1 和 E2 起始标记的隐状态，拼接后接全连接层。
相比 [CLS] 分类，这种方式能更直接捕捉实体对语义（论文引用 Milošević 2023）。
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BioBERTRelationClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.2",
        num_labels: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout)
        # 拼接 E1_start 和 E2_start 的隐状态（论文3.4.1节）
        self.classifier = nn.Linear(hidden * 2, num_labels)

    def resize_token_embeddings(self, new_num_tokens: int):
        self.bert.resize_token_embeddings(new_num_tokens)

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        e1_pos: torch.Tensor,          # (B,) E1起始标记位置
        e2_pos: torch.Tensor,          # (B,) E2起始标记位置
        labels: torch.Tensor = None,   # (B,) 训练时传入
        class_weights: torch.Tensor = None,
    ) -> dict:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)

        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        e1_repr = hidden[batch_idx, e1_pos]  # (B, H)
        e2_repr = hidden[batch_idx, e2_pos]  # (B, H)

        pooled = self.dropout(torch.cat([e1_repr, e2_repr], dim=-1))  # (B, 2H)
        logits = self.classifier(pooled)  # (B, num_labels)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            result["loss"] = loss_fn(logits, labels)

        return result


def build_class_weights(
    labels: list[int],
    num_labels: int,
    device: torch.device,
    label_names: list[str] = None,
    class_boost: dict = None,
) -> torch.Tensor:
    """基于类别频率倒数计算加权损失权重（论文3.4.3节公式3-2）。
    class_boost: 额外乘数，如 {"Substrate": 1.5}，在逆频率权重上再乘。
    """
    counts = torch.zeros(num_labels)
    for l in labels:
        counts[l] += 1
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    if class_boost and label_names:
        for name, mult in class_boost.items():
            if name in label_names:
                weights[label_names.index(name)] *= mult
    weights = weights / weights.sum() * num_labels  # 归一化
    return weights.to(device)
