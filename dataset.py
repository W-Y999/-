import json
import torch
from torch.utils.data import Dataset

class QADataDataset(Dataset):
    """读取用户作答文本、规则文本、数值特征、标签"""

    def __init__(self, json_path, tokenizer, numeric_keys):
        self.data = [json.loads(line) for line in open(json_path, "r", encoding="utf8")]
        self.tokenizer = tokenizer
        self.numeric_keys = numeric_keys

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # tokenize
        t_answer = self.tokenizer(row["text_answer"])
        t_rule   = self.tokenizer(row["text_rule"])

        # numeric features
        numeric = torch.tensor([row[k] for k in self.numeric_keys], dtype=torch.float)

        label = torch.tensor(row["label"], dtype=torch.float)

        return {
            "text_answer": torch.tensor(t_answer, dtype=torch.long),
            "text_rule": torch.tensor(t_rule, dtype=torch.long),
            "numeric": numeric,
            "label": label
        }
