import torch
from torch.utils.data import DataLoader
from model import MultiModalRegressor
from dataset import QADataDataset


def dummy_tokenizer(text, max_len=64):
    """简单 tokenizer：按空格分词并映射到 hash ID。"""
    ids = [abs(hash(w)) % 30000 for w in text.split()]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


def collate(batch):
    """padding batch"""
    return {
        "text_answer": torch.stack([b["text_answer"] for b in batch]),
        "text_rule": torch.stack([b["text_rule"] for b in batch]),
        "numeric": torch.stack([b["numeric"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]).unsqueeze(-1),
    }


def train():
    numeric_keys = ["score_user", "score_stats_mean", "score_stats_std"]

    dataset = QADataDataset(
        "data.json",
        tokenizer=dummy_tokenizer,
        numeric_keys=numeric_keys
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)

    model = MultiModalRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(3):
        for batch in loader:
            pred = model(batch["text_answer"], batch["text_rule"], batch["numeric"])
            loss = criterion(pred, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), "checkpoint.pt")
    print("Model saved to checkpoint.pt")


if __name__ == "__main__":
    train()
