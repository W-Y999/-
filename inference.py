import torch
from model import MultiModalRegressor
from train import dummy_tokenizer

def load_model(path="checkpoint.pt"):
    model = MultiModalRegressor()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(model, text_answer, text_rule, numeric_feats):
    ta = torch.tensor(dummy_tokenizer(text_answer)).unsqueeze(0)
    tr = torch.tensor(dummy_tokenizer(text_rule)).unsqueeze(0)
    num = torch.tensor(numeric_feats).float().unsqueeze(0)

    with torch.no_grad():
        pred = model(ta, tr, num)
    return pred.item()


if __name__ == "__main__":
    model = load_model()
    score = predict(model, "服务很好", "根据态度与速度评分", [5, 4.2, 0.5])
    print("Pred score =", score)
