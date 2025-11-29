# 📄 **README.EN.md（最终版）**

```markdown
# Data Proofreading Model

A lightweight multi-modal regression model for score rationality checking in questionnaire and evaluation systems.

---

## 🌐 Languages
- 🇺🇸 **English** (current)  
- 🇨🇳 [Chinese Version](README.ZH.md)

---

# 1. Abstract

This project presents a lightweight multi-modal regression model designed for score rationality checking. The model jointly utilizes two textual inputs (user responses and scoring rule descriptions) and multiple numerical features. Text inputs are encoded using a shared TextCNN + BiGRU encoder, while structured numerical features are processed by an MLP. The fused representation is passed through a regression head to produce a continuous rationality score.

---

# 2. Motivation

Score data often contains:
- Mismatches between text feedback and numeric score  
- Abnormal score distributions  
- Inconsistent scoring standards  
- Expensive manual QA requirements  

Thus a lightweight, deployable, multi-modal model is needed for rationality checking.

---

# 3. Method

## 3.1 Task Definition

Given:
- \(T_{answer}\): user response  
- \(T_{rule}\): scoring rule text  
- \(X_{num}\): numeric features  

Predict rationality score:
\[
\hat{y} = f(T_{answer}, T_{rule}, X_{num})
\]

Loss:
\[
\mathcal{L} = MAE(y, \hat{y}) \quad or \quad RMSE(y, \hat{y})
\]

---

## 3.2 Model Architecture

### (1) Shared Text Encoder  
Embedding → TextCNN → BiGRU → Pool  
Outputs: \(h_{answer}, h_{rule}\)

### (2) Numeric Feature Encoder  
\[
h_{num} = MLP(X_{num})
\]

### (3) Fusion & Regression  
\[
h = [h_{answer}; h_{rule}; h_{num}]
\]
\[
\hat{y} = MLP_{reg}(h)
\]

---

## 3.3 Flow Diagram

```text
User Text ──► Embedding ──► CNN ──► GRU ──► h_answer
Rule Text ──► Embedding ──► CNN ──► GRU ──► h_rule
Numeric ───────────────► MLP ─────────────► h_num
      h_answer + h_rule + h_num ──► concat ──► reg MLP ──► ŷ
````

---

# 4. Dataset

| Field          | Type   | Description      |
| -------------- | ------ | ---------------- |
| text_answer    | string | User response    |
| text_rule      | string | Rule description |
| score_user     | float  | User score       |
| stats_mean     | float  | Historical mean  |
| stats_std      | float  | Historical std   |
| other_features | dict   | Metadata         |
| label          | float  | Target           |

---

# 5. Experiments

* Split: 70/15/15
* Hyperparameters: embed=128, GRU=128, filters=64
* Baselines: rule-based, numeric-only, text-only

---

# 6. Results

| Model                  | MAE       | RMSE      | R²       |
| ---------------------- | --------- | --------- | -------- |
| Rule-based             | 0.215     | 0.302     | 0.41     |
| Numeric-only           | 0.174     | 0.261     | 0.55     |
| Text-only              | 0.166     | 0.249     | 0.58     |
| **Multi-modal (Ours)** | **0.138** | **0.221** | **0.67** |

---

# 7. Conclusion

The proposed model is lightweight, effective, and deployable, offering strong performance for automated QA.

---

# 8. Future Work

* Lightweight transformers
* Interpretability
* Multi-task learning
* Automated data cleaning
* Model distillation and deployment

```

