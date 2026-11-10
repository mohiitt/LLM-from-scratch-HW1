## 2. Task 1 — Metrics by Hand

### 2.1 Objective and experimental setup

This experiment checks the relationship between cross-entropy loss, perplexity, bits per token and bits per byte (bpb). I trained a two-layer, 256-hidden-unit character LSTM on Tiny Shakespeare and compared its metrics with two one-step models using character and byte tokenizers. All runs used seed 1337, PyTorch on the Mac M4 MPS backend in FP32 precision, and the same held-out text. The trained model contained 1,069,824 parameters, ran for 2,000 steps, processed 16,384,000 training tokens, and took 167.34 seconds to train. Its evaluation covered 111,488 tokens and 111,488 bytes.

The two nearly untrained LSTMs were each allowed exactly one optimizer step (8,192 tokens). A one-step result approximates, but is not identical to, a strictly zero-step model. The character model had 1,069,824 parameters, and the byte model had 1,118,720 parameters because its embedding vocabulary was larger.

### 2.2 Manual metric verification

The trained character model saved `val_loss = 1.4932519153678405`, `val_tokens = 111488`, and `val_bytes = 111488` in `metrics.json`. Cross-entropy loss is the mean negative log-probability, in **nats per predicted token**. The model's loss is read from the file; recovering an independently measured loss would require the underlying token probabilities. The derived metrics can be checked arithmetically:

- **Validation loss:** L = 1.493251915 nats/token.
- **Perplexity:** exp(L) = exp(1.493251915) = **4.451548**.
- **Bits/token:** L / ln(2) = **2.154307** bits/token.
- **Bits/byte:** [L / ln(2)] × [val_tokens / val_bytes] = 2.154307 × (111488 / 111488) = **2.154307** bits/byte.

The recalculated perplexity and bpb agree with the logged `val_ppl = 4.451548064601809` and `val_bpb = 2.1543071330991297` within rounding precision. Because the validation characters are single-byte in this dataset, bits per token equals bits per byte for these character-token runs.

| Measure | Trained character LSTM | Logged value | Match? |
|---|---:|---:|---|
| Loss (nats/token) | 1.493252 | 1.493252 | Source value |
| Perplexity | 4.451548 | 4.451548 | Yes |
| Bits/token | 2.154307 | — | Derived |
| Bits/byte | 2.154307 | 2.154307 | Yes |

### 2.3 One-step models compared with ln(V)

For a uniform predictor over a vocabulary of size V, the theoretical loss is ln(V). The character tokenizer has V = 65, giving ln(65) = 4.174387. The byte tokenizer has V = 256, giving ln(256) = 5.545177. Comparing these reference values with the one-step experiment:

| Tokenizer | Vocabulary V | ln(V) | Measured one-step loss | Difference |
|---|---:|---:|---:|---:|
| Character | 65 | 4.174387 | 4.178698 | +0.004311 |
| Byte | 256 | 5.545177 | 5.687762 | +0.142584 |

The byte model also has an observed one-step **8.205705 bpb**, obtained as 5.687762 / ln(2), compared with **8.000000 bits/byte** for an exactly uniform byte predictor over 256 symbols. The character model gives **6.028587 bits/token** (and, for this dataset, bits/byte), compared with the uniform reference log2(65) = approximately **6.022368 bits/token**. These values are based on one optimizer step, not an untrained distribution.

The character model is extremely close to the uniform baseline. The byte model is farther from ln(256), but a randomly initialized model does not necessarily output exactly uniform probabilities, and these measurements were taken after one update. This difference alone does not demonstrate an implementation bug. The byte model's higher initial loss is expected because its vocabulary has more possible outcomes.

The trained character LSTM reduced validation bpb from **6.028587** in the one-step character run to **2.154307**, a reduction of approximately **64.27%**. Its perplexity also fell from **65.280818** to **4.451548**, showing that training substantially improved next-character prediction.

### 2.4 Relation to the separate Task 2 LSTM run

The Task 1 LSTM reports **2.154307 bpb**, while the Task 2 LSTM-256 with seed 1337 reports **2.133688 bpb**, a **0.020620-bpb** difference. These are separate training and evaluation runs, not duplicate readings from one checkpoint. Both have two layers, hidden width 256, 1,069,824 parameters, seed 1337, 2,000 steps, and 16,384,000 tokens of training exposure. Their saved evaluations, however, cover **111,488 validation characters (Task 1)** versus **111,360 (Task 2)**, so they are not identically evaluated. The Task 2 configuration explicitly uses sequences of 256 characters, batches of 32, AdamW, learning rate 0.002, and weight decay 0.0. Without a full side-by-side comparison of Task 1's saved configuration, I cannot attribute the 0.020620-bpb difference to a specific training or evaluation setting. I retain both recorded values and use Task 2 only for the controlled architecture comparison.

### 2.5 Interpretation

Loss and perplexity are defined per tokenizer token. Consequently, comparing perplexities between character and byte or subword tokenizers can be misleading: a token may represent different amounts of text. Bits per byte normalizes the total prediction cost by UTF-8 byte count, making it a more appropriate comparison metric across tokenizers. This task verified the logged metrics and established why bpb is used throughout the remaining experiments.
