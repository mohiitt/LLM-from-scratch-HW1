# Experiment 2 - Architecture Comparison (fill after runs)

## Objective
Compare recurrent neural networks with Transformer variants under the same character-level training-token budget and held-out validation protocol.

## Experimental design
- Dataset: Tiny Shakespeare, character tokenizer; fixed validation tail.
- Architectures: vanilla RNN (256), GRU (256), LSTM (256), LSTM (512), textbook Transformer, modern Transformer.
- Random seeds: 1337 and 42.
- Each run: [choose full 16,384,000 / quick 8,192,000] tokens.
- Shared: batch size 32, sequence length 256, same validation text, bpb as primary comparison metric.
- Optimizer: AdamW and cosine schedule, with architecture-appropriate original learning rate / weight decay from the professor's sweep.
- Hardware: MacBook Air M4, PyTorch MPS, fp32.

## Results
Insert `task2_results_table.md` here after the training sweep completes.

**Figure 1.** Validation bpb versus training tokens (two-seed average); insert `task2_learning_curves.png`.

**Figure 2.** Training versus validation loss for 256- and 512-wide LSTMs; insert `task2_lstm_overfit.png`.

## Interpretation (to write from actual measurements)

**Why does the vanilla RNN fall behind?** Explain how repeated Jacobian products during backpropagation through time can cause vanishing gradients, unlike gated memory in LSTM/GRU. Connect the explanation to the measured bpb gap and validation curves.

**Is the larger LSTM better or worse?** Compare 256 versus 512 LSTM validation bpb across both seeds. Inspect training and validation loss trajectories for evidence of overfitting. Do not claim overfitting unless your figures support it.

**Does the modern recipe help?** Compare the modern Transformer with the GPT-2-style Transformer, noting that the modern configuration combines RMSNorm, RoPE, SwiGLU and removal of bias (not an isolated ablation).

**Which model wins?** State which architecture has the lowest mean validation bpb. Discuss seed variability and parameter count; avoid making definitive claims if the gap is comparable to two-seed noise.

**Context-length effectiveness** is addressed in Task 3; do not infer it from Task 2 alone.
