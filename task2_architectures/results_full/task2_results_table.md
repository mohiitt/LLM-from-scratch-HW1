# Task 2: RNN, GRU, LSTM, and Transformer Comparison

All bpb numbers below are from real saved `metrics.json` files. Lower is better.

| Architecture | Parameters | Seed 1337 bpb | Seed 42 bpb | Mean +/- sample SD bpb |
|---|---:|---:|---:|---:|
| rnn | 280,320 | 2.4409 | 2.4330 | 2.4369 +/- 0.0056 |
| gru | 806,656 | 2.1612 | 2.1587 | 2.1600 +/- 0.0018 |
| lstm | 1,069,824 | 2.1337 | 2.1405 | 2.1371 +/- 0.0048 |
| lstm_512 | 4,236,800 | 2.1794 | 2.1608 | 2.1701 +/- 0.0132 |
| gpt_textbook | 3,438,336 | 2.1864 | 2.1841 | 2.1852 +/- 0.0016 |
| gpt_modern | 3,162,624 | 2.1165 | 2.1152 | 2.1159 +/- 0.0010 |

Run root: `/Users/mohit/LLM-from-scratch-HW1/DeepDataMiningLearning/DeepDataMiningLearning/llm/outputs/nanolm/sweeps/hw1_char_models_full`

Training dataset and token budget are checked across completed runs.
Periodic training curves use `eval_log.jsonl`; final values use full validation-set `metrics.json`.

Caution: all interpretations must use these results, not numbers in the tutorial.
