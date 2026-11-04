metric: `val_bpb` (lower is better)

```
              n_seeds  val_bpb_mean  val_bpb_std  params_M  train_min  delta_vs_gpt_textbook  paired_p
name                                                                                                  
gpt_modern          2        2.1159       0.0010    3.1626    14.0203                -0.0694    0.0039
lstm                2        2.1371       0.0048    1.0698     3.2541                -0.0481    0.0596
gru                 2        2.1600       0.0018    0.8067     7.2538                -0.0253    0.0037
lstm_512            2        2.1701       0.0132    4.2368    10.7389                -0.0152    0.3157
gpt_textbook        2        2.1852       0.0016    3.4383    11.0943                 0.0000       NaN
rnn                 2        2.4369       0.0056    0.2803     4.1356                 0.2517    0.0072
```
