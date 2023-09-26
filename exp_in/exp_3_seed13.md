## Experiment 3:

Random seed: 13

Hyperpatameters:

```bash
Data(
    x=[19793, 8710],
    y=[19793],
    train_pos_edge_index=[2, 57079],
    test_pos_edge_index=[2, 3171],
    test_neg_edge_index=[2, 3171],
    val_pos_edge_index=[2, 3171],
    val_neg_edge_index=[2, 3171]
)

Model type:  gcn

Epochs:  2000
```

```bash
nohup python train_gnn.py --dataset Cora --gnn gcn --epochs 2000 --random_seed 13 > gcn_original_Cora_13.txt &
```

Original Model results:

```python
Test Logs:
{
    'test_loss': 0.65680330991745,
    'test_dt_auc': 0.9700824177162934,
    'test_dt_aup': 0.9696732106111076
}
```

### Deleting 100 edges in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 13 > gcn_gnndelete_in_0_100_Cora_13.txt &
```

Deleting the following edges (index):

[ 0, 1, 2, 4, 6, 7, 8, 9, 14, 16, 17, 18, 19, 21,
22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
37, 38, 39, 40, 41, 43, 44, 45, 47, 48, 49, 55, 58, 63,
64, 65, 66, 67, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80,
81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
96, 97, 99, 100, 102, 104, 105, 107, 108, 109, 110, 111, 112, 114,
117, 119, 120, 121, 124, 126, 128, 129, 131, 133, 134, 136, 137, 140,
141, 142]

Results:

```python
Test Logs:
{
    'test_loss': 0.696242094039917,
    'test_dt_auc': 0.7666245393819999,
    'test_dt_aup': 0.7875495502611549,
    'test_df_auc': 0.7822333,
    'test_df_aup': 0.8616442744775553,
    'test_df_logit_mean': 0.5007986688613891,
    'test_df_logit_std': 0.0014816733047966042
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▇▆▄▄▃▂▂▂▂▂▂▃▂▁▂▁▂▂▁▂▂▂▂▂▂▂▂▂▁▂▁▁▂▁▁▁▂▁▂
wandb:            loss_r ▇▄█▅▂▃▄▃▄▁▂▅▃▃▂▆▇▂▇▇▄▄▄▆▅▆▂▂▇▇▄▅▅▄▂▃▄▂▅▆
wandb:        train_loss ▃█▅▇▃▅▅▅▅▅▁▄▃▅▆▄▅▆▄▆▇▇█▅▄▃▇▆▃▇▃▄▄▅▅▅▃▄▆▆
wandb:      train_loss_l █▇▅▄▄▃▂▂▂▃▂▂▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▂▂▁▁▂▂▁▂
wandb:      train_loss_r ▁▃▄▄▃▁▃▅▁▄▅▄▃▃▂▄▄▂▃▄▃▅▄▅▃▅▄▁▆▅▃▅▃▃▅▃▃█▃▅
wandb:        train_time █▆▂▃▅▅▆▄▃▆▃▁▄▅▃▃▆▇▅▃▆▇▃▄▆▃▆▃▄▃▇▅▅▄▃▃▆▁▅▄
wandb:        val_df_auc █▄▃▅▂▂▂▁▂▃▃▃▂▃▂▂▂▂▃▃▂▃▂▂▂▃▃▃▃▂▂▂▂▃▂▂▂▂▂▂
wandb:        val_df_aup █▃▂▄▂▂▂▁▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂
wandb: val_df_logit_mean ▁▁▃▃▄▄▅▆▆▆▆▆▅▆▇▇▇▇▇▇▇▆▇▇▆▇▇▇▇█▇█▇▇▇█▇▇▇▇
wandb:  val_df_logit_std ▁▂▃▄▅▅▆▆▆▇▆▆▅▆▇▇▇▇▇▇▇▆▇▇▇▇▇▇▇███▇▇▇▇▇▇▇▇
wandb:        val_dt_auc ▁▂▄▆▅▄▅▃▆▇▇▇▄█▇▄▅▆▇█▆▇▅▄▄▆▆▆▇▇▇▇▆▇▆▆▆▆▇▇
wandb:        val_dt_aup ▁▂▄▆▅▅▆▅▇▇▇▇▆█▇▆▇▇▇█▇█▇▆▆▇▇▇▇██▇▇▇▇▇▇▇█▇
wandb:          val_loss █▇▅▄▄▃▂▂▂▂▂▂▃▂▂▂▁▂▁▁▂▂▂▂▂▂▂▂▂▁▁▁▁▂▁▁▁▂▁▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.03174
wandb:            loss_r 1.13632
wandb:        train_loss 0.58403
wandb:      train_loss_l 0.03174
wandb:      train_loss_r 1.13632
wandb:        train_time 1.5021
wandb:        val_df_auc 0.70021
wandb:        val_df_aup 0.81551
wandb: val_df_logit_mean 0.56371
wandb:  val_df_logit_std 0.0701
wandb:        val_dt_auc 0.82473
wandb:        val_dt_aup 0.84928
wandb:          val_loss 0.68726
```

### Deleting next 100 edges in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --seqlearn True --epochs 2000 --random_seed 13 > gcn_gnndelete_in_100_200_Cora_13.txt &
```

Deleting the following edges (index):

Results:

```python
{
'test_loss': 0.6935510635375977,
'test_dt_auc': 0.7903102968889557,
'test_dt_aup': 0.8083124240478015,
'test_df_auc': 0.8473544,
'test_df_aup': 0.9010189405365433,
'test_df_logit_mean': 0.5006920877099037,
'test_df_logit_std': 0.0016916673072391263
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l ████▇▅▆▅▅▄▄▄▃▃▃▃▂▃▂▁▃▃▃▄▃▃▃▃▃▁▃▃▂▂▂▂▂▂▂▂
wandb:            loss_r ▁█▆▃▄▄▄▂▃▁▃▅▄▂▄▂▅▂▆▄▄▃▅▃▄▂▂▄▄▄▅▅▃▂▃▃▃▆▄▅
wandb:        train_loss ▃▆▃▆▄▄▄▅▅█▁▄▄▄▄▃▂▅▂█▅▃▆▃▃▃▄▃▂▇▃▂▅▄▄▂▂▄▅▆
wandb:      train_loss_l ████▆▅▆▅▅▄▅▄▄▃▃▃▂▃▂▁▃▃▃▄▃▃▃▃▃▁▃▃▂▂▂▂▂▂▂▂
wandb:      train_loss_r ▃▅▄▃▅▄▅▆▃▄▅▅▇▄▅▃▇▃▄▇▁▅▆▄▃▃▃▂▅█▆▆▅▃▅▂▄▇▆█
wandb:        train_time ▅▇▇▇▇▇▆▇█▇▇▇█▆▇▇▇█▇▅▇▇▇▇▇▇▆▇▇▇▇▇▆▇▇▇▇▇▇▁
wandb:        val_df_auc ▄▄█▂▄▄▅▁▄▅▃▃▃▄▃▂▃▃▃▃▂▁▃▂▂▂▃▃▂▁▂▂▃▂▃▃▂▂▅▂
wandb:        val_df_aup ▅▅█▂▄▄▄▁▃▄▂▃▃▃▂▁▂▂▂▃▂▁▃▂▂▂▂▂▂▁▂▁▂▂▂▂▂▂▃▁
wandb: val_df_logit_mean ▁▁▁▁▂▅▃▄▄▄▅▅▅▆▅▅▆▆▆▇▆▆▅▅▆▅▆▆▆█▆▆▇▆▆▇▆▆▆▆
wandb:  val_df_logit_std ▁▁▁▁▃▆▃▄▄▄▆█▅▅▅▅▇▆▆█▅▆▅▄▅▅▇▆▆█▅▅▆▆▆▇▆▆▇▇
wandb:        val_dt_auc ▁▁▆▁▅▅▇▃▆▇▄▆▅█▆▅▇▆▇█▆▃▇▅▆▅▆▆▆▆▅▆▇▅▅▆▄▆█▄
wandb:        val_dt_aup ▁▁▅▁▅▆▆▃▆▇▅▆▅▇▆▅▇▆▇█▆▄▇▅▆▅▆▆▆▆▅▆▇▅▆▇▅▆█▄
wandb:          val_loss ██▇▇▅▄▅▅▄▃▃▂▄▂▃▃▂▃▂▁▃▃▂▄▃▃▂▃▃▂▃▃▁▃▂▂▂▂▁▃
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.00611
wandb:            loss_r 1.4186
wandb:        train_loss 0.71235
wandb:      train_loss_l 0.00611
wandb:      train_loss_r 1.4186
wandb:        train_time 0.91547
wandb:        val_df_auc 0.69594
wandb:        val_df_aup 0.80858
wandb: val_df_logit_mean 0.51479
wandb:  val_df_logit_std 0.02081
wandb:        val_dt_auc 0.77429
wandb:        val_dt_aup 0.80288
wandb:          val_loss 0.69296
```

### Deleting 200 edges (Union set) in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 200 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 13 > gcn_gnndelete_in_union200_Cora_13.txt &
```

Deleting the following edges (index):

[ 0, 1, 2, 4, 6, 7, 8, 9, 14, 16, 17, 18, 19, 21,
22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
37, 38, 39, 40, 41, 43, 44, 45, 47, 48, 49, 55, 58, 63,
64, 65, 66, 67, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80,
81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
96, 97, 99, 100, 102, 104, 105, 107, 108, 109, 110, 111, 112, 114,
117, 119, 120, 121, 124, 126, 128, 129, 131, 133, 134, 136, 137, 140,
141, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155,
157, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
172, 174, 175, 176, 177, 178, 179, 181, 184, 185, 186, 188, 189, 190,
193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 204, 206, 209, 211,
212, 214, 217, 219, 220, 221, 222, 223, 224, 225, 228, 230, 231, 233,
234, 235, 237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 248, 249,
250, 252, 253, 255, 256, 258, 259, 260, 261, 262, 263, 264, 267, 268,
269, 271, 272, 273]

Results:

```python
{
'test_loss': 0.704846978187561,
'test_dt_auc': 0.740459079996193,
'test_dt_aup': 0.7462982990906106,
'test_df_auc': 0.735485975,
'test_df_aup': 0.8189065671628344,
'test_df_logit_mean': 0.501010715663433,
'test_df_logit_std': 0.0016710351367735272
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▇▆▅▃▄▄▃▃▃▃▂▃▂▂▂▃▂▂▂▁▂▁▂▂▂▂▁▃▂▂▂▂▂▁▂▁▁▂▁
wandb:            loss_r ▁▂▄▃▄▂▄▃▆▃▄▃▅▃▃▄▂█▁▄▂▃▅▅▆▅▅▇▄▂▃▃▃▂▆▅▅▂██
wandb:        train_loss ▆▃▃▂▅▁▄▄▄▆▄▅▁▄▆▃▇█▄▅▅▆▂▄▃▃▅▅▅▃▅▅▅▆▄▅▅▇▄▃
wandb:      train_loss_l █▇▅▅▃▄▄▂▃▂▃▂▂▂▂▂▂▂▂▂▁▂▁▂▂▂▁▁▂▂▂▂▂▂▁▂▁▁▁▁
wandb:      train_loss_r ▃▅▅▃▆▄▃▃█▃▆▃▅▄▃▅█▆▁▃▂▄▅▄▅▄▃▃▃▃▂▅▂▂▅▄▄▄▃▄
wandb:        train_time ▆▇▄▆██▄▅▇▇▆▇▇▁▆▇▅▁▇▇▆▁▇▇▇▇▁▇▇▇▇▇▂▆▆▇▇▇▅▇
wandb:        val_df_auc █▁▄▂▃▃▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▂▂▂▁▁▂▁▂▂▂▂▂▂▂▂▂▃▂▁
wandb:        val_df_aup █▂▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▂▁▂▁▁▂▁▁▁▁▁▂▁▁▂▁▂▁▁
wandb: val_df_logit_mean ▁▂▂▃▅▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇▇█▇▇▇▇▇▇██████
wandb:  val_df_logit_std ▁▂▃▄▆▅▅▆▅▆▆▆▆▇▆▆▇▇▇▇▇▇█▇▇▇▇█▇▇█▇▇▇██████
wandb:        val_dt_auc ▂▁▆▃▇▄▄▅▅▆▆▄▅▅▆▆▆▆▅▅▄█▆▅▅▅▇▅▅▆▆▅▇▅▇▇▆█▅▄
wandb:        val_dt_aup ▁▁▅▄▇▅▅▆▆▆▆▅▆▆▆▆▇▇▆▆▆▇▇▆▆▆▇▇▆▇▇▆▇▆█▇▇█▆▆
wandb:          val_loss █▇▆▅▃▄▄▃▃▃▃▃▂▂▂▂▂▂▂▂▁▂▁▂▂▂▁▁▂▂▂▂▂▂▁▁▁▁▁▂
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.03355
wandb:            loss_r 0.81419
wandb:        train_loss 0.42387
wandb:      train_loss_l 0.03355
wandb:      train_loss_r 0.81419
wandb:        train_time 1.68571
wandb:        val_df_auc 0.61907
wandb:        val_df_aup 0.75223
wandb: val_df_logit_mean 0.55292
wandb:  val_df_logit_std 0.06178
wandb:        val_dt_auc 0.76154
wandb:        val_dt_aup 0.79262
wandb:          val_loss 0.69809
```
