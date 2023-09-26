## Experiment 1:

Random seed: 21

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
nohup python train_gnn.py --dataset Cora --gnn gcn --epochs 2000 --random_seed 21 > gcn_original_Cora_21.txt &
```

Original Model results:

```python
test Logs:
{
'test_loss': 0.654935896396637,
'test_dt_auc': 0.9715753207705315,
'test_dt_aup': 0.9735260885296284
}
```

### Deleting 100 edges outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 21 > gcn_gnndelete_out_0_100_Cora_21.txt &
```

Deleting the following edges (index):

[ 1, 3, 5, 6, 9, 12, 13, 15, 18, 23, 26, 32, 33, 46,
47, 49, 53, 57, 58, 59, 68, 71, 73, 86, 89, 92, 96, 99,
101, 104, 106, 121, 127, 128, 129, 133, 136, 137, 141, 142, 149, 151,
153, 161, 163, 170, 172, 174, 175, 183, 184, 190, 191, 192, 194, 197,
201, 202, 206, 207, 208, 218, 219, 222, 224, 225, 230, 232, 234, 236,
239, 240, 241, 245, 253, 254, 257, 258, 261, 266, 270, 276, 277, 278,
279, 282, 285, 287, 288, 294, 298, 299, 304, 306, 308, 314, 315, 317,
322, 323]

Results:

```python
test Logs:
{
'test_loss': 0.6813064813613892,
'test_dt_auc': 0.8608375473049329,
'test_dt_aup': 0.8783080095191004,
'test_df_auc': 0.8290993000000001,
'test_df_aup': 0.8831705916868116,
'test_df_logit_mean': 0.5257729589939117,
'test_df_logit_std': 0.0517350192777478
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▆▅▅▄▃▃▂▂▂▂▂▂▂▂▁▂▂▁▂▂▂▂▁▁▁▂▂▂▂▁▁▁▂▁▂▂▁▂▁
wandb:            loss_r ▃▇▂▁▄▁▄▃▄▁▃█▃▂▁▆▃▃▂▅▂▃▃▃▄▃▂█▄▂▂▄▅▁▂▃▄▂▅▃
wandb:        train_loss ▃▃▄▃▄▁▄▂▄▅▄▄▂▄▅▂▄▅▂▃▄▃▄▂█▂▃▄▁▂▄▃▅▂▇▂▅▃█▅
wandb:      train_loss_l █▆▅▅▄▃▃▃▃▂▂▂▂▂▂▁▂▂▁▂▂▂▂▁▁▁▂▂▂▂▁▁▁▂▁▁▂▁▂▁
wandb:      train_loss_r ▃▃▂▁▃▂▄▄▅▄▂█▃▂▁▁▁▃▅▅▂▄▃▁▁▃▃▄▄▃▃▂▇▁▁▁▄▂▁▃
wandb:        train_time ▆█▇▇▆▆▅▆▇▇▆▆▆▆▆▆▆▇▆▇▅▇▆▆▅▆▆▅▇▇▆▇▆▅▆▇▆▆▅▁
wandb:        val_df_auc ▅▅█▄▅▆▅▅▃▂▃▂▄▅▅▃▄▅▃▂▄▄▃▄▄▂▄▄▄▅▄▃▃▃▁▂▅▄▄▄
wandb:        val_df_aup ▇▅█▅▅▆▄▅▃▂▃▂▄▅▄▃▃▄▃▂▃▃▂▃▄▂▄▄▄▄▃▂▂▃▁▂▄▃▃▃
wandb: val_df_logit_mean ▁▂▂▂▃▄▄▅▅▅▅▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█▇▇█▇▇▇█
wandb:  val_df_logit_std ▁▂▂▂▃▄▅▅▅▅▅▆▆▆▆▇▇███▇▇▇█▇▇▇▇▇▇▇▇█▇▇█████
wandb:        val_dt_auc ▁▅▇▅▇▇▆▇▆▅▅▆▆█▇▇▇█▇▅▇▆▆▇▇▅▇█▇▇▇▆▆▆▅▆█▇▇▇
wandb:        val_dt_aup ▁▅▆▅▇▇▇▇▇▆▆▆▇██▇▇█▇▆▇▇▇▇▇▆▇█▇▇█▇▇▇▆▇██▇█
wandb:          val_loss █▅▄▄▃▂▂▂▂▂▂▂▂▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.01951
wandb:            loss_r 1.14318
wandb:        train_loss 0.58135
wandb:      train_loss_l 0.01951
wandb:      train_loss_r 1.14318
wandb:        train_time 0.93866
wandb:        val_df_auc 0.78917
wandb:        val_df_aup 0.86225
wandb: val_df_logit_mean 0.5365
wandb:  val_df_logit_std 0.07404
wandb:        val_dt_auc 0.83271
wandb:        val_dt_aup 0.85787
wandb:          val_loss 0.68362
```

### Deleting next 100 edges outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --seqlearn True --epochs 2000 --random_seed 21 > gcn_gnndelete_out_100_200_Cora_21.txt &
```

Deleting the following edges (index):

[324, 327, 330, 334, 336, 337, 341, 344, 345, 346, 347, 349, 350, 357,
360, 367, 376, 379, 383, 384, 386, 387, 393, 396, 401, 406, 407, 409,
411, 413, 421, 423, 425, 432, 433, 436, 440, 442, 458, 459, 461, 464,
466, 468, 470, 479, 481, 487, 489, 491, 498, 502, 503, 505, 511, 513,
525, 527, 528, 530, 531, 535, 536, 539, 543, 544, 550, 555, 556, 559,
560, 563, 564, 573, 574, 578, 581, 589, 590, 594, 601, 609, 612, 614,
615, 624, 627, 636, 643, 645, 647, 651, 652, 655, 658, 660, 661, 663,
668, 674]

Results:

```python
{
'test_loss': 0.6813974380493164,
'test_dt_auc': 0.8611263519193622,
'test_dt_aup': 0.8781503172397213,
'test_df_auc': 0.8512157000000001,
'test_df_aup': 0.8982132417105972,
'test_df_logit_mean': 0.5128236019611359,
'test_df_logit_std': 0.026135325244436177
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l ▂▃▂▆▇▇▅█▄▆▅▄▇▆▅▃▆▅▅▅▅▄▇▄▄▂▆▆▆▇▅▂▂▇▁▆▆▅▆▃
wandb:            loss_r ▄▆▃▂▃▁▂▂▃▂▄▇▁▂▂▅▂▄▃▄▂▃▂▄▄▃▂█▃▁▂▄▄▃▂▂▄▃▅▃
wandb:        train_loss ▃▁█▄▄▃▂▁▄▆▄▅▃▆▅▃▄▅▇▄▅▄▄▃▇▇▅▄▅▃▃▇▅▃▅▄▅▃▆▆
wandb:      train_loss_l ▂▂▃▆██▅▇▄▇▆▄█▅▅▅▇▄▅▇▆▄▇▄▃▂▇▅▆▇▅▂▄▇▁▂▆▆▅▄
wandb:      train_loss_r ▅▃▃▂▃▂▁▅▃▄▂█▃▂▂▁▃▄▄▃▃▄▂▁▁▂▃▂▃▂▂▁▆▃▁▁▄▁▄▃
wandb:        train_time ▆▄▄▆▄▄▃▄▃▄▄▃▃▄▃▄▆▄▄▆▃▇▅▄▄▆▄▅▁▆▄▇▅▂█▆▅▅▄▅
wandb:        val_df_auc ▃▄▄▃█▆▃▅▃▃▂▁▄▂▃▆▄▅▇▅▄▅▃▄▃▃▆▆▃▁▆▂▆▁▅▃▂▃▅▄
wandb:        val_df_aup ▃▄▄▃█▆▃▄▃▃▃▁▄▂▃▆▄▅▇▅▄▄▃▄▂▃▆▆▃▁▇▂▆▂▅▃▂▂▅▄
wandb: val_df_logit_mean █▇▅▄▃▃▅▄▄▄▂▄▃▃▄▃▃▃▃▂▂▃▂▂▃▄▂▁▃▂▃▅▄▃▅▇▄▄▃▅
wandb:  val_df_logit_std █▆▅▄▄▃▄▄▄▄▃▃▂▂▃▃▃▃▃▃▄▃▃▃▃▃▂▁▃▂▂▄▄▃▄▅▃▄▃▄
wandb:        val_dt_auc ▃▄▄▂█▅▄▆▃▄▁▃▆▃▄▆▅▅▇▅▄▅▃▅▄▄▆▇▃▁▇▅▆▁▆▃▃▄▆▇
wandb:        val_dt_aup ▄▅▄▃█▅▄▅▃▄▂▃▅▃▄▇▄▅▇▅▄▅▃▆▄▄▆▇▃▁▇▅▆▁▆▄▃▄▆▇
wandb:          val_loss ▁▃▄▆▆▇▆▇▆▇█▆▇▆▆▅▇▅▅▇▆▅▇▆▅▄▇▆▇▇▅▄▅█▄▅▇▇▅▅
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.01226
wandb:            loss_r 1.20827
wandb:        train_loss 0.61026
wandb:      train_loss_l 0.01226
wandb:      train_loss_r 1.20827
wandb:        train_time 3.43995
wandb:        val_df_auc 0.80091
wandb:        val_df_aup 0.87638
wandb: val_df_logit_mean 0.51396
wandb:  val_df_logit_std 0.02579
wandb:        val_dt_auc 0.84503
wandb:        val_dt_aup 0.86572
wandb:          val_loss 0.68304
```

### Deleting 200 edges (Union set) outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 200 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 21 > gcn_gnndelete_out_union200_Cora_21.txt &
```

Results:

```python
{
'test_loss': 0.6981834173202515,
'test_dt_auc': 0.7931391201861795,
'test_dt_aup': 0.8002787945141422,
'test_df_auc': 0.8235407250000001,
'test_df_aup': 0.8774327034197177,
'test_df_logit_mean': 0.5007650396227836,
'test_df_logit_std': 0.0025279303384931427
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▆▅▄▃▃▃▂▂▂▂▂▁▂▂▂▁▁▁▂▁▂▂▂▂▂▂▂▁▁▂▂▁▂▁▂▁▂▂▂
wandb:            loss_r ▇▃▃▅▄▅▅▃▇▄▅▂▃▃▄▃▅▄▆▅▃▅▄▃▅▂▂▅█▅▅▄▄▃▄▆▆█▄▁
wandb:        train_loss ▁▃▄▂▄▂▄▂▃▄▄▃█▆▄▆▄▃▅▄▆▃▄▄▃▂▇▃▃▅▃▅▃▂▄▃▆▇▆▃
wandb:      train_loss_l █▆▅▄▃▃▃▂▂▂▂▂▁▂▂▂▁▁▂▂▁▂▁▁▂▂▁▂▁▁▂▂▁▁▁▂▁▂▂▂
wandb:      train_loss_r ▄▄▃▅▄▄▄▇▄▅▇▁▅▃▄▄▆▄▅▄▃▆▃▄▅▁▆▆█▄▅▃▃▃▆▄▅▅▂▆
wandb:        train_time ▄▅█▄▂▃▄▇▄▄▄▅▃▃▆▁▄▄▆▁▆▆▄▅▆▃▁▄▃▇▆▂▂▄▇▂▂▁▄▃
wandb:        val_df_auc ▃█▅▃▃▄▃▂▂▄▃▃▃▂▄▂▃▂▃▃▃▃▃▂▅▃▂▅▂▂▃▂▂▁▂▂▁▂▂▃
wandb:        val_df_aup ▁█▅▃▃▄▄▃▃▄▄▃▃▃▄▃▃▃▃▃▃▄▄▃▄▃▃▅▃▃▄▃▃▂▃▃▂▂▂▃
wandb: val_df_logit_mean ▁▁▃▄▅▅▄▅▆▆▆▇▆▇▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇███▇█▇▇▇
wandb:  val_df_logit_std ▁▁▃▄▅▅▄▅▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█▇▇███▇███▇
wandb:        val_dt_auc ▁▅▆▇▆▆▆▆▆█▆▆▆▇▆▆▇▆▇▆▆▇▆▅█▇▇█▇▆▇▆▇▅▇▆▆▅▆▇
wandb:        val_dt_aup ▁▅▆▇▇▇▆▇▇█▇▇▇█▇▇█▇▇▇▇█▇▇██▇█▇▇▇▇█▇█▇▇▆▇▇
wandb:          val_loss █▆▄▃▂▂▂▂▂▁▂▂▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.02452
wandb:            loss_r 0.96365
wandb:        train_loss 0.49408
wandb:      train_loss_l 0.02452
wandb:      train_loss_r 0.96365
wandb:        train_time 3.46005
wandb:        val_df_auc 0.70311
wandb:        val_df_aup 0.81975
wandb: val_df_logit_mean 0.52377
wandb:  val_df_logit_std 0.04387
wandb:        val_dt_auc 0.79159
wandb:        val_dt_aup 0.82496
wandb:          val_loss 0.69117
```
