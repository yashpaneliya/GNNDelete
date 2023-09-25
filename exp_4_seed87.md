## Experiment 4:

Random seed: 87

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
nohup python train_gnn.py --dataset Cora --gnn gcn --epochs 2000 --random_seed 87 > gcn_original_Cora_87.txt &
```

Original Model results:

```python
Test Logs:
{
    'test_loss': 0.6560620069503784,
    'test_dt_auc': 0.9684825057897667,
    'test_dt_aup': 0.968274443237336
}
```

### Deleting 100 edges in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 87 > gcn_gnndelete_in_0_100_Cora_87.txt &
```

Deleting the following edges (index):

[ 0, 1, 2, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16,
21, 23, 24, 25, 26, 27, 28, 30, 32, 34, 35, 37, 38, 39,
41, 42, 43, 44, 45, 46, 48, 49, 51, 52, 56, 58, 59, 60,
61, 63, 64, 66, 67, 68, 70, 71, 74, 75, 76, 77, 78, 80,
81, 82, 83, 85, 87, 89, 90, 92, 93, 94, 95, 96, 97, 99,
100, 101, 103, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117,
118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 133,
134, 136]

Results:

```python
Test Logs:
{
    'test_loss': 0.6960921883583069,
    'test_dt_auc': 0.7740877617950679,
    'test_dt_aup': 0.789203719307774,
    'test_df_auc': 0.7646564,
    'test_df_aup': 0.8439009194529653,
    'test_df_logit_mean': 0.5019703230261803,
    'test_df_logit_std': 0.003005810757230079
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▇▅▄▄▃▃▂▂▂▂▂▂▂▂▂▂▁▂▂▁▂▂▂▂▃▂▂▂▁▁▂▂▂▁▁▁▂▁▁
wandb:            loss_r ▂▂▂▄▄▁▂▄▃▄▂▄▂▃▁▅▁▂▃▂█▃▂▂▂▃▄▂▃▃▂▃▂▄▃▂▃▂▂▂
wandb:        train_loss ▂▂▁▂▁▃▁▂▂▂▂▂▁█▃▂▂▂▃▁▁▃▂▁▄▂▂▂▂▂▂▁▁▂▂▁▁▁▂▂
wandb:      train_loss_l █▇▅▄▃▃▃▂▂▂▂▂▂▂▂▂▂▁▂▂▁▂▁▂▂▃▂▂▂▁▁▁▂▁▁▁▁▂▁▂
wandb:      train_loss_r ▂▂▂▃▄▃▁▂▁█▂▃▂▂▁▂▃▂▅▂▂▂▂▃▃▂▁▃▂▂▂▃▁▃▂▂▂▃▅▂
wandb:        train_time ▆▆▇▆▇▇▆▆▇▅██▇▇▇▇▆▆▅▆▆▅▇▇▆▇▆▇▆▆▇▆▆▇█▇█▆▆▁
wandb:        val_df_auc █▄▃▂▁▂▁▁▁▁▂▂▁▂▂▂▁▂▁▁▁▂▁▁▂▂▂▂▁▁▂▂▁▂▁▁▁▁▁▂
wandb:        val_df_aup █▃▂▂▁▂▂▂▁▁▂▂▁▂▂▂▂▂▁▁▁▂▂▁▂▂▂▂▁▁▂▂▂▂▁▁▂▂▂▂
wandb: val_df_logit_mean ▁▂▃▄▅▆▆▆▆▆▇▇▇▆▆▆▇▇▇▇▇▇▇▇▇▆▇▇▇█▇▇▇▇███▇█▇
wandb:  val_df_logit_std ▁▂▃▄▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█▇▇▇▇██▇▇▇▇
wandb:        val_dt_auc ▁▃▅▅▄▆▅▆▇▅▆▆▅▆▇▆▆▇▅▆▆█▆▇▇▇▇▇▆▇▇▆▆▇▇▆▅▆▅▇
wandb:        val_dt_aup ▁▃▅▆▅▆▆▇▇▆▇▇▆▆▇▇▇█▆▆▇█▇▇█▇▇▇▇▇█▇▇█▇▇▆▇▆▇
wandb:          val_loss █▇▅▄▃▃▃▂▂▂▂▂▂▂▂▂▂▁▂▂▁▁▁▂▁▂▂▂▂▁▁▁▂▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.03118
wandb:            loss_r 0.89734
wandb:        train_loss 0.46426
wandb:      train_loss_l 0.03118
wandb:      train_loss_r 0.89734
wandb:        train_time 0.94598
wandb:        val_df_auc 0.68603
wandb:        val_df_aup 0.80451
wandb: val_df_logit_mean 0.5737
wandb:  val_df_logit_std 0.07757
wandb:        val_dt_auc 0.82723
wandb:        val_dt_aup 0.84683
wandb:          val_loss 0.68739
```

### Deleting next 100 edges in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --seqlearn True --epochs 2000 --random_seed 87 > gcn_gnndelete_in_100_200_Cora_87.txt &
```

Deleting the following edges (index):

Results:

```python
{
'test_loss': 0.694471001625061,
'test_dt_auc': 0.8175241150361289,
'test_dt_aup': 0.8190004533764211,
'test_df_auc': 0.790153,
'test_df_aup': 0.8437217493045343,
'test_df_logit_mean': 0.5083927899599076,
'test_df_logit_std': 0.016234442030553835
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▇▆▇▇▅▅▃▅▄▄▅▄▄▄▄▂▁▂▃▄▄▂▃▄▃▃▄▃▃▃▄▄▃▂▄▂▂▂▃
wandb:            loss_r ▃▂▂▅▄▂▁▆▆▂▁▇▄▄▂▆▅▃▆▃▆▃▃▃▄▄▃▂█▄▃▄▄▆▃▁▄▂▂▂
wandb:        train_loss ▃▃▃▄▃▆▁▃▄▂▄▄▃▅▇▂▃▅▅▃▃█▃▂█▄▁▃▂▅▄▂▂▄▄▂▂▃▂▁
wandb:      train_loss_l █▇▄▇▆▄▅▃▅▄▄▅▄▄▄▄▁▁▂▃▄▄▃▃▃▃▄▄▃▄▃▄▄▃▂▄▂▂▂▃
wandb:      train_loss_r ▃▃▃▅█▄▁▃▂▄▂▆▄▅▁▃▃▃█▂▄▁▃▅▅▄▅▄▇▁▃▄▂▅▃▃▃▆▄▃
wandb:        train_time ▁▅▆▆▇▆▆▆▆▇▇▇▇▇▆▆█▆▅▆▇▇▇▆▆▅▆▆▆▆▇▆▇▆▆▆▇▄▆▆
wandb:        val_df_auc ▇▇▃▆▅▄▂▃▄▆▃█▁▆▅▂▃▃▄▇▄▆▂▃▅▄▆▃▃▄▆▂▅▃▁▄▃▂▁▃
wandb:        val_df_aup ▇█▃▅▆▄▃▃▄▅▄▇▁▅▅▂▃▃▄▆▃▅▂▃▄▂▅▂▄▄▅▂▄▃▂▄▃▂▁▂
wandb: val_df_logit_mean ▁▁▄▃▃▅▅▅▃▄▄▄▄▄▅▅▇▇█▅▅▅▆▆▇▇▅▅▇▆▇▅▅▆▇▆▇▇▆▇
wandb:  val_df_logit_std ▁▂▄▂▂▄▄▄▃▄▄▄▅▅▄▅▇▇█▅▅▆▆▅▆▆▅▅▆▅▆▅▅▆▆▅▆▆▆▇
wandb:        val_dt_auc ▂▄▃▇▅▅▃▄▄▇▄▇▁▇▆▃▆▅▆█▄█▄▅█▇▇▃▅▅█▃▇▅▂▄▅▄▂▅
wandb:        val_dt_aup ▁▃▃▆▅▅▃▄▄▆▄▇▁▇▅▃▇▆▇█▄▇▄▅█▆▆▃▆▅█▃▇▅▃▅▅▅▃▅
wandb:          val_loss █▇▅▆▆▄▄▄▅▅▅▅▆▄▄▅▂▂▁▄▅▄▄▄▃▃▄▅▃▄▃▄▄▃▃▄▃▃▄▃
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.00618
wandb:            loss_r 0.84242
wandb:        train_loss 0.4243
wandb:      train_loss_l 0.00618
wandb:      train_loss_r 0.84242
wandb:        train_time 1.59396
wandb:        val_df_auc 0.69909
wandb:        val_df_aup 0.80706
wandb: val_df_logit_mean 0.51253
wandb:  val_df_logit_std 0.02325
wandb:        val_dt_auc 0.79029
wandb:        val_dt_aup 0.80969
wandb:          val_loss 0.69274
```

### Deleting 200 edges (Union set) in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 200 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 87 > gcn_gnndelete_in_union200_Cora_87.txt &
```

Deleting the following edges (index):

[ 0, 1, 2, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16,
21, 23, 24, 25, 26, 27, 28, 30, 32, 34, 35, 37, 38, 39,
41, 42, 43, 44, 45, 46, 48, 49, 51, 52, 56, 58, 59, 60,
61, 63, 64, 66, 67, 68, 70, 71, 74, 75, 76, 77, 78, 80,
81, 82, 83, 85, 87, 89, 90, 92, 93, 94, 95, 96, 97, 99,
100, 101, 103, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117,
118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 133,
134, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 149, 150,
151, 152, 153, 155, 157, 159, 161, 162, 163, 165, 166, 167, 168, 169,
170, 171, 172, 173, 174, 177, 178, 179, 182, 183, 184, 185, 186, 188,
189, 190, 191, 192, 195, 196, 198, 199, 202, 203, 204, 205, 206, 207,
209, 210, 211, 212, 213, 215, 217, 219, 220, 222, 223, 228, 229, 230,
231, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
247, 248, 249, 250, 251, 253, 254, 255, 256, 258, 260, 261, 262, 263,
264, 265, 266, 268]

Results:

```python
{
'test_loss': 0.6994182467460632,
'test_dt_auc': 0.79837136673303,
'test_dt_aup': 0.8042850225733496,
'test_df_auc': 0.6487311250000001,
'test_df_aup': 0.7647121463926069,
'test_df_logit_mean': 0.5428835360705853,
'test_df_logit_std': 0.05246788146513062
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▇▅▅▄▄▃▂▃▂▂▂▃▃▂▂▂▂▂▃▁▂▂▁▂▂▂▂▂▃▁▂▂▁▁▂▁▁▁▂
wandb:            loss_r ▃▂▅▃▄▃▂▄▂▃▅▄▃▃▄▅▃▅▂▂█▃▁▂▄█▂▂▂▃▃▃▁▅▃▃▆▃▁▃
wandb:        train_loss ▄▃▃▄▅▇▂▅▁▄▄▂█▅▅▅▇▄▂▂▃▇▃▁▃▄▆▄▄▃▄▃▃▃▄▃▂▃▆▅
wandb:      train_loss_l █▇▅▅▄▄▃▂▃▂▂▂▃▃▂▁▂▂▂▃▁▃▂▁▂▂▂▂▂▂▁▂▁▁▁▂▁▂▁▁
wandb:      train_loss_r ▄▃▄▃▁▅▂▄▂▇▅▄▂▁▄▂▁▅▃▃▂▂▁▂▂█▂▄▂▄▂▃▃▅▃▅▆▅▃▅
wandb:        train_time █▇█▇▅▇█▇▇▇▇▇▅▇▇▇▇▇▆▇▇▇▇▆▅▅▇▇▇▇▇▅▇▇▇▇▇▇▇▁
wandb:        val_df_auc █▄▂▃▃▂▂▁▁▂▂▂▂▃▃▃▃▃▂▁▂▁▂▁▂▂▂▂▂▁▂▂▁▁▂▂▂▂▂▂
wandb:        val_df_aup █▃▂▂▂▂▁▁▁▂▂▃▂▃▃▂▃▂▂▂▂▁▂▁▂▂▂▃▂▁▂▃▂▁▂▂▁▁▂▂
wandb: val_df_logit_mean ▁▂▃▄▄▄▅▆▅▆▆▆▆▅▆▇▆▇▇▆▇▆▇▇▆▇▇▇▇▆▇▇█▇█▇█▇▇█
wandb:  val_df_logit_std ▁▂▃▄▅▄▆▆▆▆▇▆▆▅▆▇▆▇▇▆▇▇▇▇▆▇▇▇▆▆▇▇▇▇▇▇██▇█
wandb:        val_dt_auc ▁▄▆▅▆▃▅▅▆▆█▇▆▇████▇▅▆▆▆▇▆▇▆█▆▄▆▇▇▄▇▇▅▆██
wandb:        val_dt_aup ▁▃▅▅▆▅▆▆▆▇█▇▇▇██▇██▆▇▇▇█▇█▇█▇▆▇▇█▆██▇▇██
wandb:          val_loss █▇▅▅▄▄▃▃▃▂▂▂▃▃▂▂▂▂▂▃▁▂▂▁▂▂▂▂▂▂▁▂▁▂▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.03246
wandb:            loss_r 0.86307
wandb:        train_loss 0.44776
wandb:      train_loss_l 0.03246
wandb:      train_loss_r 0.86307
wandb:        train_time 1.05071
wandb:        val_df_auc 0.63587
wandb:        val_df_aup 0.76021
wandb: val_df_logit_mean 0.55085
wandb:  val_df_logit_std 0.05998
wandb:        val_dt_auc 0.78519
wandb:        val_dt_aup 0.80299
wandb:          val_loss 0.69777
```
