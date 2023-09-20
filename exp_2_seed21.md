## Experiment 2:

Random seed: 21

Hyperpatameters:

```
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

```
nohup python train_gnn.py --dataset Cora --gnn gcn --epochs 2000 --random_seed 21 > gcn_original_Cora_21.txt &
```

Original Model results:

```python
test Logs:
{
    'test_loss': 0.6549398303031921,
    'test_dt_auc': 0.9715748235174074,
    'test_dt_aup': 0.9735161831785036
}
```

### Deleting 100 edges in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 21 > gcn_gnndelete_in_0_100_Cora_21.txt &
```

Deleting the following edges (index):

[ 0, 2, 4, 7, 8, 10, 11, 14, 16, 17, 19, 20, 21, 22,
24, 25, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40,
41, 42, 43, 44, 45, 48, 50, 51, 52, 54, 55, 56, 60, 61,
62, 63, 64, 65, 66, 67, 69, 70, 72, 74, 75, 76, 77, 78,
79, 80, 81, 82, 83, 84, 85, 87, 88, 90, 91, 93, 94, 95,
97, 98, 100, 102, 103, 105, 107, 108, 109, 110, 111, 112, 113, 114,
115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 130, 131, 132,
134, 135]

Results:

```python
test Logs:
{
    'test_loss': 0.6956506967544556,
    'test_dt_auc': 0.7914183260252041,
    'test_dt_aup': 0.804312828232771,
    'test_df_auc': 0.7498881000000001,
    'test_df_aup': 0.831106271481129,
    'test_df_logit_mean': 0.5039809003472329,
    'test_df_logit_std': 0.0060063416058627975
}
```

```
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▆▅▅▅▅▄▄▃▃▄▃▂▂▂▃▃▃▂▂▁▂▂▂▂▂▂▂▂▂▁▁▂▂▁▂▂▁▂▁
wandb:            loss_r ▄█▁▂▄▂▃▃▅▁▄▅▃▂▁▇▂▄▄▄▄▃▂▄▄▃▁█▅▁▃▅▅▃▂▄▃▆▄▆
wandb:        train_loss ▃▂▃▅▄▁▄▂▄▃▄▂▃▇▅▅▂▄▅▅▅▃▄▃█▇▆▅▅▃▃▅▃▃▆▄▄▄▅▅
wandb:      train_loss_l █▆▆▅▄▅▄▄▃▃▃▃▃▂▂▃▃▃▃▂▁▂▂▂▃▂▂▂▂▂▁▁▂▂▂▂▂▁▂▁
wandb:      train_loss_r ▃▄▂▂▂▂▃▃▅▂▃▄▃▃▁▂▅▃▅█▃▅▂▁▁▃▂▃▅▂▄▂█▃▂▁▃▁▄▁
wandb:        train_time ▅▁▅▇▄▄▁▅▃▆▄▄▆▅▇█▅▃▅▁▇▆▇▆▆▃▅▅▁▇▅▅▄▆▆▇▁▅▇▁
wandb:        val_df_auc █▂▅▃▅▄▄▃▂▂▄▅▅▅▄▄▄▃▃▄▅▂▂▃▂▃▂▄▃▅▄▅▂▃▄▁▂▃▃▄
wandb:        val_df_aup █▄▆▅▅▅▅▅▄▅▅▅▅▅▄▄▃▃▄▄▄▃▂▃▂▃▂▄▃▃▄▃▁▃▂▁▁▂▃▃
wandb: val_df_logit_mean ▁▃▂▃▄▄▄▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇▇▇▇▇█▇▇▇███▇████▇█
wandb:  val_df_logit_std ▁▃▃▄▄▄▅▆▆▆▆▆▆▆▆▆▇▇▇▇▇████▇█▇▇▇███▇████▇█
wandb:        val_dt_auc ▁▃▄▄▆▅▅▅▅▆▆▆▇▇▆▆▆▆▅▆▇▆▅▇▅▆▅▇▅███▅▆▇▅▆▇▅▇
wandb:        val_dt_aup ▁▃▄▄▆▅▅▆▆▆▆▇▇▇▆▆▆▆▆▇▇▆▆▇▆▆▆▇▆███▆▆▇▆▇█▆▇
wandb:          val_loss █▆▆▆▄▅▄▃▃▃▃▃▂▂▂▃▃▂▂▂▁▂▂▂▂▂▂▂▂▂▁▁▂▂▁▂▁▁▂▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.03113
wandb:            loss_r 0.66036
wandb:        train_loss 0.34575
wandb:      train_loss_l 0.03113
wandb:      train_loss_r 0.66036
wandb:        train_time 1.45994
wandb:        val_df_auc 0.68415
wandb:        val_df_aup 0.78101
wandb: val_df_logit_mean 0.57558
wandb:  val_df_logit_std 0.0947
wandb:        val_dt_auc 0.80935
wandb:        val_dt_aup 0.83417
wandb:          val_loss 0.68969
```

### Deleting next 100 edges in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --seqlearn True --epochs 2000 --random_seed 21 > gcn_gnndelete_in_100_200_Cora_21.txt &
```

Deleting the following edges (index):

Results:

```python
{
    'test_loss': 0.6938236355781555,
    'test_dt_auc': 0.8082479077328927,
    'test_dt_aup': 0.8175177915419025,
    'test_df_auc': 0.7528768,
    'test_df_aup': 0.8258517513556057,
    'test_df_logit_mean': 0.5058314055204391,
    'test_df_logit_std': 0.01110093988680161
}
```

```bash
wandb: Run history:

```

### Deleting 200 edges (Union set) in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 200 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 21 > gcn_gnndelete_in_union200_Cora_21.txt &
```

Deleting the following edges (index):

Results:

```python
{
'test_loss': 0.7038817405700684,
'test_dt_auc': 0.7684586078046265,
'test_dt_aup': 0.7680258025431205,
'test_df_auc': 0.7271963499999999,
'test_df_aup': 0.7976965124950552,
'test_df_logit_mean': 0.502712013721466,
'test_df_logit_std': 0.003979550162314575
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▇▅▅▄▃▃▂▃▂▂▂▂▂▂▂▂▂▂▂▁▁▂▂▁▂▂▂▂▂▂▂▁▂▁▂▁▁▁▂
wandb:            loss_r ▃▃▄▃▄▅▃▃█▆▆▂▄▅▁▂▃▄▃▃▂▅▄▂▅▄▁▂▅▄▅▄▁▄▃▄▄█▁▂
wandb:        train_loss ▂▂▅▁▃▃▂▃▁▄▂▄▆▅▆▄▇▂▄▄▄▄▆▃▃▂█▄▂▆▄▅▂▁▄▃▅█▅▂
wandb:      train_loss_l █▇▅▅▄▃▃▂▃▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▁▂▂▂▂▂▂▂▁▂▁▂▁▁▁▂
wandb:      train_loss_r ▆▄▅▄▄▅▃▅▅▅▅▃▆▄▂▇▅▅▄▃▁█▅▃▆▄▆▇▆▅▅▅▅▄▆▅▅█▃▆
wandb:        train_time ▁▁▁▇██▆█▇▇▇█▆▆▅▇▆▇▇█▆█▇▇▇█▆▆▆▇▇▆▆▇▇▇▇▆▆▇
wandb:        val_df_auc █▃▂▂▁▁▂▁▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▁▂▁▁▁▂▁▁▂▁▁▁▂▂
wandb:        val_df_aup █▂▁▂▂▂▃▂▂▂▃▃▃▃▂▃▃▃▂▂▂▂▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂
wandb: val_df_logit_mean ▁▂▃▃▅▆▅▇▆▆▇▆▇▇▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇███▇
wandb:  val_df_logit_std ▁▂▄▄▅▆▆▇▇▆▇▇▇▇▇▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇███▇
wandb:        val_dt_auc ▃▁▃▃▃▅▇▅▅▅█▅▅▇▅▆▄▇▅▆▅▄▆▅▄▅▄▇▅▄▄▅▅▆▅▅▆▄▇▅
wandb:        val_dt_aup ▁▁▄▄▅▆▇▇▆▆█▇▇█▇▇▆▇▇▇▇▇▇▇▆▇▆▇▇▇▆▇▇▇▇▇▇▇█▇
wandb:          val_loss █▇▅▅▄▃▃▂▂▂▂▂▂▁▂▂▂▂▂▁▁▁▂▂▁▂▂▂▂▁▂▂▁▂▁▁▁▁▁▂
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.03198
wandb:            loss_r 0.81131
wandb:        train_loss 0.42164
wandb:      train_loss_l 0.03198
wandb:      train_loss_r 0.81131
wandb:        train_time 3.23829
wandb:        val_df_auc 0.63461
wandb:        val_df_aup 0.76506
wandb: val_df_logit_mean 0.55521
wandb:  val_df_logit_std 0.05707
wandb:        val_dt_auc 0.7747
wandb:        val_dt_aup 0.80175
wandb:          val_loss 0.69766
```
