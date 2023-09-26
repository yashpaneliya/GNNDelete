## Experiment 1:

Random seed: 42

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
nohup python train_gnn.py --dataset Cora --gnn gcn --epochs 2000 --random_seed 42 > gcn_original_Cora_42.txt &
```

Original Model results:

```python
test Logs:
{
'test_loss': 0.6568813323974609,
'test_dt_auc': 0.9678981339184213,
'test_dt_aup': 0.9668032953465076,
}
```

### Deleting 100 edges in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 42 > gcn_gnndelete_in_0_100_Cora_42.txt &
```

Deleting the following edges (index):

[ 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15,
16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
46, 47, 49, 51, 52, 53, 54, 59, 60, 61, 62, 64, 66, 67,
68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
82, 84, 85, 88, 90, 92, 93, 94, 96, 97, 100, 102, 103, 104,
105, 107, 108, 109, 110, 111, 114, 115, 116, 117, 118, 122, 124, 126,
129, 130]

Results:

```python
test Logs:
{
    'test_loss': 0.6943832039833069,
    'test_dt_auc': 0.7937003697872582,
    'test_dt_aup': 0.8036164549581106,
    'test_df_auc': 0.8240494,
    'test_df_aup': 0.8847948367127326,
    'test_df_logit_mean': 0.5015694046020508,
    'test_df_logit_std': 0.0020286554844923914
}
```

```
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▇▆▅▅▄▃▃▂▃▃▃▂▃▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▁▁▁▂▁▁▁▁▁▂▁
wandb:            loss_r ▆▃▅▇▄▄▅▃▂▄▅▅▆▇▅▅▇▄▆▅▄▇▁▂▁▄▆▆▅▂▄▅▇▄▃▆▄█▄▇
wandb:        train_loss ▅█▃▃▅▄▄▅▅▄▂▆▄▅▅█▆▄▇▃▃▅▄▅▃▄▅▅▃▅▃▄▄▄▁▄▅▃▃▃
wandb:      train_loss_l █▇▆▅▅▄▃▃▂▃▃▃▂▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▂▁▂▁▁▁▂▁
wandb:      train_loss_r ▃▄▂▄▅▂▃█▃▁▃▃▄▂▃▃▅▃▂▂▂▂▁▃▄▃▇▄▃▄▃▄▅▃▂▆▂▃▆▃
wandb:        train_time ▅█▄▃▂▄▃▂▃▁▃▃▅▃▁▂▁▂▃▃▁▅▃▃▃▁▁▁▂▁▄▂▅▁▃▃▃▄▄▅
wandb:        val_df_auc █▁▃▃▂▂▁▂▁▂▂▂▂▂▁▂▂▁▂▂▂▂▂▂▁▂▂▁▂▁▂▂▁▂▂▂▁▂▁▁
wandb:        val_df_aup █▂▃▃▃▃▂▂▂▂▂▂▂▂▁▂▂▁▂▂▂▂▃▂▂▂▂▂▂▁▁▁▂▁▁▁▁▁▁▁
wandb: val_df_logit_mean ▁▁▂▃▃▄▅▅▆▅▅▅▆▆▆▆▆▇▆▆▆▆▇▇▇▇▇▇▇▇██▇█████▇█
wandb:  val_df_logit_std ▁▁▃▄▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇███████████
wandb:        val_dt_auc ▄▁▆▆▆▆▆▅▆▇▆▆▇▆▅▅▇▆▆▆▆▆█▆▆▇▆▅▆▆▇█▅▇▇▇▆▇▆▆
wandb:        val_dt_aup ▃▁▅▅▅▆▆▆▆▇▆▆▇▆▆▆▇▆▆▆▆▇█▆▇▇▇▆▇▇██▆█▇█▇▇▇▇
wandb:          val_loss ██▆▅▄▄▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▁▁▁▂▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.02801
wandb:            loss_r 0.96782
wandb:        train_loss 0.49791
wandb:      train_loss_l 0.02801
wandb:      train_loss_r 0.96782
wandb:        train_time 0.92924
wandb:        val_df_auc 0.67832
wandb:        val_df_aup 0.79407
wandb: val_df_logit_mean 0.61519
wandb:  val_df_logit_std 0.11026
wandb:        val_dt_auc 0.81687
wandb:        val_dt_aup 0.85077
wandb:          val_loss 0.68575
```

### Deleting next 100 edges in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --seqlearn True --epochs 2000 --random_seed 42 > gcn_gnndelete_in_100_200_Cora_42.txt &
```

Deleting the following edges (index):

[131, 132, 134, 137, 138, 139, 140, 141, 143, 145, 146, 148, 149, 151,
152, 153, 154, 156, 158, 160, 161, 162, 164, 165, 168, 171, 172, 173,
174, 175, 177, 179, 180, 181, 182, 183, 184, 185, 187, 188, 189, 191,
193, 194, 197, 199, 200, 201, 203, 205, 206, 207, 210, 212, 213, 214,
215, 216, 217, 220, 221, 222, 223, 226, 228, 229, 230, 231, 232, 233,
234, 235, 238, 239, 241, 242, 243, 244, 245, 248, 249, 251, 252, 254,
256, 257, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271,
272, 275]

Results:

```python
{
    'test_loss': 0.6945428848266602,
    'test_dt_auc': 0.800959221166355,
    'test_dt_aup': 0.8109434094032837,
    'test_df_auc': 0.7676423,
    'test_df_aup': 0.8407347223596123,
    'test_df_logit_mean': 0.5049973240494728,
    'test_df_logit_std': 0.006690676961244771
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▆▇▇▇▇▇▆▅▄▄▅▄▄▄▄▃▃▃▂▃▂▂▁▁▂▂▂▂▂▁▂▃▂▁▁▂▂▂▁
wandb:            loss_r ▆▄▇▅▅▄▆▃▂▂▃▄▆▃▅▅▆▄▆▄▄▆▁▂▃▄▆▂▄▁▄▆▇▂▃▃▄█▃▄
wandb:        train_loss ▄█▃▄▆▄▅▆▆▃▂▇▅▃▅▇▅▅▇▄▄▅▃▇▃▄▅▅▃▄▃▄▄▄▁▃▅▃▃▄
wandb:      train_loss_l █▆▇▇▇▇▆▆▅▄▅▅▄▄▄▄▃▃▃▃▃▂▂▁▁▂▂▂▂▂▁▂▃▂▂▁▂▂▂▂
wandb:      train_loss_r ▅▅▄▄▇▃▅▆▄▂▅▄▅▂▄▄▆▃▃▃▃▃▁▅▄▄█▆▄▃▄▆▇▂▂▆▃▆█▂
wandb:        train_time ▂▇▆▇▆█▇▆▇▇▇█▇▇▇▇█▆▆▇▇▁▁▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁
wandb:        val_df_auc █▅▆▇▅▃▂▅▃▃▄▄▄▆▂▄▂▄▅▁▄▁▂▂▄▃▄▂▃▁▂▂▂▂▃▃▁▃▃▂
wandb:        val_df_aup █▅▆▆▅▃▃▅▃▃▃▃▃▅▂▃▂▃▄▁▃▁▁▂▃▃▃▂▂▁▂▂▂▂▃▃▂▂▂▁
wandb: val_df_logit_mean ▁▂▂▂▂▂▂▃▄▄▄▄▆▅▅▆▆▆▆█▅▆▆█▇▆▆▆▆▆▇▆▅▇▇█▆▇▆▇
wandb:  val_df_logit_std ▁▂▁▁▁▂▂▃▃▄▄▄▆▅▅▇▇▆▅█▅▆▆▇▆▆▆▆▆▆▆▅▅▆▆▇▆▆▇▇
wandb:        val_dt_auc ▂▄▄▇▄▁▂▆▄▄▄▃▆▇▂▆▂▅▇▅▅▃▃▄▆▆▆▄▅▂▅▅▅▄▇▅▃█▆▅
wandb:        val_dt_aup ▂▄▄▆▄▁▂▆▄▄▄▃▇▇▃▆▃▆▇▅▅▄▄▅▇▇▆▅▅▃▆▅▅▅▇▆▄█▇▆
wandb:          val_loss █▆▇▇▇▇▇▅▅▅▆▆▃▄▅▄▄▃▂▂▄▃▃▂▁▁▃▃▃▃▂▃▄▃▂▂▃▂▃▃
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.00673
wandb:            loss_r 0.76738
wandb:        train_loss 0.38706
wandb:      train_loss_l 0.00673
wandb:      train_loss_r 0.76738
wandb:        train_time 0.9523
wandb:        val_df_auc 0.68664
wandb:        val_df_aup 0.79785
wandb: val_df_logit_mean 0.51628
wandb:  val_df_logit_std 0.02422
wandb:        val_dt_auc 0.78541
wandb:        val_dt_aup 0.80606
wandb:          val_loss 0.69431
```

### Deleting 200 edges (Union set) in 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 200 --df in --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 42 > gcn_gnndelete_in_0_200_Cora_42.txt &
```

Deleting the following edges (index):

[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15,
16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
46, 47, 49, 51, 52, 53, 54, 59, 60, 61, 62, 64, 66, 67,
68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
82, 84, 85, 88, 90, 92, 93, 94, 96, 97, 100, 102, 103, 104,
105, 107, 108, 109, 110, 111, 114, 115, 116, 117, 118, 122, 124, 126,
129, 130, 131, 132, 134, 137, 138, 139, 140, 141, 143, 145, 146, 148,
149, 151, 152, 153, 154, 156, 158, 160, 161, 162, 164, 165, 168, 171,
172, 173, 174, 175, 177, 179, 180, 181, 182, 183, 184, 185, 187, 188,
189, 191, 193, 194, 197, 199, 200, 201, 203, 205, 206, 207, 210, 212,
213, 214, 215, 216, 217, 220, 221, 222, 223, 226, 228, 229, 230, 231,
232, 233, 234, 235, 238, 239, 241, 242, 243, 244, 245, 248, 249, 251,
252, 254, 256, 257, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269,
270, 271, 272, 275]

Results:

```python
{
    'test_loss': 0.6956503391265869, 
    'test_dt_auc': 0.8048177562327944, 
    'test_dt_aup': 0.8223495941998156, 
    'test_df_auc': 0.654685525, 
    'test_df_aup': 0.7726058039283002, 
    'test_df_logit_mean': 0.5648048342764378, 
    'test_df_logit_std': 0.06930450216671713
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l ███▆▅▅▃▄▃▂▃▃▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▁▂▂▁▁▂▂▁▂▂▁▁▂
wandb:            loss_r ▄▄▄█▁▃▃▁▂▃▂▂▇▄▃▇▄▅▄▁▂▂▁▇▆▄▂▄▃▂▃▄▃▃▃▄▄▂▃▃
wandb:        train_loss ▇▄▃▄▃▂▃█▃▃▃▃▂▅▂▂▃▄▅▁▂▂▆▄▃▄▄▁▃▄▂▃▃▃▄▄▂▂▂▂
wandb:      train_loss_l ██▇▆▅▅▃▄▃▂▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▂▂▁▂▂▁▁▂
wandb:      train_loss_r █▄▂▆▃▃▃▁▃▂▂▂▂▂▂▃▅▄▂▃▂▃▁▂▃▃▂▂▃▃▁▂▃▃▂▃▃▂▃▂
wandb:        train_time ▁▁▇▆▇▆▆▇▅█▇▆▇▇▇▇▄▆▇▇▇▆▆▅█▇▇▆▇▇▇▇▇▇▇▇▇▆▇▁
wandb:        val_df_auc ▅▆▇▃▁█▆▂▅▆▅▆▄▅▅▃▃▅▂▄▄▅▄▆▄▅▅▃▇▂▆▂▃▅▃▆▂▄▁▅
wandb:        val_df_aup ▁▆▇▆▆█▇▆▇▇▇▇▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▅▇▅▆▅▅▅▅▆▅▅▄▅
wandb: val_df_logit_mean ▁▁▁▂▃▃▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇▇▇▇▇▇███▇██▇▇███
wandb:  val_df_logit_std ▁▁▂▂▃▃▅▄▅▆▅▆▆▆▆▆▆▇▆▇▇▇▇▇▇▇▇▇▇▇█████▇▇███
wandb:        val_dt_auc ▁▂▃▄▃▆▆▂▆▆▄▆▆▅▆▄▄▆▅▅▅▆▆▇▆█▆▅▇▅█▄▆█▅▇▃▇▄▇
wandb:        val_dt_aup ▁▁▂▃▄▅▆▄▆▆▅▆▇▆▇▆▆▇▆▆▆▇▇▇▇█▇▇▇▆█▇▇█▇█▆█▆▇
wandb:          val_loss ██▇▆▅▅▃▄▃▂▃▃▂▂▂▂▂▂▂▂▂▂▂▁▂▁▂▁▂▁▁▁▂▁▁▁▂▁▁▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.03271
wandb:            loss_r 0.67935
wandb:        train_loss 0.35603
wandb:      train_loss_l 0.03271
wandb:      train_loss_r 0.67935
wandb:        train_time 1.03806
wandb:        val_df_auc 0.64313
wandb:        val_df_aup 0.76552
wandb: val_df_logit_mean 0.57108
wandb:  val_df_logit_std 0.07738
wandb:        val_dt_auc 0.79174
wandb:        val_dt_aup 0.81632
wandb:          val_loss 0.69589
```
