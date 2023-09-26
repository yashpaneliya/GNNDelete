## Experiment 1:

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
test Logs:
{
'test_loss': 0.6566945314407349,
'test_dt_auc': 0.970374454476029,
'test_dt_aup': 0.9698457383109746
}
```

### Deleting 100 edges outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 13 > gcn_gnndelete_out_0_100_Cora_13.txt &
```

Deleting the following edges (index):

[ 3, 5, 10, 11, 12, 13, 15, 20, 36, 42, 46, 50, 51, 52,
53, 54, 56, 57, 59, 60, 61, 62, 68, 69, 73, 82, 98, 101,
103, 106, 113, 115, 116, 118, 122, 123, 125, 127, 130, 132, 135, 138,
139, 146, 156, 159, 173, 180, 182, 183, 187, 191, 192, 197, 205, 207,
208, 210, 213, 215, 216, 218, 226, 227, 229, 232, 236, 241, 251, 254,
257, 265, 266, 270, 274, 276, 279, 283, 286, 291, 296, 298, 305, 325,
331, 332, 337, 340, 344, 346, 353, 354, 355, 356, 359, 367, 380, 389,
394, 396]

Results:

```python
test Logs:
{
'test_loss': 0.6898840665817261,
'test_dt_auc': 0.8235623591716995,
'test_dt_aup': 0.8340891485795627,
'test_df_auc': 0.8811762,
'test_df_aup': 0.9248429283256069,
'test_df_logit_mean': 0.5013490855693817,
'test_df_logit_std': 0.002777667376317628
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▆▅▄▄▄▃▃▂▂▂▂▃▂▂▃▂▂▂▂▂▂▂▂▁▁▁▁▂▂▁▂▂▁▂▂▂▂▂▁
wandb:            loss_r ▂▅▅▁▄▆▅▃▄▃▄▅▅▂▅▆▆▄▇▆▄▅▇▄▃▇▅▄▇▇▆█▄▃▅▂▅▅▅▅
wandb:        train_loss ▅▅▆▆▆▅▅▇▄▇▁▅▃▃▆▃▃▅▄▇▄▃█▄▂▃▅▃▃▆▄▃▃▂▄▄▃▃▆▅
wandb:      train_loss_l █▆▅▄▄▄▃▃▂▂▂▂▂▂▂▃▂▂▂▂▂▁▂▂▂▁▁▁▂▂▁▂▂▁▂▂▂▃▂▁
wandb:      train_loss_r ▂▃▄▁▅▂▄▇▄▅▃▄▅▃▄▄▄▃▄▅▂▄▆▅▃▆▁▅▆▆█▅▃▃▄▄▄▄▄▃
wandb:        train_time ▅▅▇▅▇▅▇▄▂▇▆█▄▅▆▃▅▆▇▆▄▅█▇▅▆▄▅▆█▆▆▅▆▅▆▄▅▃▁
wandb:        val_df_auc █▆▄▄▁▄▂▅▄▃▄▂▂▃▂▄▃▃▃▂▂▁▁▂▂▃▄▃▅▃▄▃▂▃▂▂▂▃▃▂
wandb:        val_df_aup █▅▃▄▁▄▂▄▃▂▃▂▂▂▂▃▂▂▂▂▂▁▁▂▂▃▃▂▃▃▃▂▁▂▂▂▂▃▂▂
wandb: val_df_logit_mean ▁▂▃▃▃▃▅▄▅▅▅▇▅▆▆▅▆▆▇▇▇█▇▇▇▇█▇▆▇█▇█▇▇▇▆▆▆▇
wandb:  val_df_logit_std ▁▃▄▃▃▄▅▄▆▅▆▇▆▆▆▆▇▇▇▇▇▇▇▇▇██▇▇▇█▇▇▇▇▇▇▆▆▇
wandb:        val_dt_auc ▃▅▄▄▁▅▄▇▅▄▅▅▃▆▄▆▅▅▇▄▆▄▃▆▃▆█▅█▆▆▇▅▆▄▄▄▆▆▅
wandb:        val_dt_aup ▁▄▄▄▂▅▅▇▆▅▆▆▅▆▅▆▆▆▇▅▇▆▅▆▅▇█▆█▇▇▇▆▇▆▅▆▇▇▆
wandb:          val_loss █▅▅▄▄▄▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▁▁▁▂▁▁▁▂▁▂▂▂▂▂▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.02128
wandb:            loss_r 0.97422
wandb:        train_loss 0.49775
wandb:      train_loss_l 0.02128
wandb:      train_loss_r 0.97422
wandb:        train_time 2.54264
wandb:        val_df_auc 0.76763
wandb:        val_df_aup 0.86045
wandb: val_df_logit_mean 0.52479
wandb:  val_df_logit_std 0.04731
wandb:        val_dt_auc 0.83333
wandb:        val_dt_aup 0.86155
wandb:          val_loss 0.68196
```

### Deleting next 100 edges outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --seqlearn True --epochs 2000 --random_seed 13 > gcn_gnndelete_out_100_200_Cora_13.txt &
```

Deleting the following edges (index):

[399, 401, 405, 416, 421, 428, 437, 438, 439, 448, 449, 452, 453, 454,
458, 460, 464, 467, 470, 476, 477, 487, 488, 495, 498, 499, 500, 501,
504, 505, 509, 510, 513, 515, 516, 517, 524, 527, 529, 536, 545, 546,
553, 555, 566, 573, 574, 577, 582, 587, 593, 595, 600, 602, 606, 610,
612, 619, 620, 621, 622, 626, 627, 628, 638, 639, 644, 653, 655, 659,
660, 672, 673, 674, 677, 689, 691, 693, 694, 696, 697, 699, 706, 710,
711, 713, 717, 718, 720, 734, 740, 744, 750, 758, 764, 766, 767, 769,
775, 779]

Results:

```python
{
'test_loss': 0.6846030354499817, 
'test_dt_auc': 0.8483733507729948, 
'test_dt_aup': 0.8607054433146181, 
'test_df_auc': 0.8757965000000001, 
'test_df_aup': 0.9171896510313967, 
'test_df_logit_mean': 0.5055030697584152, 
'test_df_logit_std': 0.013152277135751107
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l ▇█▇▇▅▄▄▄▃▃▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▃▂▂▂▂▁▁▁▁▂▁▁
wandb:            loss_r ▁█▃▅▄▄▄▃▂▃▃▄▃▃▄▆▅▂▅▅▄▃▇▄▅▄▄▂▇▄▅▄▅▂▄▁▂▅▃▄
wandb:        train_loss ▂▅▅▅▇▅▄▆▄█▁▄▃▃▅▃▄▄▄▆▆▆▇▄▂▄▄▂▃▅▅▃▅▃▄▅▃▂▄█
wandb:      train_loss_l ▇█▆▇▅▄▄▄▃▃▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▃▂▂▂▂▁▁▁▁▂▁▁
wandb:      train_loss_r ▁▅▄▅▃▅▄▇▃▂▃▄█▅▅▃▇▂▅█▃▅▇▃▂▄▂▅█▅▅▆▃▂▅▃▂▅▃█
wandb:        train_time ▅▅▆▄▃▄▇▄▂▄▃▃▁▅▁▁▄▃▄▆▆▅▅▃▄▂▄█▃▆▄▄▄█▄▆▅▃▃▇
wandb:        val_df_auc ▆█▃▃▅▃█▄▃▃▅▅▃▂▃▃▄▅▂▃▂▄▂▃▄▅▅▂█▅▂▄▃▁▄▄▁▁▅▃
wandb:        val_df_aup ▅█▂▃▄▂▇▄▃▂▄▄▂▂▂▂▃▄▂▂▂▃▂▂▃▃▄▂▆▄▂▃▂▁▃▃▁▁▄▂
wandb: val_df_logit_mean ▂▁▃▂▃▄▃▃▄▄▄▄▄▅▅▅▅▅▆▇▆▆▆▇▆▆▆▆▅▆▆▆▆▇▇▇█▇▇▇
wandb:  val_df_logit_std ▂▁▃▃▄▅▄▄▄▅▄▄▅▅▅▆▅▆▆▇▇▇▇▇▇▇▇▇▆▆▆▆▇▇███▇▇█
wandb:        val_dt_auc ▃▃▃▁▅▄▆▄▄▃▅▆▃▄▄▄▆▆▅▄▄▆▅▅▅▆▆▄█▆▄▅▅▄▆▆▅▃▇▆
wandb:        val_dt_aup ▃▂▃▁▄▄▆▄▄▄▅▆▄▅▅▅▆▆▆▅▅▇▆▆▆▇▇▅█▇▅▆▆▅▇▇▆▅█▇
wandb:          val_loss ▇█▆▇▅▄▅▄▄▄▄▃▄▃▃▃▃▃▂▃▂▂▂▃▂▂▂▂▂▂▃▃▂▂▁▁▁▂▂▂
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.01122
wandb:            loss_r 1.50065
wandb:        train_loss 0.75594
wandb:      train_loss_l 0.01122
wandb:      train_loss_r 1.50065
wandb:        train_time 2.61097
wandb:        val_df_auc 0.78755
wandb:        val_df_aup 0.87212
wandb: val_df_logit_mean 0.51237
wandb:  val_df_logit_std 0.02508
wandb:        val_dt_auc 0.83747
wandb:        val_dt_aup 0.85874
wandb:          val_loss 0.68389
```

### Deleting 200 edges (Union set) outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 200 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 13 > gcn_gnndelete_out_union200_Cora_13.txt &
```

Results:

```python
{
'test_loss': 0.6955719590187073, 
'test_dt_auc': 0.8105065308728057, 
'test_dt_aup': 0.819093243263234, 
'test_df_auc': 0.794599125, 
'test_df_aup': 0.8610436886262411, 
'test_df_logit_mean': 0.5053741401433944, 
'test_df_logit_std': 0.0116040656839674
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▆▅▄▄▃▃▂▂▂▃▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▂▂▂▂▁▂▂▁▁▂▁▁▁▁
wandb:            loss_r ▄▄▆▄▃▂▁▅▅▅▂▄▆▃▅▅▄▇▂▃▂▂▄▄█▇▅▅▃▁▂▃▂▂▅█▅▂▇▅
wandb:        train_loss ▆▅▄▅█▁▃▅▃▆▆▅▁▂▅▄██▃▆▄▆▂▅▅▃▃▆▄▄▄▃▄▅▂▆▆▇▄▃
wandb:      train_loss_l █▇▅▅▄▃▃▃▂▃▃▂▂▂▃▂▂▂▂▂▂▁▁▂▁▁▂▂▂▂▂▂▃▂▁▂▁▁▁▁
wandb:      train_loss_r ▄▅▂▄▃▂▁▃█▁▅▄▄▄▆▃▄█▂▄▁▅▄▆▇█▃▁▃▃▂▇▂▂▃▅▆▃▃▆
wandb:        train_time ▆▆▆▆▅▆▆▆▇▆▆▆▇▅▆▆▆▅▅▅▆▆▆▇▇▆▅▆▆█▆▆▅▅▆▆▆▆▅▁
wandb:        val_df_auc ▁▆██▃▃▂▄▅▃▃▃▄▃▅▃▃▃▃▂▂▃▃▃▃▄▃▂▂▂▁▂▁▂▂▁▂▂▂▂
wandb:        val_df_aup ▁▆██▅▅▄▅▅▅▄▄▅▄▅▄▄▄▄▃▄▄▄▄▄▅▄▄▃▃▃▄▃▄▃▃▄▄▃▃
wandb: val_df_logit_mean ▂▁▁▂▂▃▄▅▄▅▅▆▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇█▇▇▇█▇
wandb:  val_df_logit_std ▁▁▂▂▃▃▄▄▄▄▅▅▅▆▅▆▆▇▆▆▇▇▇▇▇▇█▇▇▇▇▇▇▇█▇████
wandb:        val_dt_auc ▁▄▆▇▅▅▅▇▇▆▆▇▇▆█▇▆▆▆▆▇▇▆▇▇█▇▇▆▅▆█▅▆▆▅▆▆▆▆
wandb:        val_dt_aup ▁▄▅▆▆▆▆▇▇▇▇▇▇▇█▇▇▇▇▇▇█▇▇███▇▇▇▇█▆▇▇▆▇▇▇▇
wandb:          val_loss █▆▅▄▄▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▂▁▁▂▂▂▂▁▂▂▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.02398
wandb:            loss_r 0.93305
wandb:        train_loss 0.47852
wandb:      train_loss_l 0.02398
wandb:      train_loss_r 0.93305
wandb:        train_time 1.0488
wandb:        val_df_auc 0.70325
wandb:        val_df_aup 0.81944
wandb: val_df_logit_mean 0.52378
wandb:  val_df_logit_std 0.04964
wandb:        val_dt_auc 0.79865
wandb:        val_dt_aup 0.83097
wandb:          val_loss 0.69017
```
