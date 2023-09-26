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
'test_loss': 0.6568805575370789,
'test_dt_auc': 0.9678890341862516,
'test_dt_aup': 0.9667993509795219
}
```

### Deleting 100 edges outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 42 > gcn_gnndelete_out_0_100_Cora_42.txt &
```

Deleting the following edges (index):

[ 0, 7, 18, 35, 48, 50, 55, 56, 57, 58, 63, 65, 83, 86,
87, 89, 91, 95, 98, 99, 101, 106, 112, 113, 119, 120, 121, 123,
125, 127, 128, 133, 135, 136, 142, 144, 147, 150, 155, 157, 159, 163,
166, 167, 169, 170, 176, 178, 186, 190, 192, 195, 196, 198, 202, 204,
208, 209, 211, 218, 219, 224, 225, 227, 236, 237, 240, 246, 247, 250,
253, 255, 258, 263, 273, 274, 277, 279, 283, 285, 293, 299, 301, 305,
309, 312, 314, 317, 319, 323, 324, 326, 328, 338, 341, 344, 345, 346,
347, 348]

Results:

```python
test Logs:
{
'test_loss': 0.6836599707603455,
'test_dt_auc': 0.8531189356873694,
'test_dt_aup': 0.865351536555352,
'test_df_auc': 0.8637157999999999,
'test_df_aup': 0.9086175200306601,
'test_df_logit_mean': 0.505244964659214,
'test_df_logit_std': 0.007910410968484974
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▇▅▅▄▃▃▃▂▂▂▂▂▂▁▂▁▁▁▂▂▂▂▂▂▂▂▁▁▁▂▁▁▁▁▁▁▁▁▁
wandb:            loss_r ▆▂▆▅▄▃▃▃▂▃▃▅▇▂▄▅▇▃█▄▄▇▁▁▂▄▄▆▃▁▅▃▇▅▃▄▄▃▃▅
wandb:        train_loss ▄▃▂▄▆▄▃▇▇▅▃▆▆▃▆▇▃▆▅▄▅▃▄█▃▄▅▄▅▃▃▄▅▅▁▄▅▃▃▃
wandb:      train_loss_l █▇▅▅▄▃▃▂▃▃▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▁▂▁▂▁▁▂▁▁▁▁▁▁
wandb:      train_loss_r ▃▄▃▃▃▃▂▅▂▂▃▃█▃▃▂▅▂▃▃▄▃▁▄▄▃▇▅▂▄▃▆▇▄▂▅▃▆▅▂
wandb:        train_time ▁▅▇▇▆█▆▇▇▆▇▇▆▆▅▅▇▅▆▆▆▇▆▆▆▆▆▆▇▆▇▆██▅▅▆▆▅▆
wandb:        val_df_auc ██▂▅▅▄▄▃▂▄▂▂▄▂▂▄▅▄▄▁▄▆▆▅▄▄▄▆▃▃▅▄▁▃▄▃▁▂▃▃
wandb:        val_df_aup █▆▁▄▄▃▃▃▂▃▁▁▃▂▂▃▄▃▃▁▃▄▄▄▂▃▃▄▂▂▃▃▁▂▃▂▁▁▂▂
wandb: val_df_logit_mean ▁▂▂▂▃▃▄▄▄▄▅▅▅▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▇▇▇▇▇▇▇▇█▇▇
wandb:  val_df_logit_std ▁▁▂▂▃▃▄▅▅▅▅▆▆▆▆▆▆▇▇▆▆▇▆▆▇▇▇▇▇▇▇▇█▇██████
wandb:        val_dt_auc ▁▅▃▅▆▆▆▅▅▆▅▆▇▆▅▇▇▆▆▅▆▇▇▇▆▆▆▇▆▇█▇▅▆█▇▆▇▇▇
wandb:        val_dt_aup ▁▄▄▅▆▇▆▆▆▇▆▇█▇▇▇▇▇▇▆▇▇██▇▇▇█▇███▆▇██▇█▇▇
wandb:          val_loss █▇▅▄▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▂▂▁▁▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.01862
wandb:            loss_r 0.91759
wandb:        train_loss 0.4681
wandb:      train_loss_l 0.01862
wandb:      train_loss_r 0.91759
wandb:        train_time 3.01471
wandb:        val_df_auc 0.80861
wandb:        val_df_aup 0.88061
wandb: val_df_logit_mean 0.53184
wandb:  val_df_logit_std 0.06219
wandb:        val_dt_auc 0.84314
wandb:        val_dt_aup 0.86872
wandb:          val_loss 0.68109
```

### Deleting next 100 edges outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --seqlearn True --epochs 2000 --random_seed 42 > gcn_gnndelete_out_100_200_Cora_42.txt &
```

Deleting the following edges (index):

[351, 352, 354, 358, 360, 368, 374, 375, 380, 381, 384, 386, 389, 392,
402, 405, 407, 410, 418, 427, 428, 435, 440, 442, 443, 445, 451, 455,
456, 458, 462, 463, 466, 468, 473, 476, 478, 480, 481, 482, 495, 497,
499, 500, 502, 507, 511, 512, 524, 526, 537, 538, 543, 545, 547, 553,
557, 564, 565, 580, 581, 583, 589, 590, 591, 592, 596, 597, 604, 605,
607, 609, 611, 613, 617, 618, 622, 623, 631, 633, 634, 635, 639, 641,
643, 645, 647, 648, 649, 656, 667, 672, 675, 678, 680, 685, 695, 696,
705, 707]

Results:

```python
{
'test_loss': 0.6869715452194214,
'test_dt_auc': 0.8375741566015176,
'test_dt_aup': 0.8491006984134866,
'test_df_auc': 0.8706846,
'test_df_aup': 0.9117488594102973,
'test_df_logit_mean': 0.50241644769907,
'test_df_logit_std': 0.0038767550304040508
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l ██▇▆▅▄▄▃▄▄▅▃▂▂▁▂▂▄▃▂▂▃▂▂▂▁▁▁▂▂▂▂▂▂▂▂▁▁▃▂
wandb:            loss_r ▅▃▆▄▄▄▅▂▄▃▃▂▂▅▅▃▇▃▅▃▃▆▂▂▂▃▄▄▄▁▄▃█▃▄▄▂▆▃▅
wandb:        train_loss ▅▂▂▃▄▃▄▅▄▂▂▄▄▄▅▆▃▄█▃▂▂▃▅▂▂▂▂▃▃▂▃▃▃▁▄▃▁▁▂
wandb:      train_loss_l ██▇▆▅▄▄▃▄▄▄▃▂▂▁▂▁▄▃▂▂▂▂▂▂▁▁▁▂▂▂▂▂▂▂▂▁▁▃▂
wandb:      train_loss_r ▂▂▂▃▄▂▃▇▁▁▂▁█▂▃▂▂▂▂▂▃▂▂▃▃▂▄▅▃▃▂▄▆▂▁▆▁▅▅▁
wandb:        train_time ▇▁▄█▇▆▇▄▃▆█▃▁█▆▆▇▃▃▆▂▇▃█▄▃▃▅█▃▄▄▃▇▅▇▇▁▁█
wandb:        val_df_auc ▅█▆▇▅▅▂▅▃▁▃▃▄▆▃▇▃▇▃▄▄▅▄▅▆▂▃▆▃▄▄▄▃▄▂▅▃▂▄▁
wandb:        val_df_aup ▅█▆█▅▄▂▄▃▁▂▃▄▆▂▇▂█▃▄▄▅▃▅▅▁▂▆▂▄▄▃▃▃▂▄▃▂▄▁
wandb: val_df_logit_mean ▄▂▁▁▂▅▂▃▄▃▃▃▃▄▅▅▅▅▅▅▄▄▅▆▆▇█▅▆▆▆▆▆█▆▅▆▇▆▇
wandb:  val_df_logit_std ▂▂▂▁▂▅▂▃▃▃▂▃▃▄▅▅▆▆▅▆▅▅▅▅▆▆█▆▆▅▅▅▅█▅▅▅▆▆▆
wandb:        val_dt_auc ▅▇▄▆▄▅▂▅▄▁▃▄▄▆▃█▃▇▃▅▅▆▆▆█▄▄█▄▆▆▆▅▆▄▆▆▅▆▂
wandb:        val_dt_aup ▄▅▃▄▃▅▁▅▄▁▃▄▅▆▄█▄▇▄▅▅▆▆▆█▄▅█▅▆▆▆▅▆▄▆▆▅▆▃
wandb:          val_loss ██▇▆▆▄▆▄▄▅▅▄▃▂▂▁▂▁▃▂▃▃▂▂▁▂▂▁▂▂▂▂▂▂▂▁▁▁▂▂
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.01046
wandb:            loss_r 0.88611
wandb:        train_loss 0.44829
wandb:      train_loss_l 0.01046
wandb:      train_loss_r 0.88611
wandb:        train_time 0.91965
wandb:        val_df_auc 0.75727
wandb:        val_df_aup 0.85742
wandb: val_df_logit_mean 0.50735
wandb:  val_df_logit_std 0.01345
wandb:        val_dt_auc 0.80668
wandb:        val_dt_aup 0.83941
wandb:          val_loss 0.68565
```

### Deleting 200 edges (Union set) outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 200 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 42 > gcn_gnndelete_out_union200_Cora_42.txt &
```

Results:

```python
{
'test_loss': 0.6966456770896912, 'test_dt_auc': 0.7735501317173801, 'test_dt_aup': 0.7918058667886952, 'test_df_auc': 0.8725864249999999, 'test_df_aup': 0.9270784974938192, 'test_df_logit_mean': 0.5001861852407455, 'test_df_logit_std': 0.00029533268999435596
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▆▄▄▃▄▃▂▂▂▂▃▂▂▂▂▂▁▁▁▁▂▂▁▁▁▂▂▂▂▂▂▂▂▁▂▁▂▂▂
wandb:            loss_r ▆▅▄▇▁▂▄▃▂▃▃▁▄▄▄▆▃▅▄▁▃▃▂▆█▃▂▆▂▄▄▅▃▃▅▄▂▂▃▃
wandb:        train_loss ▂▂▄▃▃▂▃▅▂▃▃▄▂▄▁▂▃▁▃▁▂▃▅▃▃▄▃▄▂▄▃▂▄▂▃▄█▁▁▁
wandb:      train_loss_l █▇▄▄▄▄▃▂▂▂▂▃▂▂▂▂▂▁▁▂▁▂▂▁▁▂▂▂▂▂▂▂▁▂▂▂▁▂▂▂
wandb:      train_loss_r █▄▂▅▂▃▃▂▁▂▃▁▄▁▃▂▅▃▂▂▂▂▁▂▄▂▁▁▂▂▂▃▃▂▂▂▂▂▂▂
wandb:        train_time ▁▅▇▅▆▄▅▅▂▇▃▇▆▃▄▄▅▃▄▇▅▄▆▁▂▁▄█▆▄▅▆▆▄▃▄▄▆▅▄
wandb:        val_df_auc ▄▄▇▄▅▂▂▁█▃▂▅▄▃▅▄▆▂▃▅▃▅▃▃▂▃▄▂▃▂▂▄▂▁▃▃▂▅▃▃
wandb:        val_df_aup ▁▄▇▆▆▄▄▄█▅▄▅▅▅▆▆▆▄▅▆▅▆▅▅▄▅▆▄▅▄▄▅▄▄▅▅▄▆▅▅
wandb: val_df_logit_mean ▁▁▃▂▂▃▄▅▆▆▅▆▆▆▆▇▆▇█▇▇▇▇██▇▇▇▇██▇███▇██▇▇
wandb:  val_df_logit_std ▁▂▂▂▃▃▄▅▆▆▆▆▆▆▆▇▆▇▇▇▇▇▇███▇▇▇▇▇▇▇██▇████
wandb:        val_dt_auc ▁▁▆▂▄▁▃▃█▄▃▅▅▅▅▅▇▅▇█▇▇▆▇▆▆▆▄▇▅▅▅▅▄▆▅▄▇▄▅
wandb:        val_dt_aup ▁▂▆▄▅▄▅▆█▆▆▆▇▇▇▇▇▇██▇█▇█▇▇▇▆▇▆▇▇▇▆▇▇▆█▆▇
wandb:          val_loss █▇▄▄▃▃▃▂▁▂▂▂▂▁▂▂▂▁▁▁▁▁▂▁▁▁▁▂▂▂▂▂▁▁▁▁▁▁▂▂
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.02441
wandb:            loss_r 0.78954
wandb:        train_loss 0.40698
wandb:      train_loss_l 0.02441
wandb:      train_loss_r 0.78954
wandb:        train_time 2.98134
wandb:        val_df_auc 0.71349
wandb:        val_df_aup 0.82616
wandb: val_df_logit_mean 0.52215
wandb:  val_df_logit_std 0.04888
wandb:        val_dt_auc 0.78391
wandb:        val_dt_aup 0.81986
wandb:          val_loss 0.6906
```
