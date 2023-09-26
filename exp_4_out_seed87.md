## Experiment 1:

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
test Logs:
{
'test_loss': 0.6560657024383545,
'test_dt_auc': 0.9684674887454214,
'test_dt_aup': 0.9682675346211134
}
```

### Deleting 100 edges outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 87 > gcn_gnndelete_out_0_100_Cora_87.txt &
```

Deleting the following edges (index):

[ 3, 5, 10, 17, 18, 19, 20, 22, 29, 31, 33, 36, 40, 47,
50, 53, 54, 55, 57, 62, 65, 69, 72, 73, 79, 84, 86, 88,
91, 98, 102, 104, 105, 108, 119, 130, 135, 143, 148, 154, 156, 158,
160, 164, 175, 176, 180, 181, 187, 193, 194, 197, 200, 201, 208, 214,
216, 218, 221, 224, 225, 226, 227, 232, 246, 252, 257, 259, 267, 269,
275, 277, 279, 284, 287, 290, 297, 304, 309, 313, 316, 317, 318, 320,
321, 324, 329, 335, 337, 339, 341, 343, 344, 348, 352, 354, 364, 372,
375, 381]

Results:

```python
test Logs:
{
'test_loss': 0.6802141070365906,
'test_dt_auc': 0.8592863164592474,
'test_dt_aup': 0.8761387984699991,
'test_df_auc': 0.8576217000000002,
'test_df_aup': 0.9056274791499006,
'test_df_logit_mean': 0.5119028371572495,
'test_df_logit_std': 0.02408952379997438
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▄▄▄▄▃▃▂▂▃▂▂▂▁▁▂▂▁▁▂▂▂▂▂▂▁▁▁▃▂▁▂▁▁▁▂▁▁▁▁
wandb:            loss_r ▄▃▃▄▄▃▃▄▄▃▁▄▄▃▁▄▄▄▆▂█▃▃▂▂▄▄▂▇▄▃▃▂▅▃▂▃▄▃▂
wandb:        train_loss ▁▁▂▂▂▂▂▃▂▂▃▂▂█▄▂▂▂▃▁▁▄▂▂▁▂▁▂▁▂▂▂▁▂▃▂▁▂▂▁
wandb:      train_loss_l █▅▄▄▃▃▃▂▂▃▂▂▂▁▁▂▂▁▁▂▂▂▂▂▂▁▁▁▃▂▁▁▁▁▁▁▁▁▁▁
wandb:      train_loss_r ▃▃▂▃▅▃▂▂▂█▂▃▃▂▁▃▃▃▅▂▃▂▂▂▄▃▂▃▅▂▂▃▂▄▂▃▂▄▅▂
wandb:        train_time ▂▃▅▄▄▄▁▃▅▃▂▄▆▂▄▆▆▄▆▆▅▆▃▄▆▇▇██▆▆▃▁▅▇▁▄▁▄▄
wandb:        val_df_auc ▄▃▂▆▅█▅▅▃▃▆▆▄▄▃▅▅▂▅▄▆▄▄▂▃▂▂▄▃▃▄▅▃▃▃▂▂▂▁▃
wandb:        val_df_aup ▆▃▂▅▅█▅▅▃▃▆▆▄▃▂▄▅▂▅▄▆▄▄▂▂▂▁▄▂▃▄▅▂▃▂▂▁▂▁▂
wandb: val_df_logit_mean ▁▄▃▃▄▄▄▅▆▅▆▅▆▇▇▆▆█▇▇▆▆▇▇▇▇▇▇▇██▇▇███▇█▇█
wandb:  val_df_logit_std ▁▄▄▄▅▄▅▅▆▅▆▆▆▇▇▇▇█▇▇▇▇▇▇▇▇▇▇███▇▇█▇▇▇███
wandb:        val_dt_auc ▁▄▄▆▆▇▆▇▆▆▇▇▇▇▆▇▇▆▇▆▇▆▇▆▇▆▆▇▅▇██▇▇▆▆▆▆▅▆
wandb:        val_dt_aup ▁▅▄▆▆▇▆▇▇▆█▇▇▇▆▇▇▇█▇▇▇▇▇▇▇▆█▆▇██▇▇▇▇▆▆▆▇
wandb:          val_loss █▄▄▃▃▂▂▂▂▂▁▂▂▁▁▁▁▁▁▂▂▂▂▂▂▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.01921
wandb:            loss_r 0.90056
wandb:        train_loss 0.45989
wandb:      train_loss_l 0.01921
wandb:      train_loss_r 0.90056
wandb:        train_time 2.6349
wandb:        val_df_auc 0.7962
wandb:        val_df_aup 0.87689
wandb: val_df_logit_mean 0.52551
wandb:  val_df_logit_std 0.04203
wandb:        val_dt_auc 0.859
wandb:        val_dt_aup 0.88108
wandb:          val_loss 0.6776
```

### Deleting next 100 edges outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 100 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --seqlearn True --epochs 2000 --random_seed 87 > gcn_gnndelete_out_100_200_Cora_87.txt &
```

Deleting the following edges (index):

[386, 391, 392, 393, 394, 396, 397, 404, 408, 413, 416, 426, 429, 432,
434, 436, 440, 446, 453, 456, 463, 464, 476, 481, 487, 488, 489, 491,
494, 497, 498, 504, 509, 516, 519, 522, 529, 530, 531, 534, 539, 543,
545, 548, 559, 563, 566, 569, 572, 579, 586, 587, 594, 595, 598, 610,
617, 619, 623, 626, 631, 633, 634, 637, 642, 650, 651, 654, 656, 660,
662, 664, 665, 668, 673, 688, 695, 705, 707, 708, 711, 715, 718, 720,
724, 725, 728, 730, 733, 739, 741, 742, 746, 748, 758, 761, 762, 765,
767, 768]

Results:

```python
{
'test_loss': 0.6786626577377319,
'test_dt_auc': 0.8738804470226025,
'test_dt_aup': 0.8872570802233322,
'test_df_auc': 0.8673473,
'test_df_aup': 0.9119593445088576,
'test_df_logit_mean': 0.5103856536746025,
'test_df_logit_std': 0.019654415841789622
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l ██▆▆▆▆▅▄▄▅▄▃▃▄▆▅▆▃▄▃█▄▃▃▅▄▃▄▆▂▃▆▃▂▁▂▃▁▃▁
wandb:            loss_r ▃▂▄▅▄▃▂▄▄▂▂▄▅▂▁▂▃▃▅▂█▃▂▂▁▃▄▁▆▄▁▄▂▅▃▂▂▃▂▂
wandb:        train_loss ▂▁▂▂▁▃▂▃▃▂▃▂▁█▄▁▂▃▃▂▁▄▂▂▄▂▁▁▁▂▂▂▁▂▃▃▁▁▂▁
wandb:      train_loss_l ██▅▆▆▆▅▄▄▅▄▃▃▆▆▅▆▃▅▃█▄▃▄▅▄▃▄▆▂▄▆▃▂▁▃▃▂▃▁
wandb:      train_loss_r ▃▃▂▅▆▃▂▂▁█▂▄▃▃▁▂▄▃█▂▃▂▂▂▄▃▃▄▆▂▂▃▁▄▂▄▂▅▄▁
wandb:        train_time ▄▁█▇▂▄▄▅▂▄▅▆▅▅▅▅▂▃▄▄▁▅▄▅▁▄▇▂▃▇▁▅▄▅▃▃▅▂▆▅
wandb:        val_df_auc ▅█▂▃▄▂▇▃▄▃▅▃▂▃▃▄▄▃▅▁▆▄▆▆▅▄▃▂▅▃▇▆▃▄▃▄▄▂▁▃
wandb:        val_df_aup ▅█▂▃▄▂▇▃▄▃▅▃▂▂▃▄▄▃▄▁▆▄▆▅▄▃▃▂▄▃▆▅▃▃▂▃▃▁▁▃
wandb: val_df_logit_mean ▃▁▃▃▁▂▂▃▂▃▃▃▅▅▃▃▄▆▄▄▄▄▄▄▅▄▅▄▄▅▅▅▆▆▇▇▅▇██
wandb:  val_df_logit_std ▂▁▂▃▂▂▃▃▂▂▄▄▄▄▃▃▄▅▄▄▄▄▄▄▅▅▅▅▅▅▅▆▇▆▇▇▇███
wandb:        val_dt_auc ▅▇▃▄▄▁▆▅▄▃▅▄▃▅▄▅▅▅▇▃▅▄█▇▆▄▅▁▄▄█▆▄▇▄▆▄▃▂▄
wandb:        val_dt_aup ▅▆▃▄▄▁▆▅▅▃▅▄▄▅▄▅▅▅▇▃▅▅█▇▆▄▅▁▄▄█▆▅▇▄▆▄▃▂▅
wandb:          val_loss ▅▇▇▆▇█▅▅▅▆▄▃▃▅▇▅▅▃▅▅▆▄▃▄▅▅▄▆▅▃▄▅▂▂▁▂▄▂▂▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.01199
wandb:            loss_r 0.84046
wandb:        train_loss 0.42623
wandb:      train_loss_l 0.01199
wandb:      train_loss_r 0.84046
wandb:        train_time 3.47187
wandb:        val_df_auc 0.81277
wandb:        val_df_aup 0.88638
wandb: val_df_logit_mean 0.51521
wandb:  val_df_logit_std 0.0272
wandb:        val_dt_auc 0.85328
wandb:        val_dt_aup 0.87631
wandb:          val_loss 0.67865
```

### Deleting 200 edges (Union set) outside 2-hop neighbourhood of test dataset

Reproduce:

```bash
nohup python exp_delete_gnn.py --df_size 200 --df out --dataset Cora --gnn gcn --unlearning_model gnndelete --epochs 2000 --random_seed 87 > gcn_gnndelete_out_union200_Cora_87.txt &
```

Results:

```python
{
'test_loss': 0.6890530586242676,
'test_dt_auc': 0.8185145935338596,
'test_dt_aup': 0.8390642912616793,
'test_df_auc': 0.77219315,
'test_df_aup': 0.8524648933545613,
'test_df_logit_mean': 0.5089609195291996,
'test_df_logit_std': 0.018709370857960636
}
```

```bash
wandb: Run history:
wandb:             Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            loss_l █▆▅▃▃▃▂▂▂▂▃▂▃▂▃▂▂▂▂▂▂▂▂▁▁▁▂▂▁▂▂▂▂▂▂▂▁▁▁▁
wandb:            loss_r ▄▁▃▃▅▃▅▃▂▄█▄▄▂▆▄▃▄▂▂▆▃▃▃▅▆▃▂▃▂▆▄▂▄▃▂▆▃▁▂
wandb:        train_loss ▂▂▂▄▂▂▄▃▂▂▂▂▁▂▃▂▃▂▂▂▃▃▂▁▂▂▇▄▂▂▂▂▂▂▂▃▂▂█▃
wandb:      train_loss_l █▆▅▃▂▂▂▂▂▂▃▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▂▂▁▁▂▂▂▂▂▁▁▁▁▁
wandb:      train_loss_r ▃▁▅▃▁▅▆▃▃▆▄▅▃▂█▃▂▅▄▃▂▃▃▆▄█▂▅▄▄▃▂▅▆▅▅▇▆▃▄
wandb:        train_time ▄▆▇▇▆▆▅▆▅▄▅▄▃█▆▆▅▅▇▄▇▇▆▇██▆▅▆▆▄▆▆█▅▅█▅▇▁
wandb:        val_df_auc █▅▅▆▃▄▃▅▄▃▃▄▂▃▄▃▁▁▂▂▂▃▄▂▂▂▂▂▁▄▃▄▃▃▃▄▁▃▄▃
wandb:        val_df_aup █▃▄▅▃▃▂▃▃▂▂▃▁▂▃▂▁▁▁▂▂▂▃▂▂▂▂▂▁▃▂▃▂▂▂▃▁▂▃▂
wandb: val_df_logit_mean ▁▂▃▃▄▄▅▅▅▆▅▆▆▆▆▆▇▇▆▇▇▆▇▇█▇▇▇█▇▇▆▇▇▇▇█▇█▇
wandb:  val_df_logit_std ▁▂▃▄▅▅▅▅▅▆▅▆▆▆▆▆▆▆▆▆▇▇▇▇▇▇▇▇██▇▇▇▇▇█████
wandb:        val_dt_auc ▁▄▅▆▅▆▇▇▇▇▇█▆▆▇▇▆▅▆▆▆▆▇▆▆▆▆▅▆█▅▆▇▆▆▇▅▆▇▆
wandb:        val_dt_aup ▁▄▅▇▆▇▇▇▇▇▇█▇▇▇▇▇▆▇▇▇▇█▇▇▇▇▇▇█▆▇▇▇▇▇▇▇█▇
wandb:          val_loss █▅▄▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▂▁▁▁▂▂▁▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:             Epoch 1999
wandb:             epoch 1999
wandb:            loss_l 0.02187
wandb:            loss_r 0.88382
wandb:        train_loss 0.45285
wandb:      train_loss_l 0.02187
wandb:      train_loss_r 0.88382
wandb:        train_time 1.70162
wandb:        val_df_auc 0.72423
wandb:        val_df_aup 0.83317
wandb: val_df_logit_mean 0.52479
wandb:  val_df_logit_std 0.04532
wandb:        val_dt_auc 0.81484
wandb:        val_dt_aup 0.84379
wandb:          val_loss 0.68606
```
