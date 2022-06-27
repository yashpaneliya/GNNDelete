import os
import time
import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from .base import Trainer
from ..evaluation import *
from ..utils import *


class RetrainTrainer(Trainer):

    def freeze_unused_mask(self, model, edge_to_delete, subgraph, h):
        gradient_mask = torch.zeros_like(delete_model.operator)
        
        edges = subgraph[h]
        for s, t in edges:
            if s < t:
                gradient_mask[s, t] = 1
        gradient_mask = gradient_mask.to(device)
        model.operator.register_hook(lambda grad: grad.mul_(gradient_mask))
        
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        start_time = time.time()
        best_valid_loss = 100000
        loss_fct = nn.MSELoss()

        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before
        
        for epoch in trange(args.epochs, desc='Unlearning'):
            model.train()
            total_step = 0
            total_loss = 0

            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index[:, data.dr_mask],
                num_nodes=data.num_nodes,
                num_neg_samples=data.dr_mask.sum())

            z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])
            logits = model.decode(z, data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            label = self.get_link_labels(data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, label)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            total_step += 1
            total_loss += loss.item()

            log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'loss': loss.item(),
            }
            wandb.log(log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
            tqdm.write(' | '.join(msg))

            valid_loss, auc, aup, df_logt, logit_all_pair = self.eval(model, data, 'val')

            self.trainer_log['log'].append({
                'epoch': epoch,
                'dt_loss': valid_loss,
                'dt_auc': auc,
                'dt_aup': aup
            })

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch

                print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                ckpt = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }
                torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
        
        self.trainer_log['training_time'] = time.time() - start_time

        # Save
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

