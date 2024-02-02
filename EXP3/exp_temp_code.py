            # Generating metrics for One distance edges
            # tensor_onel_edge_index = torch.tensor(onel).T
            # onel_neg_edge_index = generate_negative_edges_from_edges(tensor_onel_edge_index, len(onel))
            # one_pos_logits = model.decode(z, tensor_onel_edge_index, onel_neg_edge_index).sigmoid()
            # one_pos_labels = self.get_link_labels(tensor_onel_edge_index, onel_neg_edge_index)
            # print("Logits len:",len(one_pos_logits))
            # # print(one_pos_logits)
            # print("Lables len:",len(one_pos_labels))
            # # print(one_pos_labels)
            
            # one_pos_loss = F.binary_cross_entropy_with_logits(one_pos_logits, one_pos_labels, reduction='none')
            # print("POS LOSS: ", one_pos_loss.size())
            # one_pos_dt_auc = roc_auc_score(one_pos_labels.cpu(), one_pos_logits.cpu())
            # print("POS AUC: ", one_pos_dt_auc)
            # one_pos_dt_aup = average_precision_score(one_pos_labels.cpu(), one_pos_logits.cpu())
            # print("POS AUP: ", one_pos_dt_aup)


# ============================ EXPERIMENT CODE ==============================


 # Distance > 1 and <=2
            # print("Distance 2:")
            # tensor_twol_edge_index = torch.tensor(twol).T
            # twol_neg_edge_index = generate_negative_edges_from_edges(tensor_twol_edge_index, len(twol))
            # twol_pos_logits = model.decode(z, tensor_twol_edge_index, twol_neg_edge_index).sigmoid()
            # twol_pos_labels = self.get_link_labels(tensor_twol_edge_index, twol_neg_edge_index)
            # print("Logits len:",len(twol_pos_logits))
            # # print(twol_pos_logits)
            # print("Lables len:",len(twol_pos_labels))
            # # print(twol_pos_labels)
            
            # twol_pos_loss = F.binary_cross_entropy_with_logits(twol_pos_logits, twol_pos_labels, reduction='none')
            # print("POS LOSS: ", twol_pos_loss.size())
            # twol_pos_dt_auc = roc_auc_score(twol_pos_labels.cpu(), twol_pos_logits.cpu())
            # print("POS AUC: ", twol_pos_dt_auc)
            # twol_pos_dt_aup = average_precision_score(twol_pos_labels.cpu(), twol_pos_logits.cpu())
            # print("POS AUP: ", twol_pos_dt_aup)

            # # Distance > 2 and <=3
            # print("Distance 3:")
            # tensor_threel_edge_index = torch.tensor(threel).T
            # threel_neg_edge_index = generate_negative_edges_from_edges(tensor_threel_edge_index, len(threel))
            # threel_pos_logits = model.decode(z, tensor_threel_edge_index, threel_neg_edge_index).sigmoid()
            # threel_pos_labels = self.get_link_labels(tensor_threel_edge_index, threel_neg_edge_index)
            # print("Logits len:",len(threel_pos_logits))
            # # print(threel_pos_logits)
            # print("Lables len:",len(threel_pos_labels))
            # # print(threel_pos_labels)
            
            # threel_pos_loss = F.binary_cross_entropy_with_logits(threel_pos_logits, threel_pos_labels, reduction='none')
            # print("POS LOSS: ", threel_pos_loss.size())
            # threel_pos_dt_auc = roc_auc_score(threel_pos_labels.cpu(), threel_pos_logits.cpu())
            # print("POS AUC: ", threel_pos_dt_auc)
            # threel_pos_dt_aup = average_precision_score(threel_pos_labels.cpu(), threel_pos_logits.cpu())
            # print("POS AUP: ", threel_pos_dt_aup)

            # # Distance > 3 and <=4
            # print("Distance >=4:")
            # tensor_fourl_edge_index = torch.tensor(fourl).T
            # fourl_neg_edge_index = generate_negative_edges_from_edges(tensor_fourl_edge_index, len(fourl))
            # fourl_pos_logits = model.decode(z, tensor_fourl_edge_index, fourl_neg_edge_index).sigmoid()
            # fourl_pos_labels = self.get_link_labels(tensor_fourl_edge_index, fourl_neg_edge_index)
            # print("Logits len:",len(fourl_pos_logits))
            # # print(fourl_pos_logits)
            # print("Lables len:",len(fourl_pos_labels))
            # # print(fourl_pos_labels)
            
            # fourl_pos_loss = F.binary_cross_entropy_with_logits(fourl_pos_logits, fourl_pos_labels, reduction='none')
            # print("POS LOSS: ", fourl_pos_loss.size())
            # fourl_pos_dt_auc = roc_auc_score(fourl_pos_labels.cpu(), fourl_pos_logits.cpu())
            # print("POS AUC: ", fourl_pos_dt_auc)
            # fourl_pos_dt_aup = average_precision_score(fourl_pos_labels.cpu(), fourl_pos_logits.cpu())
            # print("POS AUP: ", fourl_pos_dt_aup)