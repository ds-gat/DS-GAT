import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm, trange
from datetime import datetime
from models_base.rgcn import RGCN
from models_base.rgcnw import RGCNW
from models_base.gatv2 import GATv2Rel
from models_base.egat import EGATWrapper
from models_base.wsgat import WSGATWrapper
from models_base.rgat import RGAT
from models_base.dsgat2 import DSGAT2
from models_base.dsgat_a1 import DSGATA1
from models_base.dsgat_a2 import DSGATA2


from utils import (
    load_data,
    build_all_true_set,
    generate_sampled_graph_and_labels,
    generate_sampled_graph_and_labels_directed,
    build_test_graph,
    build_test_graph_directed,
    build_test_graph_directed_neighbors,
    generate_sampled_graph_and_labels_directed_neighbors,

    calc_mrr,
)
import random
import time

from torch_geometric.utils import k_hop_subgraph
import os

# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def train(train_triplets, train_confidence, model, use_cuda, batch_size,
          split_size, negative_sample, reg_ratio, num_entities, num_relations,
          weighted, directed=True, all_true_set=None, sampling=None,
          epoch=0, max_epochs=15000, use_amp=False):

    device = torch.device('cuda' if use_cuda else 'cpu')

    if directed:
        if sampling == "neighbors":
            train_data = generate_sampled_graph_and_labels_directed_neighbors(
                train_triplets, batch_size, split_size, num_entities,
                num_relations, negative_sample, train_confidence,
                all_true_set=all_true_set)
        else:
            train_data = generate_sampled_graph_and_labels_directed(
                train_triplets, batch_size, split_size, num_entities,
                num_relations, negative_sample, train_confidence,
                all_true_set=all_true_set)
    else:
        train_data = generate_sampled_graph_and_labels(
            train_triplets, batch_size, split_size, num_entities,
            num_relations, negative_sample, train_confidence)

    traindata_entity = train_data.entity.to(device)
    edge_index  = train_data.edge_index.to(device)
    edge_type   = train_data.edge_type.to(device)
    edge_norm   = train_data.edge_norm.to(device)
    edge_weight = getattr(train_data, 'edge_weight', None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    samples    = train_data.samples.to(device)
    labels     = train_data.labels.to(device)
    sample_conf = getattr(train_data, 'sample_conf', None)
    if sample_conf is not None:
        sample_conf = sample_conf.to(device)

    with autocast(enabled=use_amp):
        if weighted:
            emb = model(traindata_entity, edge_index, edge_type, edge_weight=edge_weight)
        else:
            emb = model(traindata_entity, edge_index, edge_type, edge_norm=edge_norm)

        if hasattr(model, 'score_loss_soft_v2') and sample_conf is not None:
            rel_types = samples[:, 1] if hasattr(model, 'tau_mlp') else None
            loss = model.score_loss_soft_v2(emb, samples, labels, sample_conf,
                                            relation_types=rel_types)
            loss += reg_ratio * model.reg_loss(emb) 
        else:
            loss  = model.score_loss(emb, samples, labels)
            loss += reg_ratio * model.reg_loss(emb)

        if hasattr(model, 'confidence_nll_loss') and edge_weight is not None:
            nll = model.confidence_nll_loss(edge_weight.to(emb.device))
            loss = loss + getattr(model, 'nll_lambda', 0.01) * nll

        if hasattr(model, 'confidence_kl_loss') and edge_weight is not None:
            kl = model.confidence_kl_loss(edge_weight.to(emb.device))
            loss = loss + getattr(model, 'nll_lambda', 0.01) * kl

    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def valid(valid_triplets, model, test_graph, all_triplets, use_cuda, weighted=False):
    model.eval()
    device = torch.device('cuda' if use_cuda else 'cpu')
    with torch.no_grad():
        entity      = test_graph.entity.to(device)
        edge_index  = test_graph.edge_index.to(device)
        edge_type   = test_graph.edge_type.to(device)
        edge_norm   = test_graph.edge_norm.to(device)
        edge_weight = (test_graph.edge_weight
                       if hasattr(test_graph, 'edge_weight')
                       else getattr(test_graph, 'edge_norm', None))
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        valid_triplets = valid_triplets.to(device)
        all_triplets   = all_triplets.to(device)

        if weighted:
            full_embedding = model(entity, edge_index, edge_type, edge_weight=edge_weight)
        else:
            full_embedding = model(entity, edge_index, edge_type, edge_norm=edge_norm)

        # score_func: None → DistMult unchanged; lambda → ComplEx/RotatE
        if hasattr(model, '_score_for_eval') and getattr(model, 'score_function', 'dismult') in ('complex', 'rotate'):
            score_fn = lambda h, r, t: model._score_for_eval(h, r, t)
        else:
            score_fn = None
            

        mrr, hits = calc_mrr(full_embedding, model.relation_embedding,
                             valid_triplets, all_triplets, hits=[1, 3, 5, 10],
            score_func=score_fn)
        return mrr


# ─────────────────────────────────────────────────────────────────────────────
# Confidence metrics
# ─────────────────────────────────────────────────────────────────────────────

def calc_confidence_metrics(embedding, relation_embedding, test_triplets,
                             true_conf, device, model=None):
    h = test_triplets[:, 0].to(device)
    r = test_triplets[:, 1].to(device)
    t = test_triplets[:, 2].to(device)
    
    if model is not None and hasattr(model, '_calibrated_score'):
        pred_conf = model._calibrated_score(embedding, test_triplets.to(device))
    else:
        scores    = torch.sum(embedding[h] * relation_embedding[r] * embedding[t], dim=-1)
        pred_conf = torch.sigmoid(scores)
    
    true_conf = true_conf.to(device)
    mae = torch.mean(torch.abs(pred_conf - true_conf)).item()
    mse = torch.mean((pred_conf - true_conf) ** 2).item()
    return mae, mse


# ─────────────────────────────────────────────────────────────────────────────
# calc_mrr_per_relation
# ─────────────────────────────────────────────────────────────────────────────

def calc_mrr_per_relation(entity_embedding, relation_embedding, triplets,
                          all_triplets, hits=[1, 3, 5, 10],
                          conf_weights=None,
            score_func=None):
    device      = triplets.device
    metrics_dict = {}
    relations   = triplets[:, 1].unique()

    for r in relations:
        mask  = triplets[:, 1] == r
        tlist = triplets[mask]

        # per-relation confidence weights (subset)
        rel_conf = conf_weights[mask] if conf_weights is not None else None

        mrr, hits_dict = calc_mrr(
            entity_embedding, relation_embedding,
            tlist, all_triplets,
            hits=hits,
            conf_weights=rel_conf,
            score_func=score_func)

        metrics = {"mrr": float(mrr)}
        for k in hits:
            metrics[f"hits@{k}"] = float(hits_dict[k])
        if rel_conf is not None:
            metrics["wmrr"]  = float(hits_dict.get("wmrr",  0.0))
            for k in hits:
                metrics[f"whits@{k}"] = float(hits_dict.get(f"whits@{k}", 0.0))

        metrics_dict[int(r.item())] = metrics

    return metrics_dict


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

def test(test_triplets, model, test_graph, all_triplets, use_cuda,
         weighted=False, relation_map=None, model_name="",
         output_dir="output", test_conf=None):

    model.eval()
    device = torch.device('cuda' if use_cuda else 'cpu')
    with torch.no_grad():
        entity      = test_graph.entity.to(device)
        edge_index  = test_graph.edge_index.to(device)
        edge_type   = test_graph.edge_type.to(device)
        edge_norm   = test_graph.edge_norm.to(device)
        edge_weight = (test_graph.edge_weight
                       if hasattr(test_graph, 'edge_weight')
                       else getattr(test_graph, 'edge_norm', None))
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        test_triplets = test_triplets.to(device)
        all_triplets  = all_triplets.to(device)

        if weighted:
            full_embedding = model(entity, edge_index, edge_type, edge_weight=edge_weight)
        else:
            full_embedding = model(entity, edge_index, edge_type, edge_norm=edge_norm)

        # Confidence weights for wmrr / whits
        conf_weights = None
        if test_conf is not None:
            conf_weights = test_conf.to(device)

            print(f"test_conf stats:")
            print(f"  min:    {test_conf.min():.4f}")
            print(f"  max:    {test_conf.max():.4f}")
            print(f"  mean:   {test_conf.mean():.4f}")
            print(f"  std:    {test_conf.std():.4f}")
            print(f"  median: {np.median(test_conf):.4f}")


        if hasattr(model, '_score_for_eval') and getattr(model, 'score_function', 'dismult') in ('complex', 'rotate'):
            score_fn = lambda h, r, t: model._score_for_eval(h, r, t)
        else:
            score_fn = None

        # ── Overall metrics ───────────────────────────────────────────────
        mrr, overall_hits = calc_mrr(
            full_embedding, model.relation_embedding,
            test_triplets, all_triplets,
            hits=[1, 3, 5, 10, 15],
            conf_weights=conf_weights,
            score_func=score_fn)

        # ── Per-relation metrics ──────────────────────────────────────────
        metrics_by_relation = calc_mrr_per_relation(
            full_embedding, model.relation_embedding,
            test_triplets, all_triplets,
            hits=[1, 3, 5, 10, 15],
            conf_weights=conf_weights,
            score_func=score_fn)

        # ── Confidence prediction metrics ─────────────────────────────────
        mae, mse = None, None
        if test_conf is not None:
            mae, mse = calc_confidence_metrics(
    full_embedding, model.relation_embedding,
    test_triplets, test_conf.to(device), device, model=model)
            print(f"MAE: {mae:.6f}")
            print(f"MSE: {mse:.6f}")

        # ── Save CSV ──────────────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        now      = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"metrics_{model_name}_{now}.csv")

        # Determine whether weighted columns exist
        has_weighted = conf_weights is not None

        base_cols = ["relation", "mrr",
                     "hits@1", "hits@3", "hits@5", "hits@10", "hits@15",
                     "mae", "mse"]
        w_cols    = ["wmrr",
                     "whits@1", "whits@3", "whits@5", "whits@10", "whits@15"]
        fieldnames = base_cols + (w_cols if has_weighted else [])

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)

            # Overall row
            overall_row = [
                "overall", mrr,
                overall_hits[1], overall_hits[3],
                overall_hits[5], overall_hits[10], overall_hits[15],
                mae, mse,
            ]
            if has_weighted:
                overall_row += [
                    overall_hits.get("wmrr",     ""),
                    overall_hits.get("whits@1",  ""),
                    overall_hits.get("whits@3",  ""),
                    overall_hits.get("whits@5",  ""),
                    overall_hits.get("whits@10", ""),
                    overall_hits.get("whits@15", ""),
                ]
            writer.writerow(overall_row)

            # Per-relation rows
            for rel, metrics in metrics_by_relation.items():
                rel_name = relation_map.get(rel, str(rel)) if relation_map else str(rel)
                row = [
                    rel_name,
                    metrics.get("mrr",     ""),
                    metrics.get("hits@1",  ""),
                    metrics.get("hits@3",  ""),
                    metrics.get("hits@5",  ""),
                    metrics.get("hits@10", ""),
                    metrics.get("hits@15", ""),
                    "",  # mae — only meaningful overall
                    "",  # mse
                ]
                if has_weighted:
                    row += [
                        metrics.get("wmrr",     ""),
                        metrics.get("whits@1",  ""),
                        metrics.get("whits@3",  ""),
                        metrics.get("whits@5",  ""),
                        metrics.get("whits@10", ""),
                        metrics.get("whits@15", ""),
                    ]
                writer.writerow(row)

        print(f"\nMetrics saved to: {filepath}")

    return mrr


# ─────────────────────────────────────────────────────────────────────────────
# Node embedding loader
# ─────────────────────────────────────────────────────────────────────────────

def load_node_embeddings(embedding_path):
    pretrained_emb = torch.load(embedding_path)
    num_nodes, embedding_dim = pretrained_emb.shape
    print(f"Loaded node embeddings: {num_nodes} nodes, {embedding_dim} dimensions")
    missing = [i for i in range(num_nodes) if pretrained_emb[i].sum() == 0]
    if missing:
        print(f"Warning: {len(missing)} nodes have zero embeddings.")
    return pretrained_emb


def main(args):


    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    best_mrr = 0

    entity2id, relation2id, (train_triplets, train_conf), (valid_triplets, valid_conf), (test_triplets, test_conf) = load_data(f'data/{args.dataset}')

    embedding_path = f"./data/{args.dataset}/node_initial_embeddings.pt"
    if os.path.exists(embedding_path):
        node_features = load_node_embeddings(embedding_path)
    else:
        num_entities=len(set(entity2id.values()))
        print(f"No pretrained embeddings found at {embedding_path}.")
        print(f"Initializing random embeddings: [{num_entities}, {args.embedding_dim}]")
        torch.manual_seed(args.seed)
        node_features = torch.randn(num_entities, args.embedding_dim) * 0.1
    node_features_shape = node_features.shape

    weigthed = args.weighted

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    # Directed mode (default): no reverse edges → num_relations stays as-is.
    # Bidirectional mode (--no-directed): double the relation count.
    directed = not args.no_directed
    num_relations_model = len(relation2id) if directed else len(relation2id) * 2
    print(f"[{'directed' if directed else 'bidirectional'}] num_relations_model={num_relations_model}")

    print(args)
    # device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    # node_features = node_features.to(device)

    # Initialize your RGCN
    model_name=args.modelname
    file_tag = f"{model_name}_{args.score}_{args.dataset}"
    if(model_name=="RGCN"):
        model = RGCN(
            num_entities=node_features_shape[0],
            num_relations=num_relations_model,
            num_bases=args.n_bases,
            dropout=args.dropout,
            node_features=node_features,
            embedding_dim=args.embedding_dim,
            num_layers=args.numhops
        )
    if(model_name=="GATV2"):
        model = GATv2Rel(
            num_entities=node_features_shape[0],
            num_relations=num_relations_model,
            dropout=args.dropout,
            node_features=node_features,
            embedding_dim=args.embedding_dim,
            num_layers=args.numhops
        )

    if(model_name=="RGCNW"):
        model = RGCNW(
            num_entities=node_features_shape[0],
            num_relations=num_relations_model,
            num_bases=args.n_bases,
            dropout=args.dropout,
            node_features=node_features,
            embedding_dim=args.embedding_dim,
            num_layers=args.numhops,
            edge_weight_mode=args.edge_weight_mode
        )
    if(model_name=="EGAT"):
        model = EGATWrapper(
            node_features=node_features,
            num_relations=num_relations_model,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            num_layers=args.numhops
        )
    if(model_name=="WSGAT"):
        model = WSGATWrapper(
            node_features=node_features,
            embedding_dim=args.embedding_dim,
            num_relations=num_relations_model,
            dropout=args.dropout,
            num_layers=args.numhops
        )
    if(model_name=="RGAT"):
        model = RGAT(
            num_entities=node_features_shape[0],
            num_relations=num_relations_model,
            num_bases=args.n_bases,
            dropout=args.dropout,
            node_features=node_features,
            embedding_dim=args.embedding_dim,
            num_layers=args.numhops
        )
    if(model_name=="GATV2W"):
        model = GATv2Rel(
            num_entities=node_features_shape[0],
            num_relations=num_relations_model,
            dropout=args.dropout,
            node_features=node_features,
            embedding_dim=args.embedding_dim,
            num_layers=args.numhops,
            weights=True
        )

    if(model_name=="EGATR"):
        model = EGATWrapper(
            node_features=node_features,
            num_relations=num_relations_model,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            num_layers=args.numhops,
            use_bayesian=True

        )


   
    if(model_name=="DSGAT2"):
        model = DSGAT2(
                node_features=node_features,
                num_relations=num_relations_model,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
                num_layers=args.numhops,
                score_function=args.score
            )
        model.nll_lambda = args.nll_lambda
   
    _ablation_classes = {
        "DSGATA1": DSGATA1, "DSGATA2": DSGATA2, 
    }
    if model_name in _ablation_classes:
        model = _ablation_classes[model_name](
                node_features=node_features,
                num_relations=num_relations_model,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
                num_layers=args.numhops,
                score_function=args.score
            )
        model.nll_lambda = args.nll_lambda
    

    print(f"train_conf is None: {train_conf is None}")
    if train_conf is not None:
        print(f"Confidence stats: min={train_conf.min()}, max={train_conf.max()}, mean={train_conf.mean()}")
    else:
        print("No confidence scores found in dataset - using uniform weights")
    
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))

    sampling=args.sampling

    # Build all_true_set for filtered negative sampling (directed mode)
    all_true_set = None
    if directed:
        all_true_set = build_all_true_set(train_triplets, valid_triplets, test_triplets)
        print(f"[directed] all_true_set size: {len(all_true_set)}")
        
        
        if(sampling=="neighbors"):
            test_graph = build_test_graph_directed_neighbors(
            len(set(entity2id.values())), len(relation2id), train_triplets, train_conf,
            max_edges=args.test_graph_size)
        else:
            test_graph = build_test_graph_directed(
            len(set(entity2id.values())), len(relation2id), train_triplets, train_conf,
            max_edges=args.test_graph_size)
    else:
        test_graph = build_test_graph(
            len(set(entity2id.values())), len(relation2id), train_triplets, train_conf,
            max_edges=args.test_graph_size, relation2id=relation2id)

    valid_triplets = torch.LongTensor(valid_triplets)
    test_conf_tensor = torch.FloatTensor(test_conf) if test_conf is not None else None
    test_triplets = torch.LongTensor(test_triplets)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay_factor,
        patience=args.lr_patience, min_lr=args.min_lr, verbose=True)

    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    if use_cuda:
        model.cuda()

    training_log = []
    train_start_total = time.time()
    es_counter = 0   # early-stopping: eval intervals with no MRR improvement > min_delta

    # AMP: mixed precision activo solo en GPU (en CPU es no-op)
    use_amp = use_cuda
    scaler  = GradScaler(enabled=use_amp)

    

    for epoch in trange(1, (args.n_epochs + 1), desc='Epochs', position=0):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        loss = train(train_triplets, train_conf, model, use_cuda,
                     batch_size=args.graph_batch_size,
                     split_size=args.graph_split_size,
                     negative_sample=args.negative_sample,
                     reg_ratio=args.regularization,
                     num_entities=len(set(entity2id.values())),
                     num_relations=num_relations_model,
                     weighted=weigthed,
                     directed=directed,
                     all_true_set=all_true_set, sampling=sampling,
                     epoch=epoch, max_epochs=args.n_epochs,
                     use_amp=use_amp)

        if torch.isnan(loss) or torch.isinf(loss):
            print("Loss exploded:", loss.item())
            exit()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                              # necesario antes del clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_train_time = time.time() - t0

        # LR scheduler: step every epoch on train loss
        
        current_lr = optimizer.param_groups[0]['lr']

        val_mrr_log = ''
        if epoch % args.evaluate_every == 0:
            tqdm.write(f"Train Loss {loss.item():.6f}  LR {current_lr:.2e}  Epoch {epoch}")

            if use_cuda:
                torch.cuda.empty_cache()

            model.eval()
            torch.cuda.empty_cache()
            valid_mrr = valid(valid_triplets, model, test_graph, all_triplets, use_cuda, weighted=weigthed)
            val_mrr_log = valid_mrr

            scheduler.step(1.0 - valid_mrr)

            if valid_mrr > best_mrr + args.early_stop_delta:
                best_mrr   = valid_mrr
                es_counter = 0
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           f"./output/{file_tag}t_best_mrr_model.pth")
            else:
                es_counter += 1
                tqdm.write(f"  No MRR improvement ({es_counter}/{args.early_stop_patience}) "
                           f"best={best_mrr:.4f}")
                if es_counter >= args.early_stop_patience:
                    tqdm.write(f"Early stopping at epoch {epoch} "
                               f"(no improvement for {args.early_stop_patience} evals)")
                    training_log.append({
                        'epoch':              epoch,
                        'train_loss':         round(loss.item(), 6),
                        'val_mrr':            val_mrr_log,
                        'epoch_train_time_s': round(epoch_train_time, 3),
                    })
                    break
        else:
            scheduler.step(loss.item())
        training_log.append({
            'epoch':              epoch,
            'train_loss':         round(loss.item(), 6),
            'val_mrr':            val_mrr_log,
            'epoch_train_time_s': round(epoch_train_time, 3),
        })

    total_train_time = time.time() - train_start_total
    print(f"Total training time: {total_train_time:.1f}s ({total_train_time/3600:.2f}h)")

    os.makedirs("./output", exist_ok=True)
    curves_path = f"./output/training_curves_{file_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(curves_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_mrr', 'epoch_train_time_s'])
        writer.writeheader()
        writer.writerows(training_log)
    print(f"Training curves saved to: {curves_path}")

    model.eval()
    checkpoint = torch.load(f"./output/{file_tag}t_best_mrr_model.pth")

    model.load_state_dict(checkpoint['state_dict'])

    if use_cuda:
        model.cuda()

    id2relation = {v: k for k, v in relation2id.items()}
    test(test_triplets, model, test_graph, all_triplets, use_cuda, weighted=weigthed,
         relation_map=id2relation, model_name=file_tag,
     test_conf=test_conf_tensor)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execution Models')

    parser.add_argument("--graph-batch-size", type=int, default=30000)
    parser.add_argument("--graph-split-size", type=float, default=0.5)
    parser.add_argument("--negative-sample", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=3000)
    parser.add_argument("--evaluate-every", type=int, default=500)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5,
                        help="Factor to reduce LR on plateau (ReduceLROnPlateau).")
    parser.add_argument("--lr-patience", type=int, default=500,
                        help="Epochs with no loss improvement before reducing LR.")
    parser.add_argument("--min-lr", type=float, default=1e-5,
                        help="Minimum LR for ReduceLROnPlateau.")
    parser.add_argument("--early-stop-patience", type=int, default=20,
                        help="Eval intervals (each = evaluate-every epochs) with no MRR "
                             "improvement before stopping. Default 20 × 500 = 10k epochs.")
    parser.add_argument("--early-stop-delta", type=float, default=0.001,
                        help="Minimum MRR improvement to count as progress.")
    parser.add_argument("--n-bases", type=int, default=4)

    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--nll-lambda", type=float, default=0.1)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--test-graph-size", type=int, default=-1,
                        help="Maximum number of training triplets to use for test graph (to avoid OOM). Set to -1 to use all triplets (default for CN15k).")

    # NEW argument for edge_weight_mode
    parser.add_argument("--edge-weight-mode", type=str, default="bayesian",
                        choices=["normalize", "concat", "none", "learnable", "bayesian"],
                        help="Edge weight usage mode: 'normalize' for normalized weights, 'concat' to concatenate edge weights to features, 'none' for raw unnormalized multipliers, 'learnable' for MLP-transformed weights, 'bayesian' for uncertainty-aware Bayesian edge weights (default).")

    parser.add_argument("--modelname", type=str, default="RGCN",
                        choices=["RGCN", "GATV2","RGCNW", "EGAT", "WSGAT", "RGAT","GATV2W", "EGATR" ,
                       "DSGAT2", "DSGATA1", "DSGATA2"])
    
    parser.add_argument("--sampling", type=str, default="uniform",
                        choices=["uniform", "neighbors"])

    parser.add_argument("--score", type=str, default="dismult",
                        choices=["dismult", "complex","rotate", "transe", "hybrid"])

    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--no-directed", action="store_true", default=False,
                        help="Revert to bidirectional graph with unfiltered negatives (original mode). "
                             "By default directed graph + filtered negatives are used.")
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--numhops", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="ontoomics",
                        help="Dataset folder name under data/ (default: ontoomics)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")


    
    

    
    args = parser.parse_args()


    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.test_graph_size == -1:
        args.test_graph_size = None
        
    print(args)

    main(args)

