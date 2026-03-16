import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_scatter import scatter_add

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def read_triplets_numeric(file_path):
    """Read triplets where entities/relations are already numeric IDs.
    
    Returns:
        triplets: numpy array of shape (n, 3) with columns [head, relation, tail]
        confidence: numpy array of shape (n,) with confidence scores, or None if not present
    """
    triplets = []
    confidences = []
    has_confidence = False

    with open(file_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            head, relation, tail = int(parts[0]), int(parts[1]), int(parts[2])
            triplets.append((head, relation, tail))
            
            # Check if 4th column (confidence score) exists
            if len(parts) >= 4:
                has_confidence = True
                confidences.append(float(parts[3]))

    if has_confidence:
        return np.array(triplets), np.array(confidences)
    else:
        return np.array(triplets), None

def load_data(file_path):
    '''
        argument:
            file_path: ./data/cn15k 
        
        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
            (and optionally confidence scores for each split)
    '''

    print("load data from {}".format(file_path))

    # Check if CN15k format (CSV files) or FB15k-237 format (dict files)
    entity_csv = os.path.join(file_path, 'entity_id.csv')
    relation_csv = os.path.join(file_path, 'relation_id.csv')
    entity_dict = os.path.join(file_path, 'entities.dict')
    relation_dict = os.path.join(file_path, 'relations.dict')



    entity_dict_2 = os.path.join(file_path, 'edges.new.tsv.entities.tsv')
    relation_dict_2 = os.path.join(file_path, 'edges.new.tsv.relations.tsv')

    if os.path.exists(entity_csv) and os.path.exists(relation_csv):
        # CSV format (CN15k, ppi5k, etc.)
        import csv
        entity2id = {}
        with open(entity_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                entity, eid = row[0], int(row[1])
                entity2id[entity] = eid
                # ppi5k TSVs write "taxid_protein" but entity_id.csv stores "taxid.protein"
                # Register both forms so either separator works in triplet files.
                alt = entity.replace('.', '_', 1)
                if alt != entity:
                    entity2id[alt] = eid

        relation2id = {}
        with open(relation_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Read header to detect column order
            # ppi5k:  "id,rel string"       → id in col 0, name in col 1
            # CN15k:  "relation string,id"  → name in col 0, id in col 1
            id_col   = 0 if header[0].strip().lower() in ('id', 'rid') else 1
            name_col = 1 - id_col
            for row in reader:
                relation = row[name_col]
                rid      = int(row[id_col])
                relation2id[relation] = rid

        # Detect whether triplet TSVs use numeric IDs or string names
        _train_tsv = os.path.join(file_path, 'train.tsv')
        with open(_train_tsv) as _f:
            _first = _f.readline().strip().split('\t')
        try:
            int(_first[1])
            _triplets_numeric = True
        except ValueError:
            _triplets_numeric = False

        if _triplets_numeric:
            train_triplets, train_conf = read_triplets_numeric(os.path.join(file_path, 'train.tsv'))
            valid_triplets, valid_conf = read_triplets_numeric(os.path.join(file_path, 'val.tsv'))
            test_triplets,  test_conf  = read_triplets_numeric(os.path.join(file_path, 'test.tsv'))
        else:
            # ppi5k: triplets use string entity/relation names → map via dictionaries
            train_triplets, train_conf = read_triplets(os.path.join(file_path, 'train.tsv'), entity2id, relation2id)
            valid_triplets, valid_conf = read_triplets(os.path.join(file_path, 'val.tsv'),   entity2id, relation2id)
            test_triplets,  test_conf  = read_triplets(os.path.join(file_path, 'test.tsv'),  entity2id, relation2id)
        
    elif os.path.exists(entity_dict) and os.path.exists(relation_dict):
        # FB15k-237 format: dict files
        with open(entity_dict) as f:
            entity2id = dict()
            for line in f:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(relation_dict) as f:
            relation2id = dict()
            for line in f:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        train_triplets, train_conf = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
        valid_triplets, valid_conf = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
        test_triplets, test_conf = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)

    elif os.path.exists(entity_dict_2) and os.path.exists(relation_dict_2):
        # Ontoomics

        import csv
        entity2id = {}
        with open(entity_dict_2, 'r') as f:
            reader = csv.reader(f)
            for new_id, r in enumerate(reader):
                row = str(r[0]).strip().split('\t')
                _, entity = int(row[0]), row[1]
                entity2id[entity] = new_id

        
        relation2id = {}
        with open(relation_dict_2, 'r') as f:
            reader = csv.reader(f)
            for new_id, r in enumerate(reader):
                row = str(r[0]).strip().split('\t')
                _, relation = int(row[0]), row[1]
                relation2id[relation] = new_id

        


        
        # CN15k uses .tsv files and numeric IDs directly in triplets
        train_triplets, train_conf = read_triplets_numeric(os.path.join(file_path, 'edges_train_new.tsv'))
        valid_triplets, valid_conf = read_triplets_numeric(os.path.join(file_path, 'edges_val_new.tsv'))
        test_triplets, test_conf = read_triplets_numeric(os.path.join(file_path, 'edges_test_new.tsv'))


        print("Max relation id:", train_triplets[:,1].max())
        print("Num relations:", len(relation2id))


    
    else:
        raise FileNotFoundError("Could not find entity/relation mapping files. Expected either entity_id.csv/relation_id.csv (CN15k) or entities.dict/relations.dict (FB15k-237)")

    print('num_entity: {}'.format(len(set(entity2id.values()))))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))
    
    # Store confidence scores as attributes (for potential future use)
    if train_conf is not None:
        print('Loaded confidence scores for triplets')

    return entity2id, relation2id, (train_triplets, train_conf), (valid_triplets, valid_conf), (test_triplets, test_conf)

def read_triplets(file_path, entity2id, relation2id):
    """Read triplets where entities/relations are strings that need mapping.
    
    Returns:
        triplets: numpy array of shape (n, 3) with columns [head, relation, tail]
        confidence: numpy array of shape (n,) with confidence scores, or None if not present
    """
    triplets = []
    confidences = []
    has_confidence = False

    with open(file_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            head, relation, tail = parts[0], parts[1], parts[2]
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))
            
            # Check if 4th column (confidence score) exists
            if len(parts) >= 4:
                has_confidence = True
                confidences.append(float(parts[3]))

    if has_confidence:
        return np.array(triplets), np.array(confidences)
    else:
        return np.array(triplets), None

def sample_edge_uniform(n_triples, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace=False)

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick (degree-based)
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_entity, num_rels, negative_rate, confidence_scores=None ):
    """
        Get training graph and signals
        First perform edge neighborhood sampling on graph, then perform negative
        sampling to generate negative samples
        
        Args:
            triplets: Training triplets (numpy array of shape [N, 3])
            sample_size: Number of edges to sample
            split_size: Fraction of sampled edges to use as graph structure
            num_entity: Total number of entities
            num_rels: Number of relation types
            negative_rate: Number of negative samples per positive
            confidence_scores: Optional confidence scores for each triplet (1D numpy array of length N)
    """

    edges = sample_edge_uniform(len(triplets), sample_size)

    # Select sampled edges
    edges_sampled = triplets[edges]  # Shape: [sample_size, 3]
    src, rel, dst = edges_sampled.transpose()
    
    # Sample corresponding confidence scores if available
    sampled_confidence = None
    if confidence_scores is not None:
        sampled_confidence = confidence_scores[edges]  # Shape: [sample_size] - 1D array
    
    uniq_entity, edges_relabeled = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges_relabeled, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype = torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype = torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype = torch.long).contiguous()
    
    # Get confidence scores for graph edges (should be 1D array)
    if sampled_confidence is not None:
        graph_confidence = torch.tensor(sampled_confidence[graph_split_ids], dtype=torch.float).contiguous()
    else:
        graph_confidence = None

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    #Duplicate confidence scores for bidirectional edges
    if graph_confidence is not None:
        graph_confidence = torch.cat((graph_confidence, graph_confidence))
    else:
        # Create uniform weights if no confidence provided
        graph_confidence = torch.ones(src.size(0), dtype=torch.float)

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    
    # Always set edge_weight (use confidence if available, otherwise uniform)
    data.edge_weight = graph_confidence
        
    # Keep edge_norm for backward compatibility
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    # Per-sample confidence for soft-label loss (used by EGATPROP10 and variants).
    # Positives get their original edge confidence; negatives get conf=1.0 so that
    # soft_target = label * conf stays 0.0 for negatives (no change).
    if sampled_confidence is not None:
        pos_conf = sampled_confidence.astype(np.float32)        # [sample_size]
        num_neg = len(samples) - len(relabeled_edges)           # sample_size * negative_rate
        neg_conf = np.ones(num_neg, dtype=np.float32)
        data.sample_conf = torch.from_numpy(
            np.concatenate([pos_conf, neg_conf])
        )
    else:
        data.sample_conf = torch.ones(len(samples), dtype=torch.float)

    return data

# ---------------------------------------------------------------------------
# Directed graph utilities (no reverse edges) for fair comparison with
# FocusE / UPGAT reference models.
# ---------------------------------------------------------------------------

def build_all_true_set(train_triplets, valid_triplets, test_triplets):
    """Build a frozenset of all known (h, r, t) triples for filtered negative sampling.

    Uses global entity IDs as in the FocusE/UPGAT reference.
    """
    all_true = set()
    for arr in [train_triplets, valid_triplets, test_triplets]:
        for row in arr:
            all_true.add((int(row[0]), int(row[1]), int(row[2])))
    return all_true


def negative_sampling_filtered(pos_samples, num_entity, negative_rate, all_true_set):
    """Filtered negative sampling — vectorized for speed.

    Generates all negatives in one numpy batch (like negative_sampling), then
    does a single O(n) pass to replace any that land in all_true_set.
    Contamination rate on sparse graphs (ppi5k: ~40k / 175M possible triples)
    is ~0.02%, so one replacement pass is equivalent to full filtering.

    Args:
        pos_samples:   numpy array [batch, 3] with global (h, r, t) IDs.
        num_entity:    total number of entities (global).
        negative_rate: number of negatives per positive.
        all_true_set:  set of known true (h, r, t) tuples (or None to skip filtering).

    Returns:
        samples: numpy array [batch*(1+negative_rate), 3]
        labels:  numpy float32 array [batch*(1+negative_rate)]
    """
    size_of_batch  = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate

    # Vectorized generation — same as negative_sampling()
    neg_samples = np.tile(pos_samples[:, :3], (negative_rate, 1))
    values      = np.random.choice(num_entity, size=num_to_generate)
    choices     = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj  = ~subj
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj,  2] = values[obj]

    # One-pass filter: find contaminated negatives and replace each once
    if all_true_set:
        bad = np.array([(int(r[0]), int(r[1]), int(r[2])) in all_true_set
                        for r in neg_samples], dtype=bool)
        if bad.any():
            n_bad     = int(bad.sum())
            new_vals  = np.random.choice(num_entity, size=n_bad)
            new_side  = np.random.uniform(size=n_bad) > 0.5
            rep       = neg_samples[bad].copy()
            rep[new_side,  0] = new_vals[new_side]
            rep[~new_side, 2] = new_vals[~new_side]
            neg_samples[bad] = rep

    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[:size_of_batch] = 1.0
    return np.concatenate((pos_samples[:, :3], neg_samples)), labels


def edge_normalization_directed(edge_type, edge_index, num_entity, num_relation):
    """Degree-based edge normalization for directed graphs (no reverse edges).

    Identical logic to edge_normalization() but uses num_relation instead of
    2*num_relation, because there are no added reverse relation types.
    """
    one_hot = F.one_hot(edge_type, num_classes=num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * num_relation
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]
    return edge_norm

def generate_sampled_graph_and_labels_directed(
        triplets, sample_size, split_size, num_entity, num_rels,
        negative_rate, confidence_scores=None, all_true_set=None):
    """Directed version of generate_sampled_graph_and_labels.

    If sample_size == -1: full graph mode (EGAT-style, no supervision split).
      - ALL edges go into the graph for message passing.
      - Random 8192 edges used for supervision each epoch (mini-batch loss).
      - Supervision edges ARE in the graph — matches standard EGAT/GATv2 protocol.

    If sample_size > 0: subgraph sampling mode (original behaviour).
      - sample_size edges sampled uniformly.
      - split_size fraction used for graph, remainder for supervision.
    """

    # ── Edge selection ────────────────────────────────────────────────────
    full_graph = (sample_size == -1)
    if full_graph:
        edges = np.arange(len(triplets))
    else:
        edges = sample_edge_uniform(len(triplets), sample_size)

    edges_sampled = triplets[edges]               # [n_edges, 3]
    src, rel, dst = edges_sampled.transpose()     # each [n_edges]

    sampled_confidence = None
    if confidence_scores is not None:
        sampled_confidence = confidence_scores[edges]   # [n_edges]

    # ── Graph / supervision split ─────────────────────────────────────────
    if full_graph:
        # ALL edges in graph — no exclusion, matches standard protocol
        graph_split_ids = np.arange(len(edges_sampled))

        # random mini-batch for loss — keeps training tractable
        n_supervision = min(len(edges_sampled), 8192)
        pos_split_ids = np.random.choice(
            np.arange(len(edges_sampled)),
            size=n_supervision,
            replace=False)
    else:
        split_size_n    = int(len(edges) * split_size)
        graph_split_ids = np.random.choice(
            np.arange(len(edges)),
            size=split_size_n,
            replace=False)
        pos_split_ids   = np.setdiff1d(
            np.arange(len(edges)), graph_split_ids)

    pos_samples = edges_sampled[pos_split_ids]    # [n_pos, 3]

    # ── Negative sampling ─────────────────────────────────────────────────
    if all_true_set is not None:
        samples, labels = negative_sampling_filtered(
            pos_samples[:, :3], num_entity, negative_rate, all_true_set)
    else:
        samples, labels = negative_sampling(
            pos_samples[:, :3], num_entity, negative_rate)

    # ── Build directed graph (full graph — all edges) ─────────────────────
    src_g = torch.tensor(src[graph_split_ids], dtype=torch.long).contiguous()
    dst_g = torch.tensor(dst[graph_split_ids], dtype=torch.long).contiguous()
    rel_g = torch.tensor(rel[graph_split_ids], dtype=torch.long).contiguous()

    if sampled_confidence is not None:
        graph_confidence = torch.tensor(
            sampled_confidence[graph_split_ids],
            dtype=torch.float).contiguous()
    else:
        graph_confidence = torch.ones(
            len(graph_split_ids), dtype=torch.float)

    edge_index = torch.stack((src_g, dst_g))
    edge_type  = rel_g

    data = Data(edge_index=edge_index)
    data.entity      = torch.arange(num_entity)
    data.edge_type   = edge_type
    data.edge_weight = graph_confidence
    data.edge_norm   = edge_normalization_directed(
        edge_type, edge_index, num_entity, num_rels)
    data.samples     = torch.from_numpy(samples)
    data.labels      = torch.from_numpy(labels)

    # ── Per-sample confidence for loss ────────────────────────────────────
    if sampled_confidence is not None:
        pos_conf = sampled_confidence[pos_split_ids].astype(np.float32)
        num_neg  = len(samples) - len(pos_samples)
        neg_conf = np.ones(num_neg, dtype=np.float32)
        data.sample_conf = torch.from_numpy(
            np.concatenate([pos_conf, neg_conf]))
    else:
        data.sample_conf = torch.ones(len(samples), dtype=torch.float)

    return data

def build_test_graph_directed(num_nodes, num_rels, triplets, confidence_scores=None, max_edges=None):
    """Directed version of build_test_graph.

    Does NOT add reverse edges — relations stay in [0, num_rels).
    Used for evaluation when training with generate_sampled_graph_and_labels_directed.
    """
    if max_edges is not None and len(triplets) > max_edges:
        indices = np.random.choice(len(triplets), size=max_edges, replace=False)
        triplets = triplets[indices]
        if confidence_scores is not None:
            confidence_scores = confidence_scores[indices]

    src, rel, dst = triplets.transpose()

    src = torch.from_numpy(src).long()
    dst = torch.from_numpy(dst).long()
    rel = torch.from_numpy(rel).long()

    rel = torch.clamp(rel, 0, num_rels - 1)

    # Handle confidence scores
    if confidence_scores is not None:
        graph_confidence = torch.from_numpy(confidence_scores).float()
        graph_confidence = torch.clamp(graph_confidence, min=0.0)
        #graph_confidence = graph_confidence / (graph_confidence.max() + 1e-8)
    else:
        graph_confidence = torch.ones(src.size(0), dtype=torch.float)

    # NO bidirectional edges
    edge_index = torch.stack((src, dst))
    edge_type  = rel

    assert edge_type.max() < num_rels, f"edge_type too large: {edge_type.max()}"
    assert edge_type.min() >= 0,       f"edge_type negative: {edge_type.min()}"

    data = Data(edge_index=edge_index)
    data.entity      = torch.arange(num_nodes)
    data.edge_type   = edge_type
    data.edge_weight = graph_confidence
    data.edge_norm   = edge_normalization_directed(edge_type, edge_index, num_nodes, num_rels)

    return data



def generate_sampled_graph_and_labels_directed_neighbors(
        triplets, sample_size, split_size, num_entity, num_rels,
        negative_rate, confidence_scores=None, all_true_set=None,
        num_neighbors=[10, 5]):
    """
    Directed version with neighbor sampling instead of uniform edge sampling.
    Returns same structure as original generate_sampled_graph_and_labels_directed.
    """

    import numpy as np
    import torch
    from torch_geometric.data import Data

    # ---------------------------
    # Build adjacency list
    # ---------------------------
    adj = {i: [] for i in range(num_entity)}
    for h, r, t in triplets:
        adj[h].append((t, r))

    # ---------------------------
    # Sample starting nodes
    # ---------------------------
    all_nodes = np.unique(triplets[:, [0, 2]])
    sample_sized = min(sample_size, len(all_nodes))
    start_nodes = np.random.choice(all_nodes, sample_sized, replace=False)

    layer_nodes = list(start_nodes)

    all_edge_src = []
    all_edge_dst = []
    all_edge_type = []

    # ---------------------------
    # Neighbor sampling (multi-layer)
    # ---------------------------
    for n_sample in num_neighbors:
        next_layer_nodes = []

        for node in layer_nodes:
            nbrs = adj.get(node, [])

            if len(nbrs) > n_sample:
                idxs = np.random.choice(len(nbrs), n_sample, replace=False)
                nbrs = [nbrs[i] for i in idxs]

            for nbr, rel in nbrs:
                all_edge_src.append(node)
                all_edge_dst.append(nbr)
                all_edge_type.append(rel)
                next_layer_nodes.append(nbr)

        layer_nodes = list(set(next_layer_nodes))

    # Convert to tensors
    src_g = torch.tensor(all_edge_src, dtype=torch.long)
    dst_g = torch.tensor(all_edge_dst, dtype=torch.long)
    rel_g = torch.tensor(all_edge_type, dtype=torch.long)

    num_edges = len(src_g)

    if num_edges == 0:
        raise RuntimeError("Neighbor sampling produced zero edges.")

    # ---------------------------
    # Split: graph edges vs supervision edges
    # ---------------------------
    split_size_n = int(num_edges * split_size)

    graph_split_ids = np.random.choice(
        np.arange(num_edges),
        size=split_size_n,
        replace=False
    )

    pos_split_ids = np.setdiff1d(np.arange(num_edges), graph_split_ids)

    # ---------------------------
    # Positive samples (for loss)
    # ---------------------------
    pos_samples = np.stack([
        src_g[pos_split_ids].numpy(),
        rel_g[pos_split_ids].numpy(),
        dst_g[pos_split_ids].numpy()
    ], axis=1)

    # ---------------------------
    # Negative sampling
    # ---------------------------
    if all_true_set is not None:
        samples, labels = negative_sampling_filtered(
            pos_samples, num_entity, negative_rate, all_true_set
        )
    else:
        samples, labels = negative_sampling(
            pos_samples, num_entity, negative_rate
        )

    # ---------------------------
    # Graph edges (message passing)
    # ---------------------------
    edge_index = torch.stack((
        src_g[graph_split_ids],
        dst_g[graph_split_ids]
    ))

    edge_type = torch.clamp(
        rel_g[graph_split_ids],
        0,
        num_rels - 1
    )

    # ---------------------------
    # Graph confidence (ONLY graph edges)
    # ---------------------------
    if confidence_scores is not None:
        graph_confidence = torch.tensor(
            confidence_scores[graph_split_ids],
            dtype=torch.float
        )
        #graph_confidence = graph_confidence / (graph_confidence.max() + 1e-8)
    else:
        graph_confidence = torch.ones(len(graph_split_ids), dtype=torch.float)

    # ---------------------------
    # Build Data object
    # ---------------------------
    data = Data(edge_index=edge_index)
    data.entity = torch.arange(num_entity)
    data.edge_type = edge_type
    data.edge_weight = graph_confidence
    data.edge_norm = edge_normalization_directed(
        edge_type,
        edge_index,
        num_entity,
        num_rels
    )
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    # ---------------------------
    # Per-sample confidence (FIXED CORRECTLY)
    # ---------------------------
    if confidence_scores is not None:
        pos_conf = confidence_scores[pos_split_ids].astype(np.float32)
    else:
        pos_conf = np.ones(len(pos_samples), dtype=np.float32)

    num_neg = len(samples) - len(pos_samples)
    neg_conf = np.ones(num_neg, dtype=np.float32)

    data.sample_conf = torch.from_numpy(
        np.concatenate([pos_conf, neg_conf])
    )

    return data

def build_test_graph_directed_neighbors(num_nodes, num_rels, triplets, confidence_scores=None, max_edges=None, num_neighbors=[10,5]):
    """
    Directed test graph using neighbor sampling instead of full edge list.
    Inputs: 
        num_nodes: number of entities
        num_rels: number of relations
        triplets: numpy array [num_triplets, 3]
        confidence_scores: optional per-triplet confidence
        max_edges: maximum number of starting edges to sample
        num_neighbors: list of number of neighbors per layer
    Returns: torch_geometric Data object
    """

    import numpy as np
    import torch
    from torch_geometric.data import Data

    # --- Limit triplets if max_edges ---
    if max_edges is not None and len(triplets) > max_edges:
        indices = np.random.choice(len(triplets), size=max_edges, replace=False)
        triplets = triplets[indices]
        if confidence_scores is not None:
            confidence_scores = confidence_scores[indices]

    # --- Build adjacency list ---
    adj = {i: [] for i in range(num_nodes)}
    for h, r, t in triplets:
        adj[h].append((t, r))

    # --- Sample starting nodes ---
    all_nodes = np.unique(triplets[:, [0,2]])
    sample_size = min(len(all_nodes), max_edges) if max_edges else len(all_nodes)
    start_nodes = np.random.choice(all_nodes, sample_size, replace=False)

    layer_nodes = list(start_nodes)
    nodes_sampled = set(layer_nodes)
    all_edge_src, all_edge_dst, all_edge_type = [], [], []

    # --- Neighbor sampling per layer ---
    for n_sample in num_neighbors:
        next_layer_nodes = []
        for node in layer_nodes:
            nbrs = adj.get(node, [])
            if len(nbrs) > n_sample:
                idxs = np.random.choice(len(nbrs), n_sample, replace=False)
                nbrs = [nbrs[i] for i in idxs]
            for nbr, rel in nbrs:
                all_edge_src.append(node)
                all_edge_dst.append(nbr)
                all_edge_type.append(rel)
                next_layer_nodes.append(nbr)
        layer_nodes = list(set(next_layer_nodes))
        nodes_sampled.update(layer_nodes)

    # --- Convert to tensors ---
    src = torch.tensor(all_edge_src, dtype=torch.long)
    dst = torch.tensor(all_edge_dst, dtype=torch.long)
    rel = torch.tensor(all_edge_type, dtype=torch.long)
    rel = torch.clamp(rel, 0, num_rels-1)

    edge_index = torch.stack((src, dst))
    edge_type = rel

    # --- Confidence weights ---
    if confidence_scores is not None:
        sampled_confidence = np.array(confidence_scores)
        if len(sampled_confidence) < len(src):
            # replicate or clip if necessary
            sampled_confidence = np.pad(sampled_confidence, (0, len(src) - len(sampled_confidence)), 'constant', constant_values=1.0)
        graph_confidence = torch.tensor(sampled_confidence[:len(src)], dtype=torch.float)
        #graph_confidence = graph_confidence / (graph_confidence.max() + 1e-8)
    else:
        graph_confidence = torch.ones(len(src), dtype=torch.float)

    # --- Build Data object ---
    data = Data(edge_index=edge_index)
    data.entity = torch.arange(num_nodes)
    data.edge_type = edge_type
    data.edge_weight = graph_confidence
    data.edge_norm = edge_normalization_directed(edge_type, edge_index, num_nodes, num_rels)

    return data
# ---------------------------------------------------------------------------

def build_test_graph(num_nodes, num_rels, triplets, confidence_scores=None, max_edges=None, relation2id=None):
    """
    Build test graph from triplets.
    Remaps edge_type to [0, num_rels*2-1] to avoid CUDA index errors.
    """
    if max_edges is not None and len(triplets) > max_edges:
        indices = np.random.choice(len(triplets), size=max_edges, replace=False)
        triplets = triplets[indices]
        if confidence_scores is not None:
            confidence_scores = confidence_scores[indices]

    src, rel, dst = triplets.transpose()

    # Convert to torch tensors
    src = torch.from_numpy(src).long()
    dst = torch.from_numpy(dst).long()
    rel = torch.from_numpy(rel).long()

    # Clamp original relation IDs to valid range
    rel = torch.clamp(rel, 0, num_rels - 1)

    # Duplicate for bidirectional edges safely
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel_rev = torch.clamp(rel + num_rels, 0, 2*num_rels - 1)  # Safety clamp
    rel = torch.cat((rel, rel_rev))

    # Handle confidence scores
    if confidence_scores is not None:
        graph_confidence = torch.from_numpy(confidence_scores).float()

        # CRITICAL FIX: stabilize weights
        graph_confidence = torch.clamp(graph_confidence, min=0.0)
        #graph_confidence = graph_confidence / (graph_confidence.max() + 1e-8)

        graph_confidence = torch.cat((graph_confidence, graph_confidence))
    else:
        graph_confidence = torch.ones(src.size(0), dtype=torch.float)


    # Build edge_index and edge_type
    edge_index = torch.stack((src, dst))
    edge_type = rel

    # Safety check: no NaNs or huge indices
    assert edge_type.max() < 2*num_rels, f"edge_type too large: {edge_type.max()}"
    assert edge_type.min() >= 0, f"edge_type negative: {edge_type.min()}"

    data = Data(edge_index=edge_index)
    data.entity = torch.arange(num_nodes)
    data.edge_type = edge_type
    data.edge_weight = graph_confidence
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)

    return data



def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

# ── Drop-in replacement for calc_mrr in utils.py ────────────────────────────
# Add conf_weights parameter — everything else is identical to the original.
# wmrr  = Σ c_i * (1/rank_i) / Σ c_i
# whits@k = Σ c_i * 1[rank_i <= k] / Σ c_i
# ── Drop-in replacement for calc_mrr in utils.py ─────────────────────────────
# Changes vs original:
#   1. New parameter: score_func=None
#   2. Scoring block replaced with if/else (DistMult path unchanged)
# Everything else: byte-for-byte identical to original.
# ─────────────────────────────────────────────────────────────────────────────

def calc_mrr(embedding, w, test_triplets, all_triplets, hits=[],
             conf_weights=None, score_func=None):
    """Filtered MRR/Hits@k with optional ComplEx/RotatE scoring.

    Args:
        embedding:     [N, dim]   entity embeddings (GNN output, eval mode)
        w:             [R, dim]   relation_embedding — used only for DistMult
        test_triplets: [n, 3]     LongTensor
        all_triplets:  [m, 3]     LongTensor — for filtered eval
        hits:          list[int]  e.g. [1, 3, 5, 10]
        conf_weights:  [n] float  optional per-triplet confidence for wMRR
        score_func:    callable or None
            If None → DistMult (original behaviour, backward compatible).
            If provided → called as score_func(h_emb, r_idx, t_embs):
                h_emb:  [dim]          subject embedding
                r_idx:  scalar tensor  relation index
                t_embs: [n_cand, dim]  candidate tail embeddings
                returns [n_cand] scores (higher = more likely)

    How to pass score_func from models.py (add before calc_mrr call):
        if model.score_function in ("complex", "rotate"):
            score_fn = lambda h, r, t: model._score_for_eval(h, r, t)
        else:
            score_fn = None
        mrr, hits_res = calc_mrr(..., score_func=score_fn)
    """
    with torch.no_grad():

        num_entity = len(embedding)
        ranks_s    = []
        confs_s    = []

        head_relation_triplets = all_triplets[:, :2]

        for idx, test_triplet in enumerate(tqdm(test_triplets)):

            subject  = test_triplet[0]
            relation = test_triplet[1]
            object_  = test_triplet[2]

            subject_relation = test_triplet[:2]
            if embedding.is_cuda and not subject_relation.is_cuda:
                subject_relation = subject_relation.cuda()

            delete_index = torch.sum(
                head_relation_triplets == subject_relation, dim=1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 2].view(-1)
            if delete_entity_index.is_cuda:
                delete_entity_index_np = delete_entity_index.cpu().numpy()
            else:
                delete_entity_index_np = delete_entity_index.numpy()

            perturb_entity_index = np.array(
                list(set(np.arange(num_entity)) - set(delete_entity_index_np)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            if embedding.is_cuda:
                perturb_entity_index = perturb_entity_index.cuda()
            perturb_entity_index = torch.cat(
                (perturb_entity_index, object_.view(-1)))

            # ── Scoring ───────────────────────────────────────────────────────
            if score_func is not None:
                # ComplEx / RotatE — vectorised, no dim mismatch
                h_emb  = embedding[subject]               # [dim]
                t_embs = embedding[perturb_entity_index]  # [n_cand, dim]
                score  = score_func(h_emb, relation, t_embs).unsqueeze(0)  # [1, n_cand]
            else:
                # DistMult — original code, unchanged
                emb_ar = embedding[subject] * w[relation]
                emb_ar = emb_ar.view(-1, 1, 1)
                emb_c  = embedding[perturb_entity_index]
                emb_c  = emb_c.transpose(0, 1).unsqueeze(1)
                out_prod = torch.bmm(emb_ar, emb_c)
                score    = torch.sum(out_prod, dim=0)
                score    = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            if embedding.is_cuda:
                target = target.cuda()
            ranks_s.append(sort_and_rank(score, target))

            if conf_weights is not None:
                confs_s.append(conf_weights[idx].item())

        ranks_s = torch.cat(ranks_s)
        ranks   = ranks_s + 1   # 1-indexed

        # ── Standard metrics ──────────────────────────────────────────────
        mrr = torch.mean(1.0 / ranks.float())
        mr  = torch.mean(ranks.float())

        print(f"MRR (filtered): {mrr.item():.6f}")
        print(f"MR  (filtered): {mr.item():.6f}")

        hits_results = {}
        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print(f"Hits (filtered) @ {hit}: {avg_count.item():.6f}")
            hits_results[hit] = avg_count.item()

        # ── Weighted metrics ──────────────────────────────────────────────
        if conf_weights is not None and len(confs_s) > 0:
            c     = torch.tensor(confs_s, dtype=torch.float, device=ranks.device)
            c_sum = c.sum().clamp(min=1e-8)

            wmrr = (c * (1.0 / ranks.float())).sum() / c_sum
            print(f"wMRR (filtered): {wmrr.item():.6f}")
            hits_results["wmrr"] = wmrr.item()

            for hit in hits:
                whit = (c * (ranks <= hit).float()).sum() / c_sum
                print(f"wHits (filtered) @ {hit}: {whit.item():.6f}")
                hits_results[f"whits@{hit}"] = whit.item()

        return mrr.item(), hits_results

            