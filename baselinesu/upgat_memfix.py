# -*- coding: utf-8 -*-
"""Memory fix for unKR's UPGAT training step.

unKR's UPGATLitModel.training_step calls the model three times per step
(positive, negative and pseudo samples). Each call runs forward_GAT over the
FULL training graph, so three full-graph autograd graphs are alive at once
before backward. On CN15k/NL27k this needs >48 GB at step 0, independently of
train_bs.

This patch runs the original forward_GAT once per step and reuses the updated
entity/relation embeddings for the three scores. Model, loss and scoring are
unchanged; the only difference is that the attention dropout mask is shared
by the three scores of a step instead of being resampled for each one.

Enabled from UPGATdemo.py when the environment variable UPGAT_MEMFIX=1.
"""

import torch
from unKR.lit_model.UPGATLitModel import UPGATLitModel


def _gat_embeddings(model, adj_matrix):
    """Run the original forward_GAT once and return the full updated
    (entity, relation) embedding tables, with gradients attached."""
    num_ent, num_rel = model.args.num_ent, model.args.num_rel
    assert num_ent >= num_rel
    ids = torch.arange(num_ent, device=model.args.gpu)
    # head column = every entity, relation column = every relation (padded)
    probe = torch.stack((ids, ids.clamp(max=num_rel - 1), ids), dim=1)
    head_emb, rel_emb, _ = model.forward_GAT(probe, adj_matrix, None, "single")
    return head_emb.squeeze(1), rel_emb.squeeze(1)[:num_rel]


def _lookup(ent, rel, triples, negs, mode):
    triples = triples[:, :3].to(torch.long)
    if mode == "head-batch":
        return ent[negs], rel[triples[:, 1]].unsqueeze(1), ent[triples[:, 2]].unsqueeze(1)
    if mode == "tail-batch":
        return ent[triples[:, 0]].unsqueeze(1), rel[triples[:, 1]].unsqueeze(1), ent[negs]
    return (ent[triples[:, 0]].unsqueeze(1), rel[triples[:, 1]].unsqueeze(1),
            ent[triples[:, 2]].unsqueeze(1))


def _score(model, ent, rel, triples, negs=None, mode="single"):
    h, r, t = _lookup(ent, rel, triples, negs, mode)
    return model.score_func(h, r, t, mode)


def training_step(self, batch, batch_idx):
    adj_matrix = batch["adj_matrix"]
    pos_sample = batch["positive_sample"]
    neg_sample = batch["negative_sample"]
    mode = batch["mode"]

    ent, rel = _gat_embeddings(self.model, adj_matrix)
    pos_score = _score(self.model, ent, rel, pos_sample)
    neg_score = _score(self.model, ent, rel, pos_sample, neg_sample, mode)
    if self.args.teacher_model:
        loss = self.loss(pos_score, neg_score, pos_sample)
    else:
        pseudo_sample = batch["pseudo_sample"]
        pseudo_score = _score(self.model, ent, rel, pseudo_sample)
        loss = self.loss(pos_score, neg_score, pos_sample, pseudo_score, pseudo_sample)
    self.log("Train|loss", loss, on_step=False, on_epoch=True)
    return loss


def apply():
    UPGATLitModel.training_step = training_step
    print("[upgat_memfix] UPGAT training_step patched: one GAT pass per step.")
