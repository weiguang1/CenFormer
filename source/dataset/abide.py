import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_abide_data(cfg: DictConfig):

    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    final_timeseires = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]
    site = data['site']

    # Canonical orientation is (subject, node, time) — abide.npy (CC200) is
    # stored that way, but bnt_ready/abide116.npy is (subject, time, node).
    # Only FBNetGen consumes the timeseries, so the fix is invisible to the
    # other models. Disambiguate via the node count from the corr matrix.
    n_nodes = final_pearson.shape[1]
    if final_timeseires.shape[1] != n_nodes and final_timeseires.shape[2] == n_nodes:
        final_timeseires = final_timeseires.transpose(0, 2, 1)

    scaler = StandardScaler(mean=np.mean(
        final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)

    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]

    return final_timeseires, final_pearson, labels, site
