#!/usr/bin/env python3
"""Checkpoint-backed, state-safe Zebra link prediction inference."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from model.tgn_model import TGN
from utils.data_processing import DATA_ROOT, Data, load_feat
from utils.util import get_neighbor_finder


DEFAULT_MODEL_CONFIG = {
  "n_layer": 2,
  "n_head": 2,
  "drop_out": 0.3,
  "node_dim": 100,
  "time_dim": 100,
  "memory_dim": 100,
  "n_degree": 10,
  "message_function": "identity",
  "memory_updater": "gru",
  "aggregator": "last",
  "tppr_strategy": "streaming",
  "topk": 20,
  "alpha_list": [0.1, 0.1],
  "beta_list": [0.5, 0.95],
  "use_destination_embedding_in_message": False,
  "use_source_embedding_in_message": False,
}


class ZebraLinkPredictor:
  """Load Zebra parameters and expose history-aware, read-only edge scoring."""

  def __init__(self, dataset_name, checkpoint_path, device="cuda:0",
               model_config=None, replay_batch_size=200):
    self.dataset_name = dataset_name
    self.checkpoint_path = Path(checkpoint_path).resolve()
    self.device = self._resolve_device(device)
    self.replay_batch_size = int(replay_batch_size)
    if self.replay_batch_size <= 0:
      raise ValueError("replay_batch_size must be positive")

    config = dict(DEFAULT_MODEL_CONFIG)
    if model_config:
      config.update(model_config)
    self.config = config

    data_path = DATA_ROOT / dataset_name / "ml_{}.csv".format(dataset_name)
    if not data_path.is_file():
      raise FileNotFoundError("Zebra dataset not found: {}".format(data_path))
    if not self.checkpoint_path.is_file():
      raise FileNotFoundError(
        "Zebra checkpoint not found: {}".format(self.checkpoint_path)
      )
    self.graph_df = pd.read_csv(data_path).sort_values(
      ["ts", "idx"], kind="stable"
    ).reset_index(drop=True)
    self._validate_graph_data()

    sources = self.graph_df.u.to_numpy(dtype=np.int32)
    destinations = self.graph_df.i.to_numpy(dtype=np.int32)
    timestamps = self.graph_df.ts.to_numpy(dtype=np.float32)
    edge_idxs = self.graph_df.idx.to_numpy(dtype=np.int32)
    labels = self.graph_df.label.to_numpy()
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    args = SimpleNamespace(**config)
    args.n_nodes = int(max(sources.max(), destinations.max())) + 1
    args.n_edges = int(edge_idxs.max()) + 1
    node_features, edge_features = load_feat(dataset_name)
    if edge_features is None:
      edge_features = np.zeros((args.n_edges, 1), dtype=np.float32)
    if len(edge_features) < args.n_edges:
      raise ValueError("edge feature array is shorter than the edge index range")

    self.model = TGN(
      neighbor_finder=get_neighbor_finder(full_data),
      node_features=node_features,
      edge_features=edge_features,
      device=self.device,
      n_layers=args.n_layer,
      n_heads=args.n_head,
      dropout=args.drop_out,
      use_memory=True,
      node_dimension=args.node_dim,
      time_dimension=args.time_dim,
      memory_dimension=args.memory_dim,
      embedding_module_type="diffusion",
      message_function=args.message_function,
      aggregator_type=args.aggregator,
      memory_updater_type=args.memory_updater,
      n_neighbors=args.n_degree,
      use_destination_embedding_in_message=(
        args.use_destination_embedding_in_message
      ),
      use_source_embedding_in_message=args.use_source_embedding_in_message,
      args=args,
    ).to(self.device)
    checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
    if not isinstance(checkpoint, tuple) or len(checkpoint) != 2:
      raise ValueError("expected Zebra checkpoint tuple (state_dict, memory)")
    self.model.load_state_dict(checkpoint[0], strict=True)
    self.model.eval()
    self.model.reset_timer()
    self.current_timestamp = 0
    self._reset_history()

  @staticmethod
  def _resolve_device(device):
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
      raise RuntimeError(
        "CUDA was requested for Zebra inference but is not available"
      )
    return requested

  def _validate_graph_data(self):
    required = {"u", "i", "ts", "label", "idx"}
    if not required.issubset(self.graph_df.columns):
      raise ValueError("Zebra edge CSV must contain u,i,ts,label,idx")
    if self.graph_df.empty:
      raise ValueError("Zebra edge CSV is empty")
    if (self.graph_df.ts < 1).any():
      raise ValueError("Zebra timestamps must start at 1 or later")

  def _reset_history(self):
    self.model.memory.__init_memory__()
    if self.config["tppr_strategy"] == "streaming":
      self.model.embedding_module.reset_tppr()
    self.model.test_mode = False
    self.current_timestamp = 0

  @property
  def max_timestamp(self):
    return int(self.graph_df.ts.max())

  def replay_until(self, observed_timestamp):
    """Advance model state using real interactions through the given time."""
    observed_timestamp = int(observed_timestamp)
    if observed_timestamp < 0 or observed_timestamp > self.max_timestamp:
      raise ValueError(
        "observed_timestamp must be in [0, {}]".format(self.max_timestamp)
      )
    if observed_timestamp < self.current_timestamp:
      self._reset_history()
    if observed_timestamp == self.current_timestamp:
      return

    mask = (
      (self.graph_df.ts > self.current_timestamp)
      & (self.graph_df.ts <= observed_timestamp)
    )
    pending = self.graph_df.loc[mask]
    with torch.no_grad():
      for timestamp, time_rows in pending.groupby("ts", sort=True):
        for start in range(0, len(time_rows), self.replay_batch_size):
          batch = time_rows.iloc[start:start + self.replay_batch_size]
          sources = batch.u.to_numpy(dtype=np.int32)
          destinations = batch.i.to_numpy(dtype=np.int32)
          times = batch.ts.to_numpy(dtype=np.float32)
          edge_idxs = batch.idx.to_numpy(dtype=np.int32)
          self.model.compute_temporal_embeddings(
            sources,
            destinations,
            sources,
            times,
            edge_idxs,
            self.config["n_degree"],
            train=False,
          )
        self.current_timestamp = int(timestamp)

  def encode_nodes(self, node_ids, query_timestamp, batch_size=4096):
    """Encode node IDs at one future timestamp without changing history."""
    node_ids = np.asarray(node_ids, dtype=np.int32)
    if node_ids.ndim != 1:
      raise ValueError("node_ids must be one-dimensional")
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    if len(node_ids) == 0:
      hidden_dim = self.config["node_dim"] * (
        len(self.config["alpha_list"]) + 1
      )
      return torch.empty((0, hidden_dim), device=self.device)
    if node_ids.min() < 0 or node_ids.max() >= self.model.n_nodes:
      raise ValueError("node ID is outside the Zebra mapping")
    if query_timestamp <= self.current_timestamp:
      raise ValueError("query_timestamp must be later than observed history")

    embeddings = []
    with torch.no_grad():
      for start in range(0, len(node_ids), batch_size):
        batch_nodes = node_ids[start:start + batch_size]
        times = np.full(len(batch_nodes), query_timestamp, dtype=np.float32)
        embeddings.append(self.model.compute_node_embeddings_readonly(
          batch_nodes, times
        ))
    return torch.cat(embeddings, dim=0)

  def score_directed_embeddings(self, source_embeddings,
                                destination_embeddings):
    with torch.no_grad():
      return self.model.decode_edge_probabilities(
        source_embeddings, destination_embeddings
      )

  def score_undirected_embeddings(self, left_embeddings, right_embeddings):
    """Average both directed decoder orientations for an undirected score."""
    forward = self.score_directed_embeddings(
      left_embeddings, right_embeddings
    )
    reverse = self.score_directed_embeddings(
      right_embeddings, left_embeddings
    )
    return (forward + reverse) / 2
