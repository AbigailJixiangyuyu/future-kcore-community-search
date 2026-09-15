"""Portable, complete streaming Zebra state at an observed snapshot boundary.

Only tensors and primitive containers are serialized (no Numba objects). Preserve
T-PPR dictionary insertion order: it can affect ties and floating-point sums.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import torch


VERSION = 1
MEMORY_FIELDS = ("memory", "last_update", "messages", "timestamps")


def _load_state_file(path):
    # The repository also supports pre-weights_only PyTorch. Load only caches
    # generated locally in the configured trusted cache directory.
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    return torch.load(path, **options)


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ZebraHistoryCache:
    def __init__(self, predictor, directory, source_root=None):
        self.predictor = predictor
        self.directory = Path(directory)
        self.source_root = Path(source_root) if source_root else None
        self._identity = None

    def identity(self):
        if self._identity is None:
            p = self.predictor
            z = p.zebra
            if z.config["tppr_strategy"] != "streaming":
                raise ValueError("Zebra history cache supports streaming T-PPR only")
            digest = hashlib.sha256()
            # The replay dataframe order, values and feature contents are causal inputs.
            for name in ("u", "i", "ts", "idx"):
                array = np.ascontiguousarray(z.graph_df[name].to_numpy())
                digest.update(str(array.dtype).encode())
                digest.update(array.tobytes())
            features = z.model.edge_raw_features.detach().cpu().contiguous().numpy()
            digest.update(str(features.dtype).encode())
            digest.update(str(features.shape).encode())
            digest.update(features.tobytes())
            sources = {_hash_file(__file__)}
            if self.source_root:
                for folder in ("model", "modules", "utils"):
                    sources.update(_hash_file(path) for path in
                                   (self.source_root / folder).rglob("*.py"))
                sources.add(_hash_file(self.source_root / "inference.py"))
            device = torch.device(z.device)
            metadata = {
                "version": VERSION, "checkpoint": p.checkpoint_hash,
                "config": z.config, "mapping": p.mapping_hash,
                "replay_batch_size": z.replay_batch_size,
                "data_and_features": digest.hexdigest(), "sources": sorted(sources),
                "torch": str(torch.__version__), "numpy": np.__version__,
                "device_type": device.type,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            }
            self._identity = hashlib.sha256(
                json.dumps(metadata, sort_keys=True).encode()
            ).hexdigest()
        return self._identity

    def path(self, t):
        return self.directory / self.identity() / f"state_t{t:06d}.pt"

    def capture(self):
        z = self.predictor.zebra
        model = z.model
        memory = model.memory
        finder = model.embedding_module.tppr_finder
        return {
            "memory": {name: getattr(memory, name).detach().cpu().clone()
                       for name in MEMORY_FIELDS},
            "nodes": torch.from_numpy(memory.nodes.copy()),
            "norms": [torch.from_numpy(row.copy()) for row in finder.norm_list],
            "ppr": [
                [[(int(key[0]), int(key[1]), float(key[2]), float(weight))
                  for key, weight in row.items()] for row in group]
                for group in finder.PPR_list
            ],
            "current_timestamp": int(z.current_timestamp),
            "test_mode": bool(model.test_mode),
            "batch_counter": int(model.batch_counter),
            "n_update_memory": int(model.n_update_memory),
        }

    def restore(self, state, observed):
        """Validate before mutation; rebuild independently owned runtime state."""
        z = self.predictor.zebra
        model = z.model
        memory = model.memory
        finder = model.embedding_module.tppr_finder
        if not 0 <= state["current_timestamp"] <= observed:
            raise ValueError("cached history exceeds the observed boundary")
        for name in MEMORY_FIELDS:
            tensor = state["memory"][name]
            current = getattr(memory, name)
            if tensor.shape != current.shape or tensor.dtype != current.dtype:
                raise ValueError(f"incompatible memory field: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError("non-finite cached memory")
        if state["nodes"].shape != (finder.num_nodes,) or state["nodes"].dtype != torch.bool:
            raise ValueError("incompatible pending-message flags")
        if len(state["norms"]) != finder.n_tppr or len(state["ppr"]) != finder.n_tppr:
            raise ValueError("incompatible T-PPR ensemble")
        for norm, group in zip(state["norms"], state["ppr"]):
            if norm.shape != (finder.num_nodes,) or norm.dtype != torch.float64:
                raise ValueError("incompatible T-PPR normalization")
            if not torch.isfinite(norm).all() or len(group) != finder.num_nodes:
                raise ValueError("invalid T-PPR state")
            for row in group:
                if len(row) > finder.k or len({tuple(entry[:3]) for entry in row}) != len(row):
                    raise ValueError("invalid T-PPR row")
                for edge, node, timestamp, weight in row:
                    if not (0 <= node < finder.num_nodes
                            and 0 <= edge < len(model.edge_raw_features)
                            and 0 <= timestamp <= observed and np.isfinite(weight)):
                        raise ValueError("invalid T-PPR entry")
        # reset_tppr supplies the original Numba key/value types.
        finder.reset_tppr()
        for i, (norm, group) in enumerate(zip(state["norms"], state["ppr"])):
            finder.norm_list[i][:] = norm.numpy()
            for node, entries in enumerate(group):
                row = finder.PPR_list[i][node]
                for edge, neighbor, timestamp, weight in entries:
                    row[(edge, neighbor, timestamp)] = weight
        for name in MEMORY_FIELDS:
            setattr(memory, name, state["memory"][name].to(z.device).clone())
        memory.nodes = state["nodes"].numpy().copy()
        model.test_mode = state["test_mode"]
        model.batch_counter = state["batch_counter"]
        model.n_update_memory = state["n_update_memory"]
        z.current_timestamp = state["current_timestamp"]

    def load(self, t):
        path = self.path(t)
        if not path.is_file():
            return False
        try:
            payload = _load_state_file(path)
            observed = self.predictor.time_to_zebra[t]
            if (payload["version"] != VERSION or payload["identity"] != self.identity()
                    or payload["t"] != t or payload["observed"] != observed):
                raise ValueError("incompatible cache metadata")
            self.restore(payload["state"], observed)
        except Exception as error:
            # Restore may have started constructing typed dictionaries: reset fully.
            self.predictor.zebra._reset_history()
            warnings.warn(f"Ignoring invalid Zebra history cache {path}: {error}")
            return False
        return True

    def save(self, t):
        path = self.path(t)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": VERSION, "identity": self.identity(), "t": t,
            "observed": self.predictor.time_to_zebra[t], "state": self.capture(),
        }
        handle, temporary = tempfile.mkstemp(prefix=".state-", suffix=".pt", dir=path.parent)
        os.close(handle)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path
