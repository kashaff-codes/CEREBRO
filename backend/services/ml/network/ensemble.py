"""
Tree ensembles as plain arrays, scored with numpy.

Training happens offline with scikit-learn; serving does not import it. The
fitted trees are exported to one .npz (node arrays per tree, concatenated) and
scored here. Two reasons: the hosted API pins an older scikit-learn than the
training machine, and a pickled model does not survive a version change; and a
pickle is code, while arrays are data, so loading a model file cannot execute
anything.

  IsolationForestArrays.score(X)  -> anomaly score in (0, 1), higher = stranger;
                                     identical to -IsolationForest.score_samples
  ForestClassifierArrays.proba(X) -> per-class probability, identical to
                                     RandomForestClassifier.predict_proba

tests/test_network.py checks both identities against scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EULER = 0.5772156649015329


def _avg_path(n: np.ndarray) -> np.ndarray:
    """c(n): average path length of an unsuccessful BST search (Liu et al., 2008)."""
    n = np.asarray(n, dtype=np.float64)
    out = np.zeros_like(n)
    m2 = n == 2
    big = n > 2
    out[m2] = 1.0
    out[big] = 2.0 * (np.log(n[big] - 1.0) + _EULER) - 2.0 * (n[big] - 1.0) / n[big]
    return out


@dataclass
class _Trees:
    feature: np.ndarray      # int32, -2 at leaves (already mapped to global feature indices)
    threshold: np.ndarray    # float64
    left: np.ndarray         # int32, global node index, -1 at leaves
    right: np.ndarray        # int32
    root: np.ndarray         # int32, root node index of each tree

    def leaves(self, X: np.ndarray) -> np.ndarray:
        """Leaf index reached by every row in every tree: shape (n_trees, n_rows)."""
        n = X.shape[0]
        out = np.empty((len(self.root), n), dtype=np.int64)
        rows = np.arange(n)
        for t, r in enumerate(self.root):
            node = np.full(n, r, dtype=np.int64)
            while True:
                f = self.feature[node]
                internal = f >= 0
                if not internal.any():
                    break
                go_left = np.zeros(n, dtype=bool)
                idx = rows[internal]
                go_left[idx] = X[idx, f[idx]] <= self.threshold[node[idx]]
                nxt = np.where(go_left, self.left[node], self.right[node])
                node = np.where(internal, nxt, node)
            out[t] = node
        return out

    def depths(self) -> np.ndarray:
        depth = np.zeros(len(self.feature), dtype=np.int64)
        for r in self.root:
            stack = [(int(r), 0)]
            while stack:
                i, d = stack.pop()
                depth[i] = d
                if self.feature[i] >= 0:
                    stack.append((int(self.left[i]), d + 1))
                    stack.append((int(self.right[i]), d + 1))
        return depth


def _pack(estimators, feature_maps) -> dict[str, np.ndarray]:
    feat, thr, left, right, roots, nsamp, values = [], [], [], [], [], [], []
    offset = 0
    for est, fmap in zip(estimators, feature_maps):
        t = est.tree_
        f = t.feature.astype(np.int64)
        mapped = np.where(f >= 0, np.asarray(fmap)[np.clip(f, 0, None)], -2)
        feat.append(mapped.astype(np.int32))
        thr.append(t.threshold.astype(np.float64))
        left.append(np.where(t.children_left >= 0, t.children_left + offset, -1).astype(np.int32))
        right.append(np.where(t.children_right >= 0, t.children_right + offset, -1).astype(np.int32))
        nsamp.append(t.n_node_samples.astype(np.int64))
        v = t.value[:, 0, :].astype(np.float64)
        values.append(v / np.clip(v.sum(axis=1, keepdims=True), 1e-12, None))
        roots.append(offset)
        offset += t.node_count
    return {"feature": np.concatenate(feat), "threshold": np.concatenate(thr),
            "left": np.concatenate(left), "right": np.concatenate(right),
            "root": np.asarray(roots, dtype=np.int32), "n_samples": np.concatenate(nsamp),
            "value": np.concatenate(values)}


class IsolationForestArrays:
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.trees = _Trees(arrays["feature"], arrays["threshold"], arrays["left"], arrays["right"],
                            arrays["root"])
        self.max_samples = int(arrays["max_samples"])
        # Path length to each leaf, corrected for the points it still holds.
        self._leaf_path = self.trees.depths() + _avg_path(arrays["n_samples"])

    @classmethod
    def export(cls, model) -> dict[str, np.ndarray]:
        packed = _pack(model.estimators_, model.estimators_features_)
        packed["max_samples"] = np.asarray(model.max_samples_)
        packed.pop("value")
        return packed

    def score(self, X: np.ndarray) -> np.ndarray:
        leaves = self.trees.leaves(np.asarray(X, dtype=np.float64))
        mean_path = self._leaf_path[leaves].mean(axis=0)
        return 2.0 ** (-mean_path / _avg_path(np.asarray([self.max_samples]))[0])


class ForestClassifierArrays:
    def __init__(self, arrays: dict[str, np.ndarray], classes: list[str]):
        self.trees = _Trees(arrays["feature"], arrays["threshold"], arrays["left"], arrays["right"],
                            arrays["root"])
        self.value = arrays["value"]
        self.classes = list(classes)

    @classmethod
    def export(cls, model) -> dict[str, np.ndarray]:
        n = model.n_features_in_
        return _pack(model.estimators_, [np.arange(n)] * len(model.estimators_))

    def proba(self, X: np.ndarray) -> np.ndarray:
        leaves = self.trees.leaves(np.asarray(X, dtype=np.float64))
        return self.value[leaves].mean(axis=0)
