"""Structured image/text negative sampling for ``DUAL_MODALITY``.

The candidate pools are fixed from the original item features and the train
split. Sampling is deliberately kept outside autograd: it only returns item
ids, per-example weights, and masks used by training diagnostics.
"""

import hashlib
import json
import os
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F


class ModalityNegativeSampler:
    """Cross-modality KNN pools with optional train-only CF filtering."""

    CACHE_VERSION = 1

    def __init__(
        self,
        image_features,
        text_features,
        interaction_matrix,
        mode,
        structured_ratio,
        structured_weight,
        pool_size,
        difference_quantile,
        min_item_interactions,
        cf_threshold,
        seed,
        cache_directory,
        knn_block_size=256,
    ):
        started_at = perf_counter()
        self.mode = mode
        self.structured_ratio = structured_ratio
        self.structured_weight = structured_weight
        self.pool_size = pool_size
        self.difference_quantile = difference_quantile
        self.min_item_interactions = min_item_interactions
        self.cf_threshold = cf_threshold
        self.knn_block_size = knn_block_size
        self.n_items = int(image_features.size(0))

        train_csr = interaction_matrix.tocsr(copy=True)
        train_csr.sum_duplicates()
        train_csr.eliminate_zeros()
        train_csr.data.fill(1.0)
        train_csr.sort_indices()
        self.train_csr = train_csr
        self.train_csc = train_csr.tocsc()
        self.train_csc.sort_indices()
        self.item_interaction_counts = np.diff(
            self.train_csc.indptr
        ).astype(np.int64, copy=False)
        self.user_histories = [
            set(
                self.train_csr.indices[
                    self.train_csr.indptr[user]:
                    self.train_csr.indptr[user + 1]
                ].tolist()
            )
            for user in range(self.train_csr.shape[0])
        ]

        self.metadata = self._cache_metadata(
            image_features, text_features, train_csr
        )
        metadata_digest = hashlib.sha256(
            json.dumps(
                self.metadata, sort_keys=True, separators=(',', ':')
            ).encode('utf-8')
        ).hexdigest()[:20]
        self.cache_file = os.path.join(
            cache_directory,
            'dual_modality_negative_pools_v{}_{}.pt'.format(
                self.CACHE_VERSION, metadata_digest
            ),
        )

        cached = self._load_pool_cache()
        self.cache_hit = cached is not None
        if cached is None:
            image_pool, text_pool = self._build_candidate_pools(
                image_features, text_features
            )
            self.image_offsets, self.image_candidates = image_pool
            self.text_offsets, self.text_candidates = text_pool
            self._save_pool_cache()
        else:
            self.image_offsets, self.image_candidates = cached['image']
            self.text_offsets, self.text_candidates = cached['text']

        seed_sequence = np.random.SeedSequence(int(seed))
        image_seed, text_seed = seed_sequence.spawn(2)
        self.rng = {
            'image': np.random.default_rng(image_seed),
            'text': np.random.default_rng(text_seed),
        }
        # CF values are computed only for candidate/history pairs that are
        # actually inspected. This avoids an n_items x n_items matrix.
        self._cf_cache = {}
        self.preprocessing_seconds = perf_counter() - started_at

    @staticmethod
    def _tensor_digest(tensor):
        array = tensor.detach().cpu().contiguous().numpy()
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order='C'))
        return digest.hexdigest()

    @staticmethod
    def _interaction_digest(train_csr):
        digest = hashlib.sha256()
        digest.update(
            np.asarray(train_csr.shape, dtype=np.int64).tobytes()
        )
        digest.update(
            train_csr.indptr.astype(np.int64, copy=False).tobytes()
        )
        digest.update(
            train_csr.indices.astype(np.int64, copy=False).tobytes()
        )
        return digest.hexdigest()

    def _cache_metadata(self, image_features, text_features, train_csr):
        return {
            'version': self.CACHE_VERSION,
            'num_items': self.n_items,
            'image_shape': list(image_features.shape),
            'text_shape': list(text_features.shape),
            'image_digest': self._tensor_digest(image_features),
            'text_digest': self._tensor_digest(text_features),
            'train_shape': list(train_csr.shape),
            'train_digest': self._interaction_digest(train_csr),
            'mode': self.mode,
            'pool_size': self.pool_size,
            'difference_quantile': self.difference_quantile,
            'min_item_interactions': self.min_item_interactions,
            'cf_threshold': self.cf_threshold,
            'knn_block_size': self.knn_block_size,
        }

    @staticmethod
    def _load_torch_file(file_path):
        try:
            return torch.load(
                file_path, map_location='cpu', weights_only=True
            )
        except TypeError:
            return torch.load(file_path, map_location='cpu')

    def _load_pool_cache(self):
        if not os.path.isfile(self.cache_file):
            return None
        try:
            payload = self._load_torch_file(self.cache_file)
        except (OSError, RuntimeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get('metadata') != self.metadata:
            return None

        result = {}
        for modality in ('image', 'text'):
            offsets = payload.get('{}_offsets'.format(modality))
            candidates = payload.get('{}_candidates'.format(modality))
            if not torch.is_tensor(offsets) or not torch.is_tensor(candidates):
                return None
            offsets = offsets.detach().cpu().numpy().astype(
                np.int64, copy=False
            )
            candidates = candidates.detach().cpu().numpy().astype(
                np.int64, copy=False
            )
            if (
                offsets.shape != (self.n_items + 1,)
                or offsets[0] != 0
                or offsets[-1] != candidates.size
                or np.any(offsets[1:] < offsets[:-1])
            ):
                return None
            result[modality] = (offsets, candidates)
        return result

    def _save_pool_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        payload = {
            'metadata': self.metadata,
            'image_offsets': torch.from_numpy(self.image_offsets),
            'image_candidates': torch.from_numpy(self.image_candidates),
            'text_offsets': torch.from_numpy(self.text_offsets),
            'text_candidates': torch.from_numpy(self.text_candidates),
        }
        temporary_file = self.cache_file + '.tmp'
        torch.save(payload, temporary_file)
        os.replace(temporary_file, self.cache_file)

    @staticmethod
    def _normalized_features(features):
        features = features.detach()
        valid = torch.isfinite(features).all(dim=1)
        safe_features = torch.where(
            torch.isfinite(features), features, torch.zeros_like(features)
        )
        norms = torch.linalg.vector_norm(safe_features, dim=1)
        valid = valid & (norms > 0.0)
        normalized = F.normalize(safe_features, p=2, dim=1, eps=1e-12)
        normalized = normalized.masked_fill(~valid.unsqueeze(1), 0.0)
        return normalized, valid

    def _build_one_pool(
        self,
        knn_features,
        difference_features,
        knn_valid,
        difference_valid,
    ):
        pools = []
        if self.n_items <= 1:
            return self._pack_pools([[] for _ in range(self.n_items)])

        candidate_valid = knn_valid & difference_valid
        valid_count = int(candidate_valid.sum().item())
        topk = min(self.pool_size, max(0, valid_count - 1))
        if topk == 0:
            return self._pack_pools([[] for _ in range(self.n_items)])
        for start in range(0, self.n_items, self.knn_block_size):
            stop = min(start + self.knn_block_size, self.n_items)
            similarities = torch.matmul(
                knn_features[start:stop], knn_features.transpose(0, 1)
            )
            similarities[:, ~candidate_valid] = -torch.inf
            local_rows = torch.arange(
                stop - start, device=similarities.device
            )
            item_ids = torch.arange(start, stop, device=similarities.device)
            similarities[local_rows, item_ids] = -torch.inf
            nearest_scores, nearest = torch.topk(
                similarities, topk, dim=1, largest=True, sorted=True
            )
            block_valid = candidate_valid[start:stop]
            difference_scores = torch.sum(
                difference_features[nearest]
                * difference_features[start:stop].unsqueeze(1),
                dim=2,
            )
            thresholds = torch.quantile(
                difference_scores, self.difference_quantile, dim=1
            )
            retained_mask = (
                (difference_scores <= thresholds.unsqueeze(1))
                & torch.isfinite(nearest_scores)
                & block_valid.unsqueeze(1)
            )
            nearest_cpu = nearest.detach().cpu().numpy()
            retained_cpu = retained_mask.detach().cpu().numpy()
            block_valid_cpu = block_valid.detach().cpu().numpy()

            for local_row in range(stop - start):
                if not block_valid_cpu[local_row]:
                    pools.append([])
                    continue
                pools.append(
                    nearest_cpu[local_row][
                        retained_cpu[local_row]
                    ].tolist()
                )
            del similarities
        return self._pack_pools(pools)

    @staticmethod
    def _pack_pools(pools):
        lengths = np.fromiter(
            (len(pool) for pool in pools), dtype=np.int64, count=len(pools)
        )
        offsets = np.empty(len(pools) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        if offsets[-1] == 0:
            candidates = np.empty(0, dtype=np.int64)
        else:
            candidates = np.concatenate(
                [np.asarray(pool, dtype=np.int64) for pool in pools]
            )
        return offsets, candidates

    def _build_candidate_pools(self, image_features, text_features):
        image_normalized, image_valid = self._normalized_features(
            image_features
        )
        text_normalized, text_valid = self._normalized_features(text_features)
        image_pool = self._build_one_pool(
            text_normalized,
            image_normalized,
            text_valid,
            image_valid,
        )
        text_pool = self._build_one_pool(
            image_normalized,
            text_normalized,
            image_valid,
            text_valid,
        )
        return image_pool, text_pool

    def pool_for(self, modality, positive_item):
        if modality == 'image':
            offsets, candidates = self.image_offsets, self.image_candidates
        elif modality == 'text':
            offsets, candidates = self.text_offsets, self.text_candidates
        else:
            raise ValueError("modality must be 'image' or 'text'.")
        start = offsets[positive_item]
        stop = offsets[positive_item + 1]
        return candidates[start:stop]

    def item_cf(self, first_item, second_item):
        """Cosine CF from train-user incidence, cached per inspected pair."""
        if first_item == second_item:
            return 1.0 if self.item_interaction_counts[first_item] > 0 else 0.0
        lower, upper = sorted((int(first_item), int(second_item)))
        key = lower * self.n_items + upper
        cached = self._cf_cache.get(key)
        if cached is not None:
            return cached
        lower_users = self.train_csc.indices[
            self.train_csc.indptr[lower]:self.train_csc.indptr[lower + 1]
        ]
        upper_users = self.train_csc.indices[
            self.train_csc.indptr[upper]:self.train_csc.indptr[upper + 1]
        ]
        denominator = np.sqrt(lower_users.size * upper_users.size)
        if denominator == 0.0:
            value = 0.0
        else:
            overlap = np.intersect1d(
                lower_users, upper_users, assume_unique=True
            ).size
            value = float(overlap / denominator)
        self._cf_cache[key] = value
        return value

    def _passes_cf_filter(self, candidate, history):
        return all(
            self.item_cf(history_item, candidate) <= self.cf_threshold
            for history_item in history
        )

    def sample(self, modality, users, positive_items, baseline_negatives):
        """Sample independent negatives for one modality.

        The caller only invokes this method when structured sampling is
        enabled. Therefore random/ratio-zero baselines consume no sampler RNG.
        """
        users_cpu = users.detach().cpu().numpy().astype(np.int64, copy=False)
        positives_cpu = positive_items.detach().cpu().numpy().astype(
            np.int64, copy=False
        )
        negatives_cpu = baseline_negatives.detach().cpu().numpy().astype(
            np.int64, copy=True
        )
        batch_size = negatives_cpu.size
        rng = self.rng[modality]
        attempted = rng.random(batch_size) < self.structured_ratio
        structured = np.zeros(batch_size, dtype=np.bool_)
        weights = np.ones(batch_size, dtype=np.float32)

        for index in np.flatnonzero(attempted):
            user = int(users_cpu[index])
            positive = int(positives_cpu[index])
            history = self.user_histories[user]
            candidates = self.pool_for(modality, positive)
            eligible = []
            for candidate in candidates:
                candidate = int(candidate)
                if candidate in history:
                    continue
                if (
                    self.item_interaction_counts[candidate]
                    < self.min_item_interactions
                ):
                    continue
                if (
                    self.mode == 'modality_cf'
                    and not self._passes_cf_filter(candidate, history)
                ):
                    continue
                eligible.append(candidate)
            if not eligible:
                continue
            selected = eligible[int(rng.integers(len(eligible)))]
            negatives_cpu[index] = selected
            structured[index] = True
            weights[index] = self.structured_weight

        device = baseline_negatives.device
        return {
            'negative_items': torch.as_tensor(
                negatives_cpu, dtype=torch.long, device=device
            ),
            'weights': torch.as_tensor(
                weights, dtype=torch.float32, device=device
            ),
            'attempted': torch.as_tensor(
                attempted, dtype=torch.bool, device=device
            ),
            'structured': torch.as_tensor(
                structured, dtype=torch.bool, device=device
            ),
        }

    def pool_summary(self):
        result = {}
        for modality, offsets in (
            ('image', self.image_offsets), ('text', self.text_offsets)
        ):
            lengths = np.diff(offsets)
            result[modality] = {
                'non_empty_fraction': float(np.mean(lengths > 0)),
                'mean_size': float(np.mean(lengths)),
                'max_size': int(lengths.max()) if lengths.size else 0,
            }
        return result
