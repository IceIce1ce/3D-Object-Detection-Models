import itertools
import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler

class GroupedBatchSampler(BatchSampler):
    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of torch.utils.data.Sampler, but got sampler={}".format(sampler))
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven
        self.groups = torch.unique(self.group_ids).sort(0)[0]
        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        sampled_ids = torch.as_tensor(list(self.sampler))
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))
        mask = order >= 0
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        relative_order = [order[cluster] for cluster in clusters]
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))
        first_element_of_batch = [t[0].item() for t in merged]
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        first_index_of_batch = torch.as_tensor([inv_sampled_ids_map[s] for s in first_element_of_batch])
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        batches = [merged[i].tolist() for i in permutation_order]
        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)