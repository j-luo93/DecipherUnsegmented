from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from dev_misc import get_tensor
from dev_misc.devlib import BT, FT, LT
from du.ipa.ipa_data import CATEGORIES, PHONO_FEATS, OrderedCollection
from torch._C import Value
from torch.autograd.grad_mode import F

Tensor = torch.Tensor


def check_names(tensor: Union[Tensor, None], names: Sequence[Union[str, None]]) -> bool:
    """Checks if `tensor` has the required `names`.


    Note that if `None` is one of the names, that dimension is not checked.
    If `tensor` is `None`, return `True` by default.

    Args:
        tensor (Tensor): the tensor to check
        names (Sequence[Union[str, None]]): a sequence of names, including None.

    Returns:
        bool: whether `tensor` has the required `names`.
    """
    if tensor is None:
        return True

    # `tensor` must have the same length as `names`.
    if tensor.ndim != len(names):
        return False

    # `tensor` must have already been assigned names.
    if not hasattr(tensor, '_hidden_names'):
        return False

    for name, tname in zip(names, tensor._hidden_names):  # type: ignore
        # Skip if name is `None`.
        if name is None:
            continue
        if name != tname:
            return False

    return True


def assign_names(tensor: Tensor, names: Sequence[str]) -> None:
    """Assigns full `names` to the `tensor`.

    Args:
        tensor (Tensor): the tensor to assign names to.
        names (Sequence[str]): the full names to assign.

    Raises:
        RuntimeError: when `tensor` has already been assigned names once.
        ValueError: when `names` and `tensor` do not have the same dimensionality.
    """
    if hasattr(tensor, '_hidden_names'):
        raise RuntimeError(
            f'Can only assign hidden names once to a tensor, new names are {names}, but already has {tensor._hidden_names}.')  # type: ignore

    if tensor.ndim != len(names):
        raise ValueError(
            f'`tensor` and `names` must have the same dimensionality, but got {tensor.ndim} and {len(names)}.')

    tensor._hidden_names = names  # type: ignore


@dataclass
class EmbeddingParams:
    num_features: int
    embed_dim: int
    # a `str` object containing all one-lettered feature group codes. See `Category` in `ipa_data.py` for more details.
    feat_groups: str
    init_interval: Optional[float] = None


class FeatEmbedding(nn.Module):

    def __init__(self, emb_params: EmbeddingParams):
        super().__init__()
        self.emb_params = emb_params
        self._build()

    @classmethod
    def from_pretrained(cls, emb_params: EmbeddingParams, saved_dict: Dict[str, Tensor]) -> FeatEmbedding:
        """Loads a pretrained `FeatEmbedding` module from old `saved_dict`.

        Args:
            emb_params (EmbeddingParams): the hyperparameter object.
            saved_dict (Dict[str, Tensor]): the dictionary that stores the parameters.

        Returns:
            FeatEmbedding: the pretrained module.
        """
        mod = cls(emb_params)
        weight = torch.zeros(emb_params.num_features, emb_params.embed_dim)
        c_idx = saved_dict['base_embeddings.c_idx']
        for i in c_idx:
            cat = CATEGORIES[i.item()]  # NOTE(j_luo) Call `.item()` to turn it into proper `int`.
            cat_weight = saved_dict[f'base_embeddings.embed_layer.{cat.name.upper()}']
            # Get the starting index by using the first (local) feature of this category.
            cat_collection = OrderedCollection.get_instance(cat.name)
            first_feat_name = cat_collection[0].name
            start = PHONO_FEATS[f'{cat.name}/{first_feat_name}'].idx
            # Get the ending index.
            end = start + len(cat_collection)
            weight[start: end] = cat_weight
        mod.load_state_dict({'c_idx': c_idx, 'embed.weight': weight})  # type: ignore
        return mod

    def _build(self) -> None:
        """Builds this module.

        Raises:
            ValueError: when `feat_groups` contains duplicate letters.
        """
        self.embed = nn.Embedding(self.emb_params.num_features, self.emb_params.embed_dim)
        logging.warning('Sparse feature embedding not initialized')
        # Get all the relevant categories specified by `self.feat_groups`.
        fg = self.emb_params.feat_groups
        if len(set(fg)) != len(fg):
            raise ValueError(f'Duplicate values in feat_groups {fg}.')
        c_idx = list()  # Stores all relevant categories.
        for cat in CATEGORIES:
            if cat.group in fg:
                c_idx.append(cat.idx)
        self.register_buffer('c_idx', get_tensor(c_idx))

        # Prepare the parameters to have the right names.
        assign_names(self.c_idx, ['chosen_cat'])  # type: ignore
        assign_names(self.embed.weight, ['feat', 'feat_dim'])

    def _mask_out_padding(self, feat_emb: FT, padding: Optional[BT] = None) -> FT:
        """Mask out padded positions."""
        if padding is not None:
            feat_emb[padding] = 0.0  # Mask out all padded positions.
        return feat_emb

    def forward(self, feature_matrix: LT, padding: Optional[BT] = None) -> FT:
        """Obtains feature embedding from `feature_matrix`, with optional `padding`.

        Note that not every feature is used, depending on the available categories stored in `self.c_idx`.

        Args:
            feature_matrix (LT): the input feature matrix.
            padding (Optional[BT], optional): whether a position is a padding. Defaults to None.

        Returns:
            FT: the feature embedding
        """
        # Check dimensions first.
        check_names(feature_matrix, ['batch', 'length', 'cat'])
        check_names(padding, ['batch', 'length'])

        chosen_feature_matrix = feature_matrix[:, :, self.c_idx]  # batch x length x chosen_cat
        raw_feat_emb = self.embed(chosen_feature_matrix)  # batch x length x chosen_cat x feat_emb
        feat_emb = torch.flatten(raw_feat_emb, 2, 3)  # batch x length x char_emb

        # Assign names to the outputs.
        feat_emb = self._mask_out_padding(feat_emb, padding)  # type: ignore
        assign_names(feat_emb, ['batch', 'length', 'char_emb'])
        return feat_emb  # type: ignore


class DenseFeatEmbedding(FeatEmbedding):

    def _build(self):
        emb_dict = dict()
        ii = self.emb_params.init_interval
        if ii is None:
            raise ValueError(f'init_interval should not be None.')

        # Use one parameter set for each relevant category.
        for cat in CATEGORIES:
            if cat.group in self.feat_groups:
                cat_feats = OrderedCollection.get_instance(cat.name)
                nf = len(cat_feats)
                emb_dict[cat.name] = param = nn.Parameter(torch.zeros(nf, self.emb_params.embed_dim))
                logging.warning('dense feature embedding init')
                torch.nn.init.uniform_(param, -ii, ii)
        self.embed_dict = nn.ParameterDict(emb_dict)

        # Prepare the parameters to have the right names.
        for param in self.embed_dict.values():
            assign_names(param, ['feat', 'feat_dim'])

    def forward(self, dense_feat_matrices: Dict[str, FT], padding: Optional[BT] = None) -> FT:
        """Obtains feature embeddings based on dense feature matrices.

        Args:
            dense_feat_matrices (Dict[str, FT]): a mapping from category name to dense feature matrices
            padding (Optional[BT], optional): whether a position is padded. Defaults to None.

        Returns:
            FT: feature embeddings
        """
        for fm in dense_feat_matrices.values():
            check_names(fm, ['batch', 'length', 'feat'])
        check_names(padding, ['batch', 'length'])

        embs = list()
        for cat in CATEGORIES:
            if cat.name in self.embed_layer and cat.name in dense_feat_matrices:
                dfm = dense_feat_matrices[cat.name]
                emb_param = self.embed_dict[cat.name]
                emb = dfm @ emb_param  # batch x length x feat_dim
                embs.append(emb)
        feat_emb = torch.cat(embs, dim=-1)  # batch x length x char_dim

        feat_emb = self._mask_out_padding(feat_emb, padding)  # type: ignore
        assign_names(feat_emb, ['batch', 'length', 'char_emb'])
        return feat_emb
