from typing import List, Tuple

from dev_misc import LT
from dev_misc.devlib import BT
from dev_misc.devlib.helper import get_tensor, pad_to_dense
from du.ipa.ipa_data import FEAT_ORDER, PHONO_FEATS
from du.model.modules import assign_names
from ipapy.ipachar import (DG_C_MANNER, DG_C_PLACE, DG_C_VOICING,
                           DG_DIACRITICS, DG_S_BREAK, DG_S_LENGTH, DG_S_STRESS,
                           DG_T_CONTOUR, DG_T_GLOBAL, DG_T_LEVEL, DG_TYPES,
                           DG_V_BACKNESS, DG_V_HEIGHT, DG_V_ROUNDNESS)
from ipapy.ipastring import IPAString

name2dg = {
    'c_manner': DG_C_MANNER,
    'c_place': DG_C_PLACE,
    'c_voicing': DG_C_VOICING,
    'diacritics': DG_DIACRITICS,
    's_break': DG_S_BREAK,
    's_length': DG_S_LENGTH,
    's_stress': DG_S_STRESS,
    't_contour': DG_T_CONTOUR,
    't_global': DG_T_GLOBAL,
    't_level': DG_T_LEVEL,
    'ptype': DG_TYPES,
    'v_backness': DG_V_BACKNESS,
    'v_height': DG_V_HEIGHT,
    'v_roundness': DG_V_ROUNDNESS
}


def get_feat_matrices(inputs: List[str], return_global_index: bool = True) -> Tuple[LT, BT]:
    """Get batched feature matrices based on `inputs`.

    Args:
        inputs (List[str]): a list of IPA transcriptions.
        return_global_index (bool, optional): flag to return global index, the index in `PHONO_FEATS` that contains all phonological features across all categories. If `False`, returns the local index, the index in its feature category. See the comments above `PHONO_FEATS` in `ipa_data.py` for an example of global and local indices. Defaults to True.

    Returns:
        Tuple[LT, BT]: a (batched feature matrix, padding) tuple.
    """
    batched_feat_mat = list()
    for inp in inputs:
        feat_mat = list()
        for char in IPAString(unicode_string=inp):
            feat_vec = list()
            for feat_cat in FEAT_ORDER:
                # Obtain the phonological feature from `ipapy`'s API.
                dg = name2dg[feat_cat.name]
                feat_name = char.dg_value(dg)
                feat_name = feat_name or 'none'  # Use "none" if it doesn't have this feature category.
                feat_name = feat_name.replace('-', '_')  # Replace hyphens with underscores.

                # Get the corresponding `Feature` instance based on name lookup.
                if return_global_index:
                    # NOTE(j_luo) Use the proper name "{feature_category}/{feature}" for lookup.
                    feat = PHONO_FEATS[f'{feat_cat.name}/{feat_name}']
                else:
                    feat = feat_cat[feat_name]
                # We only need the feature index for the embedding layer.
                feat_vec.append(feat.idx)
            feat_mat.append(feat_vec)
        batched_feat_mat.append(feat_mat)

    # Pad the matrix to the same length.
    matrix, padding = pad_to_dense(batched_feat_mat, dtype='long', use_3d=True, length_3d=len(FEAT_ORDER))
    matrix = get_tensor(matrix)
    # NOTE(j_luo) Revert the boolean since `pad_to_dense` sets `False` to padded positions.
    padding = ~get_tensor(padding)

    # Assign meaningful names to the tensors and return.
    assign_names(matrix, ['batch', 'length', 'cat'])
    assign_names(padding, ['batch', 'length'])
    return matrix, padding  # type: ignore
