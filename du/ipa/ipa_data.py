"""This file stores the core data for all the phonological features used in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, List, TypeVar, Union


@dataclass
class Category:
    name: str


@dataclass
class Feature:
    category: Category
    name: str


ItemType = TypeVar('ItemType', Category, Feature)


class OrderedCollection(Generic[ItemType]):
    """Represents an ordered collection of items (basically list).

    The items can be instances of `Category` or `Feature`.
    """

    def __init__(self, name: str, items: List[ItemType]):
        self.name = name
        self.items = items
        self._name2item: Dict[str, ItemType] = {item.name: item for item in items}
        if len(self._name2item) != len(self.items):
            raise ValueError(f'Some items have duplicate names.')

    def __getitem__(self, key: Union[int, str]) -> ItemType:
        if isinstance(key, int):
            return self.items[key]
        else:
            return self._name2item[key]

    def __repr__(self):
        return f'"{self.name}" collection, size {len(self.items)}'

    @classmethod
    def chain(cls, name: str, *collections: OrderedCollection[ItemType]) -> OrderedCollection[ItemType]:
        """Chains a list of `OrderedCollection` instances and make a new one made up all the items. Order is preserved.

        Args:
            name (str): the name of the chained collection.

        Returns:
            OrderedCollection[ItemType]: the chained collection made up of all the items.
        """
        chained_items = list()
        for collection in collections:
            chained_items.extend(collection.items)
        return cls(name, chained_items)

# -------------------------------------------------------------- #
#    Define all the major categories of phonological features.   #
# -------------------------------------------------------------- #


CATEGORIES = OrderedCollection('category',
                               [Category(name) for name in ['ptype',
                                                            # Available for consonants.
                                                            'c_voicing', 'c_place', 'c_manner',
                                                            # Available for vowels.
                                                            'v_height', 'v_backness', 'v_roundness',
                                                            'diacritics',
                                                            # Available for suprasegmentals.
                                                            's_stress', 's_length', 's_break',
                                                            # Available for tones
                                                            't_level', 't_contour', 't_global']])

# -------------------------------------------------------------- #
#             Define all features for all categories             #
# -------------------------------------------------------------- #

PTYPE_FEATS = OrderedCollection('ptype',
                                [Feature(CATEGORIES['ptype'], name)
                                 for name in ['consonant', 'vowel']])

C_VOICING_FEATS = OrderedCollection('c_voicing',
                                    [Feature(CATEGORIES['c_voicing'], name)
                                     for name in ['none', 'voiced', 'voiceless']])

C_PLACE_FEATS = OrderedCollection('c_place',
                                  [Feature(CATEGORIES['c_place'], name)
                                   for name in ['none', 'alveolar', 'alveolo_palatal', 'bilabial', 'dental', 'glottal',
                                                'labio_alveolar', 'labio_dental', 'labio_palatal', 'labio_velar',
                                                'palatal', 'palato_alveolar', 'palato_alveolo_velar',
                                                'pharyngeal', 'retroflex', 'uvular', 'velar']])

C_MANNER_FEATS = OrderedCollection('c_manner',
                                   [Feature(CATEGORIES['c_manner'], name)
                                    for name in ['none', 'approximant', 'click', 'ejective', 'ejective_affricate',
                                                 'ejective_fricative', 'flap', 'implosive', 'lateral_affricate',
                                                 'lateral_approximant', 'lateral_click', 'lateral_ejective_affricate',
                                                 'lateral_flap', 'lateral_fricative', 'nasal', 'non_sibilant_affricate',
                                                 'non_sibilant_fricative', 'plosive', 'sibilant_affricate', 'sibilant_fricative', 'trill']])

V_HEIGHT_FEATS = OrderedCollection('v_height',
                                   [Feature(CATEGORIES['v_height'], name)
                                    for name in ['none', 'close', 'close_mid', 'mid',
                                                 'near_close', 'near_open', 'open', 'open_mid']])


V_BACKNESS_FEATS = OrderedCollection('v_backness',
                                     [Feature(CATEGORIES['v_backness'], name)
                                      for name in ['none', 'back', 'central', 'front', 'near_back', 'near_front']])


V_ROUNDNESS_FEATS = OrderedCollection('v_roundness',
                                      [Feature(CATEGORIES['v_roundness'], name)
                                       for name in ['none', 'rounded', 'unrounded']])

DIACRITICS_FEATS = OrderedCollection('diacritics',
                                     [Feature(CATEGORIES['diacritics'], name)
                                      for name in ['none', 'advanced', 'advanced_tongue_root', 'apical',
                                                   'aspirated', 'breathy_voiced', 'centralized', 'creaky_voiced',
                                                   'labialized', 'laminal', 'lateral_release', 'less_rounded',
                                                   'lowered', 'more_rounded', 'nasalized', 'no_audible_release',
                                                   'non_syllabic', 'palatalized', 'pharyngealized', 'raised',
                                                   'retracted', 'retracted_tongue_root', 'rhotacized', 'syllabic',
                                                   'tie_bar_above', 'tie_bar_below', 'velarized']])


S_STRESS_FEATS = OrderedCollection('s_stress',
                                   [Feature(CATEGORIES['s_stress'], name)
                                    for name in ['none', 'primary_stress']])


S_LENGTH_FEATS = OrderedCollection('s_length',
                                   [Feature(CATEGORIES['s_length'], name)
                                    for name in ['none', 'extra_short', 'half_long', 'long']])


S_BREAK_FEATS = OrderedCollection('s_break',
                                  [Feature(CATEGORIES['s_break'], name)
                                   for name in ['none', 'linking', 'syllable_break', 'word_break']])


T_LEVEL_FEATS = OrderedCollection('t_level',
                                  [Feature(CATEGORIES['t_level'], name)
                                   for name in ['none', 'extra_high_level', 'extra_low_level', 'high_level', 'low_level', 'mid_level']])

T_CONTOUR_FEATS = OrderedCollection('t_contour',
                                    [Feature(CATEGORIES['t_contour'], name)
                                     for name in ['none', 'falling_contour', 'high_mid_falling_contour', 'high_rising_contour',
                                                  'low_rising_contour', 'mid_low_falling_contour', 'rising_contour', 'rising_falling_contour']])

T_GLOBAL_FEATS = OrderedCollection('t_global',
                                   [Feature(CATEGORIES['t_global'], name)
                                    for name in ['none', 'downstep']])

# Put all features in a collection.
PHONO_FEATS = OrderedCollection.chain('phono',
                                      PTYPE_FEATS, C_VOICING_FEATS, C_PLACE_FEATS, C_MANNER_FEATS,
                                      V_HEIGHT_FEATS, V_BACKNESS_FEATS, V_ROUNDNESS_FEATS, DIACRITICS_FEATS,
                                      S_STRESS_FEATS, S_LENGTH_FEATS, S_BREAK_FEATS, T_LEVEL_FEATS,
                                      T_CONTOUR_FEATS, T_GLOBAL_FEATS)
