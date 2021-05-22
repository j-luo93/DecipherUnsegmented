"""This file stores the core data for all the phonological features used in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import (ClassVar, Dict, Generic, Iterator, List, Optional, TypeVar,
                    Union)


@dataclass
class Category:
    name: str

    # A one-letter code to represent the group of this category of features.
    # p: for ptype only
    # c: consonants
    # v: vowels
    # d: diacritics
    # s: suprasegmental
    # t: tone
    group: str = field(init=False)

    # This is set to None by default, and should be set by `OrderedCollection`.
    idx: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        if self.name == 'ptype':
            self.group = 'p'
        else:
            self.group = self.name[0]  # The first letter is always the group code.
            if self.group not in ['c', 'v', 'd', 's', 't']:
                raise ValueError(f'Unrecognized group code {self.group}.')


@dataclass
class Feature:
    category: Category
    name: str
    idx: Optional[int] = None


ItemType = TypeVar('ItemType', Category, Feature)


class OrderedCollection(Generic[ItemType]):
    """Represents an ordered collection of items (basically list).

    The items can be instances of `Category` or `Feature`.
    Note that every collection ever created will be stored in the class variable,
    and no duplicate name is allowed.
    The `idx` field in these items will be set after calling `__init__` on `OrderedCollection`.
    """

    _instances: ClassVar[Dict[str, OrderedCollection]] = dict()

    def __init__(self, name: str, items: List[ItemType]):
        """Creates an `OrderedCollection` instance consisting of `items`.

        Args:
            name (str): the name for this instance.
            items (List[ItemType]): the items in this instance.

        Raises:
            RuntimeError: duplicate name is provided.
            RuntimeError: some item in `items` has been indexed before.
            ValueError: some items in `items` have duplicate names.
        """
        # Check no duplicate name first.
        cls = type(self)
        if name in cls._instances:
            raise RuntimeError(f'A collection named "{name}" has been created before.')
        cls._instances[name] = self

        # Set the index for each item starting from 0. Note that we assert it has not bee indexed before.
        for i, item in enumerate(items):
            if item.idx is not None:
                raise RuntimeError(f'This item has been indexed before.')
            item.idx = i

        self.name = name
        self.items = items

        # name-to-item mapping, used for `__getitem__` when calling with `str`.
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

    def __iter__(self) -> Iterator[ItemType]:
        yield from self.items

    def __len__(self):
        return len(self.items)

    @classmethod
    def get_instance(cls, name: str) -> OrderedCollection[ItemType]:
        return cls._instances[name]

    @classmethod
    def chain(cls, name: str, *collections: OrderedCollection[ItemType]) -> OrderedCollection[ItemType]:
        """Chains a list of `OrderedCollection` instances and make a new one made up all the items.

        Note that order is preserved and the new items will have a new name with the format "{collection}/{name}",
        where "collection" is the the name of the original collection it belongs to, and
        "name" is its original name.

        Args:
            name (str): the name of the chained collection.

        Returns:
            OrderedCollection[ItemType]: the chained collection made up of all the items.
        """
        chained_items = list()
        for collection in collections:
            # NOTE(j_luo) (Shallow-)copy the dataclass since indices will be changed after chaining. Remember to modify the `idx` and the `name` fields.
            chained_items.extend(
                [replace(item, idx=None, name=f'{collection.name}/{item.name}') for item in collection.items])
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

# Put all features in a collection. This collection reindexes every feature and assign a new "global" index to them.
# The original index can be accessed in the feature's own feature category.
# For instance, for "close" feature of vowel height.
# `PHONO_FEATS["v_height/close"].idx` returns its global index,
# `V_HEIGHT_FEATS["close"].idx` returns its local index.
FEAT_ORDER = [PTYPE_FEATS, C_VOICING_FEATS, C_PLACE_FEATS, C_MANNER_FEATS,
              V_HEIGHT_FEATS, V_BACKNESS_FEATS, V_ROUNDNESS_FEATS, DIACRITICS_FEATS,
              S_STRESS_FEATS, S_LENGTH_FEATS, S_BREAK_FEATS, T_LEVEL_FEATS,
              T_CONTOUR_FEATS, T_GLOBAL_FEATS]
PHONO_FEATS = OrderedCollection.chain('phono', *FEAT_ORDER)
