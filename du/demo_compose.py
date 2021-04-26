"""A simple demo script for initializing a `FeatEmbedding` module that composes phonological embeddings from feature matrices.
"""
from du.ipa.process import get_feat_matrices
from du.model.modules import EmbeddingParams, FeatEmbedding
from du.ipa.ipa_data import PHONO_FEATS

if __name__ == '__main__':
    # Init an embedding layer with dimensionality 128 for feature groups "pcv". See `Category` in `ipa_data.py` for more details on feature groups.
    emb_params = EmbeddingParams(len(PHONO_FEATS), 128, 'pcv')
    embed_layer = FeatEmbedding(emb_params)

    # Use `get_feature_matrices` to convert a list of IPA transcriptions to a batched feature matrix (stored as a tensor).
    fm, padding = get_feat_matrices(['æpəl', 'bənænə'])
    print(embed_layer(fm, padding=padding))
