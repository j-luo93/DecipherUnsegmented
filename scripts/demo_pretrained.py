"""A simple demo script for load a pretrained `FeatEmbedding` module that composes phonological embeddings from feature matrices.
"""
import torch
import pandas as pd
import pickle
from du.ipa.ipa_data import PHONO_FEATS
from du.ipa.process import get_feat_matrices
from du.model.modules import EmbeddingParams, FeatEmbedding

if __name__ == '__main__':
    # Load a pretrained embedding layer with dimensionality 100 for feature groups "pcv". See `Category` in `ipa_data.py` for more details on feature groups.
    emb_params = EmbeddingParams(len(PHONO_FEATS), 100, 'pcv')
    saved_dict = torch.load('data/got.pretrained.pth')
    embed_layer = FeatEmbedding.from_pretrained(emb_params, saved_dict)

    # Use `get_feature_matrices` to convert a list of IPA transcriptions to a batched feature matrix (stored as a tensor).
    with open('data/segments.pkl', 'rb') as fin:
        segments = pickle.load(fin)
        # Each segment is treated as a sequence of length 1.
        fm, padding = get_feat_matrices(segments)
        emb = embed_layer(fm, padding=padding)[:, 0]  # Only take the first character in the sequence.
        emb_df = pd.DataFrame(emb.detach().cpu().numpy())
        seg_df = pd.DataFrame(segments)
        emb_df.to_csv('data/segments.emb.tsv', sep='\t', index=False)
        seg_df.to_csv('data/segments.tsv', sep='\t', index=False)
