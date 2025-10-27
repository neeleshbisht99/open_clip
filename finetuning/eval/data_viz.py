#%%#
""" To run: Just change the target_classes variable """
""" Import Dependencies """
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import numpy as np
import plotly.io as pio
import numpy as np
from sklearn.manifold import TSNE
from utils import display_projections_2d, display_projections_3d

pio.renderers.default = "vscode"

#%%#

data = np.load('img_embs_cap_embs_zeroshot_test_laion_vitb32.npz', allow_pickle=True)

image_embeddings = data['img_embs']
text_embeddings = data['cap_embs']
labels = np.array(['IMAGE'] * image_embeddings.shape[0])

#%%#
"""t-SNE (images + text) — 3d projections"""
X_joint = np.vstack([image_embeddings, text_embeddings])
tsne = TSNE(n_components=3, random_state=0)
projections = tsne.fit_transform(X_joint)
labels_joint = np.concatenate([labels, np.array(['TEXT'] * text_embeddings.shape[0])])

#%%#
"""Plot: images + text (3D)"""
# display_projections_3d(
#     labels=labels_joint,
#     projections=projections,
#     image_paths=None,
#     image_data_uris=None,
#     show_legend=True,
#     show_markers_with_text=False,
#     title="3D TSNE of CLIP Image + Text Embeddings",
#     file_name="cub200_fined_tuned_epoch_50_tsne_image_text_embeddings_3d"
# )

#%%#
"""t-SNE (images + text) — 2d projections"""
tsne = TSNE(n_components=2, random_state=0)
projections_2d = tsne.fit_transform(X_joint)

#%%#
"""Plot: images + text (2D)"""

display_projections_2d(
    labels=labels_joint,
    projections=projections_2d,
    image_paths=None,
    title="2D t-SNE — CLIP Image + Text Embeddings",
    file_name="cub200_tsne_image_text_embeddings_2d"
)