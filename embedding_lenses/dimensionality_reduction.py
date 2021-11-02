import numpy as np
import umap
from sklearn.manifold import TSNE


def get_tsne_embeddings(
    embeddings: np.ndarray, perplexity: int = 30, n_components: int = 2, init: str = "pca", n_iter: int = 5000, random_state: int = 0
) -> np.ndarray:
    tsne = TSNE(perplexity=perplexity, n_components=n_components, init=init, n_iter=n_iter, random_state=random_state)
    return tsne.fit_transform(embeddings)


def get_umap_embeddings(embeddings: np.ndarray, random_state: int = 0) -> np.ndarray:
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=random_state)
    return umap_model.fit_transform(embeddings)
