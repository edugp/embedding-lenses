import logging
from typing import Any, Callable, List, Optional
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st
import umap
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Cividis256 as Pallete
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SEED = 0


@st.cache(show_spinner=False, allow_output_mutation=True)
def load_model(model_name):
    embedder = model_name
    return SentenceTransformer(embedder)


def embed_text(text: List[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(text)


def encode_labels(labels: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(labels):
        return labels
    return labels.astype("category").cat.codes


def get_tsne_embeddings(
    embeddings: np.ndarray, perplexity: int = 30, n_components: int = 2, init: str = "pca", n_iter: int = 5000, random_state: int = SEED
) -> np.ndarray:
    tsne = TSNE(perplexity=perplexity, n_components=n_components, init=init, n_iter=n_iter, random_state=random_state)
    return tsne.fit_transform(embeddings)


def get_umap_embeddings(embeddings: np.ndarray) -> np.ndarray:
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=SEED)
    return umap_model.fit_transform(embeddings)


def draw_interactive_scatter_plot(
    texts: np.ndarray, xs: np.ndarray, ys: np.ndarray, values: np.ndarray, labels: np.ndarray, text_column: str, label_column: str
) -> Any:
    # Normalize values to range between 0-255, to assign a color for each value
    max_value = values.max()
    min_value = values.min()
    if max_value - min_value == 0:
        values_color = np.ones(len(values))
    else:
        values_color = ((values - min_value) / (max_value - min_value) * 255).round().astype(int).astype(str)
    values_color_set = sorted(values_color)

    values_list = values.astype(str).tolist()
    values_set = sorted(values_list)
    labels_list = labels.astype(str).tolist()

    source = ColumnDataSource(data=dict(x=xs, y=ys, text=texts, label=values_list, original_label=labels_list))
    hover = HoverTool(tooltips=[(text_column, "@text{safe}"), (label_column, "@original_label")])
    p = figure(plot_width=800, plot_height=800, tools=[hover], title="Embedding Lenses")
    p.circle("x", "y", size=10, source=source, fill_color=factor_cmap("label", palette=[Pallete[int(id_)] for id_ in values_color_set], factors=values_set))
    return p


def uploaded_file_to_dataframe(uploaded_file: st.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    extension = uploaded_file.name.split(".")[-1]
    return pd.read_csv(uploaded_file, sep="\t" if extension == "tsv" else ",")


def hub_dataset_to_dataframe(path: str, name: str, split: str, text_column: str, label_column: str, sample: int) -> pd.DataFrame:
    load_dataset_fn = partial(load_dataset, path=path)
    if name:
        load_dataset_fn = partial(load_dataset_fn, name=name)
    if split:
        load_dataset_fn = partial(load_dataset_fn, split=split)
    dataset = load_dataset_fn().shuffle(random_state=SEED)[:sample]
    return pd.DataFrame(dataset)


def generate_plot(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    sample: Optional[int],
    dimensionality_reduction_function: Callable,
    model: SentenceTransformer,
):
    logger.info("Loading dataset in memory")
    if text_column not in df.columns:
        raise ValueError("The specified column name doesn't exist")
    if label_column not in df.columns:
        df[label_column] = 0
    df = df.dropna(subset=[text_column, label_column])
    if sample:
        df = df.sample(min(sample, df.shape[0]), random_state=SEED)
    logger.info("Embedding sentences")
    embeddings = embed_text(df[text_column].values.tolist(), model)
    logger.info("Encoding labels")
    encoded_labels = encode_labels(df[label_column])
    logger.info("Running dimensionality reduction")
    embeddings_2d = dimensionality_reduction_function(embeddings)
    logger.info("Generating figure")
    plot = draw_interactive_scatter_plot(
        df[text_column].values, embeddings_2d[:, 0], embeddings_2d[:, 1], encoded_labels.values, df[label_column].values, text_column, label_column
    )
    return plot


st.title("Embedding Lenses")
st.write("Visualize text embeddings in 2D using colors for continuous or categorical labels.")
uploaded_file = st.file_uploader("Choose an csv/tsv file...", type=["csv", "tsv"])
st.write("Alternatively, select a dataset from the hub")
col1, col2, col3 = st.beta_columns(3)
with col1:
    hub_dataset = st.text_input("Dataset name", "ag_news")
with col2:
    hub_dataset_config = st.text_input("Dataset configuration", "")
with col3:
    hub_dataset_split = st.text_input("Dataset split", "train")

text_column = st.text_input("Text column name", "text")
label_column = st.text_input("Numerical/categorical column name (ignore if not applicable)", "label")
sample = st.number_input("Maximum number of documents to use", 1, 100000, 1000)
dimensionality_reduction = st.selectbox("Dimensionality Reduction algorithm", ["UMAP", "t-SNE"], 0)
model_name = st.selectbox("Sentence embedding model", ["distiluse-base-multilingual-cased-v1", "all-mpnet-base-v2"], 0)
model = load_model(model_name)
dimensionality_reduction_function = get_umap_embeddings if dimensionality_reduction == "UMAP" else get_tsne_embeddings

if uploaded_file or hub_dataset:
    if uploaded_file:
        df = uploaded_file_to_dataframe(uploaded_file)
    else:
        df = hub_dataset_to_dataframe(hub_dataset, hub_dataset_config, hub_dataset_split, text_column, label_column, sample)
    plot = generate_plot(df, text_column, label_column, sample, dimensionality_reduction_function, model)
    print(type(plot))
    logger.info("Displaying plot")
    st.bokeh_chart(plot)
    logger.info("Done")
