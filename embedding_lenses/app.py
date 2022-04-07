import logging
from functools import partial
from typing import Callable, Optional

import pandas as pd
import streamlit as st
from bokeh.plotting import Figure
from sentence_transformers import SentenceTransformer

from embedding_lenses.data import hub_dataset_to_dataframe, uploaded_file_to_dataframe
from embedding_lenses.dimensionality_reduction import get_tsne_embeddings, get_umap_embeddings
from embedding_lenses.embedding import embed_text, load_model
from embedding_lenses.utils import encode_labels
from embedding_lenses.visualization import draw_interactive_scatter_plot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
EMBEDDING_MODELS = ["distiluse-base-multilingual-cased-v1", "all-mpnet-base-v2", "flax-sentence-embeddings/all_datasets_v3_mpnet-base"]
DIMENSIONALITY_REDUCTION_ALGORITHMS = ["UMAP", "t-SNE"]
SEED = 0


def generate_plot(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    sample: Optional[int],
    dimensionality_reduction_function: Callable,
    model: SentenceTransformer,
) -> Figure:
    if text_column not in df.columns:
        raise ValueError(f"The specified column name doesn't exist. Columns available: {df.columns.values}")
    label_column_exists = True
    if label_column not in df.columns:
        df[label_column] = 0
        label_column_exists = False
    df = df.dropna(subset=[text_column, label_column])
    if sample:
        df = df.sample(min(sample, df.shape[0]), random_state=SEED)
    with st.spinner(text="Embedding text..."):
        embeddings = embed_text(df[text_column].values.tolist(), model)
    logger.info("Encoding labels")
    encoded_labels = encode_labels(df[label_column])
    with st.spinner("Reducing dimensionality..."):
        embeddings_2d = dimensionality_reduction_function(embeddings)
    logger.info("Generating figure")
    hover_data = {text_column: df[text_column].values}
    if label_column_exists:
        hover_data[label_column] = df[label_column].values
    plot = draw_interactive_scatter_plot(
        hover_data,
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        encoded_labels.values,
    )
    return plot


def app():
    st.title("Embedding Lenses")
    st.write("Visualize text embeddings in 2D using colors for continuous or categorical labels.")
    uploaded_file = st.file_uploader("Choose an csv/tsv file...", type=["csv", "tsv"])
    st.write("Alternatively, select a dataset from the [hub](https://huggingface.co/datasets)")
    col1, col2, col3 = st.columns(3)
    with col1:
        hub_dataset = st.text_input("Dataset name", "ag_news")
    with col2:
        hub_dataset_config = st.text_input("Dataset configuration", "")
    with col3:
        hub_dataset_split = st.text_input("Dataset split", "train")

    text_column = st.text_input("Text column name", "text")
    label_column = st.text_input("Numerical/categorical column name (ignore if not applicable)", "label")
    sample = st.number_input("Maximum number of documents to use", 1, 100000, 1000)
    dimensionality_reduction = st.selectbox("Dimensionality Reduction algorithm", DIMENSIONALITY_REDUCTION_ALGORITHMS, 0)
    model_name = st.selectbox("Sentence embedding model", EMBEDDING_MODELS, 0)
    with st.spinner(text="Loading model..."):
        model = load_model(model_name)
    dimensionality_reduction_function = (
        partial(get_umap_embeddings, random_state=SEED) if dimensionality_reduction == "UMAP" else partial(get_tsne_embeddings, random_state=SEED)
    )

    if uploaded_file or hub_dataset:
        with st.spinner("Loading dataset..."):
            if uploaded_file:
                df = uploaded_file_to_dataframe(uploaded_file)
            else:
                df = hub_dataset_to_dataframe(hub_dataset, hub_dataset_config, hub_dataset_split, sample, seed=SEED)
        plot = generate_plot(df, text_column, label_column, sample, dimensionality_reduction_function, model)
        logger.info("Displaying plot")
        st.bokeh_chart(plot)
        logger.info("Done")


if __name__ == "__main__":
    app()
