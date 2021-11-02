from typing import List

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache(show_spinner=False, allow_output_mutation=True)
def load_model(model_name: str) -> SentenceTransformer:
    embedder = model_name
    return SentenceTransformer(embedder)


def embed_text(text: List[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(text)
