from functools import partial

import pandas as pd
import streamlit as st
from datasets import load_dataset


def uploaded_file_to_dataframe(uploaded_file: st.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    extension = uploaded_file.name.split(".")[-1]
    return pd.read_csv(uploaded_file, sep="\t" if extension == "tsv" else ",")


def hub_dataset_to_dataframe(path: str, name: str, split: str, sample: int, seed: int = 0) -> pd.DataFrame:
    load_dataset_fn = partial(load_dataset, path=path)
    if name:
        load_dataset_fn = partial(load_dataset_fn, name=name)
    if split:
        load_dataset_fn = partial(load_dataset_fn, split=split)
    dataset = load_dataset_fn().shuffle(seed=seed)[:sample]
    return pd.DataFrame(dataset)
