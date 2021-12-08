from functools import partial
from typing import Optional

import pandas as pd
import streamlit as st
from datasets import load_dataset


def uploaded_file_to_dataframe(uploaded_file: st.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    extension = uploaded_file.name.split(".")[-1]
    return pd.read_csv(uploaded_file, sep="\t" if extension == "tsv" else ",")


def hub_dataset_to_dataframe(
    path: str, name: Optional[str], split: Optional[str], sample: int, seed: int = 0, data_files: Optional[str] = None
) -> pd.DataFrame:
    load_dataset_fn = partial(load_dataset, path=path)
    if name:
        load_dataset_fn = partial(load_dataset_fn, name=name)
    if split:
        load_dataset_fn = partial(load_dataset_fn, split=split)
    if data_files:
        load_dataset_fn = partial(load_dataset_fn, data_files=data_files)
    dataset = load_dataset_fn().shuffle(seed=seed)[:sample]
    return pd.DataFrame(dataset)
