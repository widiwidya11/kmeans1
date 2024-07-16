import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Fungsi untuk mengunggah dataset
def upload_dataset():
    uploaded_file = st.file_uploader("Upload dataset Anda (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None