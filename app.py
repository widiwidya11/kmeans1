import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


st.title("K-Means Clustering Web App")
st.header("Website Untuk Mengelompokkan Karakter Siswa SDI Al Amanah Berdasarkan Nilai Sikap Spiritual dan Nilai Sikap Sosial")


