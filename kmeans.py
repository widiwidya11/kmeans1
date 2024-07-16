import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Fungsi untuk menghitung elbow method
def calculate_elbow(data):
    sse = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    return k_range, sse

# Fungsi untuk membuat plot elbow method
def plot_elbow(k_range, sse):
    fig, ax = plt.subplots()
    ax.plot(k_range, sse, marker='o')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Sum of squared distances')
    ax.set_title('Elbow Method For Optimal k')
    return fig

# Fungsi untuk membuat scatter plot
def plot_scatter(data, clusters, centers):
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='Nilai Sikap Spiritual', y='Nilai Sikap Sosial', hue=clusters, palette='Set1', ax=ax)
    sns.scatterplot(x=centers[:, 0], y=centers[:, 1], s=200, color='red', ax=ax, label='Centers')
    ax.set_title('Scatter Plot of Clusters')
    return fig

# Fungsi untuk menghitung jarak Euclidean
def calculate_distances(data, centers, labels):
    distances = []
    for i in range(len(data)):
        center = centers[labels[i]]
        distance = np.linalg.norm(data.iloc[i][['Nilai Sikap Spiritual', 'Nilai Sikap Sosial']] - center)
        distances.append(distance)
    return distances

# Fungsi untuk membuat pie chart
def plot_pie(data, clusters):
    fig, ax = plt.subplots()
    cluster_counts = data[clusters].value_counts()
    ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set1'))
    ax.set_title('Cluster Distribution')
    return fig

# Fungsi untuk membuat bar chart dengan keterangan cluster
def plot_bar(data, clusters):
    fig, ax = plt.subplots()
    cluster_counts = data[clusters].value_counts().sort_index()
    labels = [f"Cluster {i}" for i in cluster_counts.index]
    ax.bar(labels, cluster_counts, color=sns.color_palette('Set1'))
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Number of Data Points')
    ax.set_title('Number of Data Points per Cluster')
    return fig

st.title("K-Means Clustering Web App")
st.header("Website Untuk Mengelompokkan Karakter Siswa SD Al Amanah Berdasarkan Nilai Sikap Spiritual dan Nilai Sikap Sosial")
#st.markdown("Nilai Sikap Spiritual Yaitu Menerima, Menjalankan, dan Menghargai Ajaran Agama Islam")
#st.markdown("Nilai Sikap Sosial Yaitu Jujur, Disiplin, Tanggung Jawab, Santun, Peduli, Percaya Diri, Kerjasama antar Teman dan Guru, serta Cinta Tanah Air")

st.header("Upload your data")
uploaded_file = st.file_uploader("Upload File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(data)

    k_range, sse = calculate_elbow(data[['Nilai Sikap Spiritual', 'Nilai Sikap Sosial']])
    elbow_fig = plot_elbow(k_range, sse)
    st.subheader("Elbow Method")
    st.pyplot(elbow_fig)

    optimal_k = st.sidebar.slider('Select number of clusters (k)', min_value=1, max_value=10, value=3)
    kmeans = KMeans(n_clusters=optimal_k)
    labels = kmeans.fit_predict(data[['Nilai Sikap Spiritual', 'Nilai Sikap Sosial']])
    data['Cluster'] = labels
    centers = kmeans.cluster_centers_


    scatter_fig = plot_scatter(data, 'Cluster', centers)
    st.subheader("Scatter Plot")
    st.pyplot(scatter_fig)
    st.markdown("Cluster 0 = Sangat Baik, Cluster 1 = Cukup, Cluster 2 = Baik ")
    
    st.subheader("Cluster Centers")
    st.write(pd.DataFrame(centers, columns=['Nilai Sikap Spiritual', 'Nilai Sikap Sosial']))

    st.subheader("Clustered Data with Distances")
    st.write(data)
    
    
    bar_fig = plot_bar(data, 'Cluster')
    st.subheader("Bar Chart")
    st.pyplot(bar_fig)
    
    pie_fig = plot_pie(data, 'Cluster')
    st.subheader("Pie Chart")
    st.pyplot(pie_fig)

    silhouette_avg = silhouette_score(data[['Nilai Sikap Spiritual', 'Nilai Sikap Sosial']], labels)
    davies_bouldin_avg = davies_bouldin_score(data[['Nilai Sikap Spiritual', 'Nilai Sikap Sosial']], labels)
    distances = calculate_distances(data, centers, labels)
    data['Euclidean Distance'] = distances

    st.subheader("Cluster Metrics")
    st.write(f"Silhouette Score: {silhouette_avg}")
    st.write(f"Davies-Bouldin Score: {davies_bouldin_avg}")

    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download clustered data as CSV",
        data=csv,
        file_name='clustered_data.csv',
        mime='text/csv',
    )
