import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

## work in progress

# LDraw clusters for the entire dataset
data = pd.read_csv('encoded_properties.csv')
X = data.drop(columns=['label', 'sequence'], errors='ignore')

# Dimensionality Reduction using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Dimensionality Reduction using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Clustering using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)

# Clustering using DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)

# Add the reduced dimensions and cluster labels to the original DataFrame
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]
data['tSNE1'] = X_tsne[:, 0]
data['tSNE2'] = X_tsne[:, 1]
data['KMeans_labels'] = kmeans_labels
data['DBSCAN_labels'] = dbscan_labels

# Plotting PCA with KMeans
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='PCA1', y='PCA2', hue='KMeans_labels', data=data, palette='viridis')
plt.title('PCA with KMeans Clustering')

# Plotting t-SNE with DBSCAN
plt.subplot(1, 2, 2)
sns.scatterplot(x='tSNE1', y='tSNE2', hue='DBSCAN_labels', data=data, palette='viridis')
plt.title('t-SNE with DBSCAN Clustering')

plt.show()

# Interactive plots using Plotly
import plotly.express as px

# Interactive PCA plot with KMeans
fig_pca_kmeans = px.scatter(data, x='PCA1', y='PCA2', color='KMeans_labels', hover_data=['Cruciani_1', 'Cruciani_2', 'Cruciani_3', 'Instability_Index', 'Boman_Index', 'Hydrophobicity_Eisenberg', 'Hydrophobic_Moment', 'Aliphatic_Index', 'Isoelectric_Point_Lehninger', 'Charge_pH7.4_Lehninger', 'Freq_Tiny', 'Freq_Small', 'Freq_Aliphatic', 'Freq_Aromatic', 'Freq_Non_polar', 'Freq_Polar', 'Freq_Charged', 'Freq_Basic', 'Freq_Acidic', 'PCA1', 'PCA2', 'tSNE1', 'tSNE2', 'KMeans_labels'])
fig_pca_kmeans.show()

# Interactive t-SNE plot with DBSCAN
fig_tsne_dbscan = px.scatter(data, x='tSNE1', y='tSNE2', color='DBSCAN_labels', hover_data=['Cruciani_1', 'Cruciani_2', 'Cruciani_3', 'Instability_Index', 'Boman_Index', 'Hydrophobicity_Eisenberg', 'Hydrophobic_Moment', 'Aliphatic_Index', 'Isoelectric_Point_Lehninger', 'Charge_pH7.4_Lehninger', 'Freq_Tiny', 'Freq_Small', 'Freq_Aliphatic', 'Freq_Aromatic', 'Freq_Non_polar', 'Freq_Polar', 'Freq_Charged', 'Freq_Basic', 'Freq_Acidic', 'PCA1', 'PCA2', 'tSNE1', 'tSNE2', 'DBSCAN_labels'])
fig_tsne_dbscan.show()


identified_points = data[
    (data['PCA1'] > 10) & (data['PCA2'] > 0.5)  # Adjust the threshold values
]

print('Identified Points:')
print(identified_points)

most_outlying_point_pca = data.iloc[np.argmax(data['PCA1']**2 + data['PCA2']**2)]
print('Most Outlying Point in PCA:')
print(most_outlying_point_pca)

most_outlying_point_tsne = data.iloc[np.argmax(data['tSNE1']**2 + data['tSNE2']**2)]
print('Most Outlying Point in t-SNE:')
print(most_outlying_point_tsne)
