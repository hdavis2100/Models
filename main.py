from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import numpy as np

scale = "standard" if len(sys.argv) > 1 and sys.argv[1] == "standard" else "none"
extraction = True if len(sys.argv) > 2 and sys.argv[2] == "extract" else False
letter_recognition = fetch_ucirepo(id=59) 
X = letter_recognition.data.features 
y = letter_recognition.data.targets 

if scale == "standard":
    scaler = StandardScaler()
    samples = scaler.fit_transform(X)
else:
    samples = X.to_numpy()

if extraction:
    samples = PCA(n_components=2).fit_transform(samples)



kmeans = KMeans(n_clusters=26, random_state=0, n_init=50, max_iter=300).fit(samples)

df = X.join(y)

d = {i: [0] * 26 for i in range(26)}
for i in range(len(samples)):
    cluster = kmeans.predict([samples[i]])
    actual = ord(y.iloc[i,0]) - ord('A')
    d[cluster[0]][actual] += 1

purity = sum(max(counts) for counts in d.values()) / len(samples)

print("Purity:", purity)

clusterLabels = kmeans.labels_
centers = kmeans.cluster_centers_

if extraction:
    plt.scatter(
    samples[:, 0],
    samples[:, 1],
    c=clusterLabels,
    s=5,
    alpha=0.5
    )
    plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200,
    edgecolor="k"
    )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("KMeans clusters (k=26) in PCA(2D) space")
    plt.tight_layout()
    plt.savefig("kmeans_letter_recognition_pca.png")

    


  

