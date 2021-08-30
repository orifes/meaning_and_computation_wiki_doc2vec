from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl

# Loading the set of words and vectors with both document wnd word vectors
with open("words_we_have.pkl", "rb") as w:
    words = pkl.load(w)

with open("w2v_vecs_we_have.pkl", "rb") as w:
    word_vecs = pkl.load(w)

with open("d2v_vecs_we_have.pkl", "rb") as w:
    doc_vecs = pkl.load(w)

# Getting the categories with more than 5 words
categories_df = pd.read_csv("categories.txt", encoding="utf-8", delimiter="\t")
categories_with_counts = list(categories_df.category.value_counts()[categories_df.category.value_counts() > 5].index)
words_in_categories = categories_df[categories_df.category.apply(lambda x: x in categories_with_counts)]

# Extracting the vectors of the word in those categories
word_indices = words_in_categories.word.apply(lambda x: words.index(x))
words_to_cluster = []
doc_vecs_to_cluster = []
word_vecs_to_cluster = []
for ind in word_indices:
    words_to_cluster.append(words[ind])
    word_vecs_to_cluster.append(word_vecs[ind])
    doc_vecs_to_cluster.append(doc_vecs[ind])

# Training the K-means models and saving the results
w2v_kmeans = KMeans(n_clusters=50)
w2v_clusters = w2v_kmeans.fit(word_vecs_to_cluster).labels_
d2v_kmeans = KMeans(n_clusters=50)
d2v_clusters = d2v_kmeans.fit(doc_vecs_to_cluster).labels_

# Adding the clusters to the df
words_in_categories["w2v_cluster"] = w2v_clusters
words_in_categories["d2v_cluster"] = d2v_clusters

# Calculating the average number of different k-means clusters assigned to the Wikipedia categories
w2v_clusters_per_cat = 0
d2v_clusters_per_cat = 0

for name, cat_group in words_in_categories.groupby("category"):
    w2v_clusters_per_cat += cat_group.w2v_cluster.value_counts().size * cat_group.w2v_cluster.value_counts().sum()
    d2v_clusters_per_cat += cat_group.d2v_cluster.value_counts().size * cat_group.d2v_cluster.value_counts().sum()

w2v_clusters_per_cat /= words_in_categories.shape[0]
d2v_clusters_per_cat /= words_in_categories.shape[0]
print(d2v_clusters_per_cat, w2v_clusters_per_cat)

# Mapping the two type of vector to two dimensions using PCA
w2v_pca = PCA(n_components=2)
w2v_comp = w2v_pca.fit_transform(word_vecs_to_cluster)
d2v_pca = PCA(n_components=2)
d2v_comp = d2v_pca.fit_transform(doc_vecs_to_cluster)

words_in_categories["w2v_pca1"] = w2v_comp[:, 0]
words_in_categories["w2v_pca2"] = w2v_comp[:, 1]
words_in_categories["d2v_pca1"] = d2v_comp[:, 0]
words_in_categories["d2v_pca2"] = d2v_comp[:, 1]

# Plotting the random category combination we got to visualize the latent space of both models
cats_to_plot = pd.concat([words_in_categories[words_in_categories.category == "ball games"],
                          words_in_categories[words_in_categories["category"] == "chinese inventions"],
                          words_in_categories[words_in_categories["category"] == "christian terminology"]])

scatter = plt.scatter(cats_to_plot.w2v_pca1, cats_to_plot.w2v_pca2,
                      c=cats_to_plot.category.astype("category").cat.codes, alpha=0.5, s=40)
plt.xlabel("word2vec PCA1")
plt.ylabel("word2vec PCA2")

plt.legend(handles=scatter.legend_elements()[0], labels=['ball games', 'chinese inventions', 'christian terminology'],
           bbox_to_anchor=(1.5, 1))
plt.title("Word2vec vectors PCA mapping by category")
plt.savefig("PCA_word2vec.png", size=(400, 400), bbox_inches="tight")


scatter = plt.scatter(cats_to_plot.d2v_pca1, cats_to_plot.d2v_pca2,
                      c=cats_to_plot.category.astype("category").cat.codes, alpha=0.5, s=40)
plt.xlabel("doc2vec PCA1")
plt.ylabel("doc2vec PCA2")
plt.legend(handles=scatter.legend_elements()[0], labels=['ball games', 'chinese inventions', 'christian terminology'],
           bbox_to_anchor=(1.5, 1))
plt.title("Doc2vec vectors PCA mapping by category")
plt.savefig("PCA_doc2vec.png", bbox_inches="tight")
