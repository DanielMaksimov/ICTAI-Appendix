from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import pickle
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

# ********** This script is for building the heatmap necessary to get the optimal number of topics and user clusters for Bert topic analysis *******************


def normalize_vectors_by_id(L1, L2):
    """
    Computes the normalized sum of vectors in L1 grouped by corresponding IDs in L2.

    Parameters:
        L1 (list of lists or numpy arrays): List of vectors.
        L2 (list): List of IDs corresponding to each vector in L1.

    Returns:
        dict: Keys are unique IDs from L2, values are the normalized sum of vectors.
    """
    vector_sums = defaultdict(lambda: np.zeros_like(L1[0]))
    count = defaultdict(int)
    for vector, id_ in zip(L1, L2):
        vector_sums[id_] += np.array(vector)
        count[id_] += 1
      
    result = {id_: vector_sums[id_] / count[id_] for id_ in vector_sums}
    return result

if __name__ == "__main__":
    filtered_userarr = np.load() #Put the path to the unzipped {filtered_user} file here
    filtered_textarr = np.load() #Put the path to the unzipped {filtered_text} file here
    # The tweets and users were filtered by taking into account only users with more than 6 tweets in the London area

    # Here we fit the BertModel for topic modelling to our data
    docs = filtered_textarr
    representation_model = KeyBERTInspired()
    topic_model = BERTopic(representation_model=representation_model)
    topics, probs = topic_model.fit_transform(docs)

    # We proceed to build the heatmap to determine optimal cluster number
    cluster_range = list(range(2, 16))  # Testing cluster numbers from 2 to 15
    topic_range = list(range(200, 4, -5))  # Testing topics from 200 to 5 in steps of 5

    silhouette_scores_tw = np.zeros((len(cluster_range), len(topic_range)))
    silhouette_scores_us = np.zeros((len(cluster_range), len(topic_range)))

    for j, nb_top in enumerate(topic_range):
        topic_model.reduce_topics(docs, nb_top)
        topic_distr, _ = topic_model.approximate_distribution(docs)
        user_vector_dict = normalize_vectors_by_id(topic_distr, filtered_userarr)

        for i, n_clusters in enumerate(cluster_range):
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=1)
            vectors = np.array(list(user_vector_dict.values()))
            labels = kmeans.fit_predict(vectors)
            us_score = silhouette_score(vectors, labels) # Calculating silhouette score for users
            silhouette_scores_us[i, j] = us_score

        ssus = np.array(silhouette_scores_us) #This is our heatmap of silhouette score
        np.save("my/path/to/heatmap_file.npy", ssus)
