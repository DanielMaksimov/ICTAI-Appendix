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

# ********** This script contains the following functions: **********
# heatmap_building_and_plotting: code for getting the heatmap. The aim is to get the optimal number of topics and user clusters for Bert topic analysis
# semantic_histograms: code for getting the histograms showing the proportion of different semantic labels in each user cluster
# Other utilitary functions whose aim is explained in the function's documentation
# *******************************************************************

def heatmap_building_and_plotting():
    """
    This function builds and displays the heatmap from the article
    """
    
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

    # Plotting the heatmap
    # Cluster range: 2 to 15
    # Topic range: 200 to 5, decreasing by 5
    plt.figure(figsize=(len(topic_range) * 0.5, len(cluster_range) * 0.5))
    sns.heatmap(ssus, xticklabels=topic_range, yticklabels=cluster_range, cmap="viridis", annot=False, square=True)    
    # Labels and title
    plt.xlabel("Number of Topics")
    plt.ylabel("Number of Clusters")
    plt.title("Silhouette Score Heatmap")
    # Show the heatmap
    plt.show()
    

def semantic_histograms():
    filtered_userarr = np.load() #Put the path to the unzipped {filtered_user} file here
    filtered_textarr = np.load() #Put the path to the unzipped {filtered_text} file here
    # The tweets and users were filtered by taking into account only users with more than 6 tweets in the London area

    # Here we fit the BertModel for topic modelling to our data
    docs = filtered_textarr
    representation_model = KeyBERTInspired()
    topic_model = BERTopic(representation_model=representation_model)
    topics, probs = topic_model.fit_transform(docs)

    # We must select a number of topics. We chose 135 topics, and 5 user clusters.
    nb_topics = 135
    topic_model.reduce_topics(docs, nb_top)
    topic_distr, _ = topic_model.approximate_distribution(docs)
    user_vector_dict = normalize_vectors_by_id(topic_distr, filtered_userarr) # Our user vectors for the selected number of topics

    # Clustering our users with defined number of clusters
    nb_clusters = 5
    kmeans = KMeans(n_clusters=nb_clusters, random_state=42, n_init=1)
    vectors = np.array(list(user_vector_dict.values()))
    ids = np.array(list(user_vector_dict.keys()))
    kmeans.fit(vectors)
    labels = kmeans.labels_
    labels_dict = dict(zip(ids, labels))

    # The following file contains the Id of each user and its corresponding GEOGRAPHICAL cluster (among the 8 we found)
    with open("kmeans8_pois.obj",'rb') as file:
        kmeans8_pois = pickle.load(file) 
    label_counts_list_pois = count_labels_in_clusters(kmeans8_pois, labels_dict)
    plot_label_histograms(label_counts_list_pois)


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


def count_labels_in_clusters(clusters, user_labels, normalize=True):
    """
    Counts occurrences of labels (0-4) in each cluster and optionally normalizes to create a distribution.

    :param clusters: List of dictionaries, where each dict contains user IDs in a cluster.
    :param user_labels: Dictionary mapping user IDs to labels (0-5).
    :param normalize: Boolean flag to normalize counts into a probability distribution.
    :return: List of dictionaries, where keys are labels and values are either counts or probabilities.
    """
    label_counts_list = []

    for cluster in clusters:
        label_counts = {i: 0 for i in range(5)}  # Initialize count for labels 0-4
        total_users = 0  # Track total valid users

        for user_id in cluster.keys():
            if user_id in user_labels:
                label_counts[user_labels[user_id]] += 1
                total_users += 1

        # Normalize to make it a probability distribution
        if normalize and total_users > 0:
            label_counts = {label: count / total_users for label, count in label_counts.items()}

        label_counts_list.append(label_counts)

    return label_counts_list


def plot_label_histograms(label_counts_list):
    """
    Plots histograms of label distributions for each cluster in a grid layout (3 columns per row).

    :param label_counts_list: List of dictionaries with normalized label counts per cluster.
    """
    num_clusters = len(label_counts_list)
    num_cols = 3  # Fixed number of columns
    num_rows = math.ceil(num_clusters / num_cols)  # Compute required rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5), sharey=True)

    # Flatten axes array for easier iteration
    axes = axes.flatten()

    for i, (ax, label_counts) in enumerate(zip(axes, label_counts_list)):
        labels = list(label_counts.keys())  # [0, 1, 2, 3, 4, 5]
        counts = list(label_counts.values())

        ax.bar(labels, counts, color="skyblue", edgecolor="black")
        ax.set_title(f"Geographical Cluster {i+1}")
        ax.set_xlabel("Semantic Labels")
        ax.set_xticks(labels)
        ax.set_ylabel("Proportion" if sum(counts) <= 1 else "Count")

    # Hide empty subplots if clusters are not a multiple of num_cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

