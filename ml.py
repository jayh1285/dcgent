# Author:   Jay Huang <askjayhuang@gmail.com>
# Created:  2017-11-12T01:52:51.908Z

"""A module for machine learning."""

################################################################################
# Imports
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from yellowbrick.cluster import KElbowVisualizer
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import readcsv
import time
import pickle
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score

################################################################################
# Functions
################################################################################


def kmeans(df, n_components=2, v=False):
    """K-Means clustering on PCA-reduced data with silhouette scores."""

    print('K-Means clustering on PCA-reduced (' + str(n_components) + ' components) ' + 'data')

    # Standardize data
    df_tr = StandardScaler().fit_transform(df)

    # Plot explained variance ratio graph
    pca = PCA().fit(df_tr)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    # Reduce features via PCA
    pca = PCA(n_components=n_components).fit(df_tr)
    reduced_data = pca.transform(df_tr)

    # # Plot PCA components composition
    # plt.matshow(pca.components_, cmap='viridis')
    # plt.yticks([0, 1], ["First component", "Second component"])
    # plt.colorbar()
    # plt.xticks(range(len(df.columns)),
    #            df.columns, rotation=60, ha='left')
    # plt.xlabel("Feature")
    # plt.ylabel("Principal components")
    # plt.show()

    silhouette_scores = []

    range_n_clusters = range(2, 76)
    for n_clusters in range_n_clusters:
        # Initialize the clusterer
        model = KMeans(n_clusters=n_clusters)
        model.fit(reduced_data)
        cluster_labels = model.predict(reduced_data)

        # The silhouette_score gives the average value for all the samples.
        silhouette_avg = silhouette_score(reduced_data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        silhouette_scores.append(silhouette_avg)

        if n_clusters == 36:
            # vis(model, reduced_data, cluster_labels, n_components, n_clusters, silhouette_avg)
            dfq['Cluster Labels'] = cluster_labels

        # # TSNE visualization
        # if n_clusters == 20:
        #     X_embedded = TSNE(n_components=2).fit_transform(reduced_data)
        #     colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #     plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker='.', s=30, lw=0, alpha=0.7,
        #                 c=colors, edgecolor='k')
        #
        #     # Labeling the clusters
        #     centers = model.cluster_centers_
        #     # Draw white circles at cluster centers
        #     plt.scatter(centers[:, 0], centers[:, 1], marker='o',
        #                 c="white", alpha=1, s=200, edgecolor='k')
        #
        #     for i, c in enumerate(centers):
        #         plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
        #                     s=50, edgecolor='k')
        #
        #     plt.show()

    # Plot elbow graph
    plt.plot(range_n_clusters, silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('The Elbow Method showing the optimal number of clusters for ' + str(n_components) + ' n_components')
    plt.show()


def kmeans_elbow(df):
    # Standardize data
    df_tr = StandardScaler().fit_transform(df)

    # # Plot explained variance ratio graph
    # pca = PCA().fit(df_tr)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of components')
    # plt.ylabel('Cumulative explained variance')
    # plt.show()

    # Initialize elbow graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initalize range of n_components
    range_n_components = range(2, 15)
    silhouette_scores_all = []

    # Iterate through range of n_components
    for n_components in tqdm(range_n_components):
        # Reduce features via PCA
        pca = PCA(n_components=n_components).fit(df_tr)
        reduced_data = pca.transform(df_tr)

        # Initalize silhouette scores array for a specific n_component
        silhouette_scores = []

        # Iterate through range of n_clusters
        range_n_clusters = range(2, 40)
        for n_clusters in tqdm(range_n_clusters):
            # Initialize clusterer and fit/predict data
            model = KMeans(n_clusters=n_clusters)
            model.fit(reduced_data)
            cluster_labels = model.predict(reduced_data)

            # The silhouette_score gives the average value for all the samples.
            silhouette_avg = silhouette_score(reduced_data, cluster_labels)

            # Append silhouette score
            silhouette_scores.append(silhouette_avg)

        silhouette_scores_all.append(silhouette_scores)
        xs = range_n_clusters
        ys = silhouette_scores
        zs = n_components
        ax.plot(xs, ys, zs, 'bx-')

    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_zlabel('N_Components in PCA Reduction')
    plt.title('The Elbow Method Showing the Optimal Number of Clusters and N_Components in PDA Reduction')
    plt.show()


def vis(model, reduced_data, cluster_labels, n_components, n_clusters, silhouette_avg):
    sns.set()

    # Create a subplot based on n_components
    if n_components == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2)
    elif n_components == 3:
        fig = plt.figure()
        ax1 = fig.add_subplot(121, sharex=None, sharey=None)
        ax2 = fig.add_subplot(122, sharex=None, sharey=None, projection='3d')

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(reduced_data) + (n_clusters + 1) * 10])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(reduced_data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # if n_components == 2:
    #     # Create a meshgrid to visualize the decision boundary
    #     h = .01
    #     x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    #     y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    #     # Obtain labels for meshgrid
    #     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    #
    #     # Plot decision boundary
    #     Z = Z.reshape(xx.shape)
    #     ax2.imshow(Z, interpolation='nearest',
    #                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #                cmap=plt.cm.Paired,
    #                aspect='auto', origin='lower')

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    if n_components == 2:
        ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
    elif n_components == 3:
        ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

    # Labeling the clusters
    centers = model.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st PCA component")
    ax2.set_ylabel("Feature space for the 2nd PCA component")
    if n_components == 3:
        ax2.set_zlabel("Feature space for the 3rd PCA component")

    plt.suptitle(("Silhouette analysis for clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()


def gmm(df, n_components=2, v=False):
    """GMM clustering on PCA-reduced data with silhouette scores."""

    print('GMM clustering on PCA-reduced (' + str(n_components) + ' components) ' + 'data')

    # Standardize data
    df_tr = StandardScaler().fit_transform(df)

    # Plot explained variance ratio graph
    pca = PCA().fit(df_tr)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    # Reduce features via PCA
    pca = PCA(n_components=n_components).fit(df_tr)
    reduced_data = pca.transform(df_tr)

    # Plot PCA components composition
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(df.columns)),
               df.columns, rotation=60, ha='left')
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.show()

    covariances = ['full', 'tied', 'diag', 'spherical']
    # covariances = ['tied']
    for covariance in covariances:
        print('\nCovariance type: ' + covariance)

        range_n_clusters = range(2, 40)
        for n_clusters in range_n_clusters:
            # Initialize the clusterer
            model = GaussianMixture(n_components=n_clusters, covariance_type=covariance)
            model.fit(reduced_data)
            cluster_labels = model.predict(reduced_data)

            # The silhouette_score gives the average value for all the samples.
            silhouette_avg = silhouette_score(reduced_data, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            if v:
                vis_prob(reduced_data, cluster_labels, model.predict_proba(
                    reduced_data), n_components, n_clusters, silhouette_avg)


def vis_prob(reduced_data, cluster_labels, probs, n_components, n_clusters, silhouette_avg):
    sns.set()

    # Create a subplot based on components
    if n_components == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2)
    elif n_components == 3:
        fig = plt.figure()
        ax1 = fig.add_subplot(121, sharex=None, sharey=None)
        ax2 = fig.add_subplot(122, sharex=None, sharey=None, projection='3d')

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(reduced_data) + (n_clusters + 1) * 10])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(reduced_data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    size = 50 * probs.max(1) ** 2
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    if n_components == 2:
        ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.', s=size, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
    elif n_components == 3:
        ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], marker='.', s=size, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st PCA component")
    ax2.set_ylabel("Feature space for the 2nd PCA component")
    if n_components == 3:
        ax2.set_zlabel("Feature space for the 3rd PCA component")

    plt.suptitle(("Silhouette analysis for clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()


def spectral(df, n_components=2):
    """Spectral clustering on PCA-reduced data with silhouette scores."""

    print('Spectral clustering on PCA-reduced data')

    # Standardize data
    df_tr = StandardScaler().fit_transform(df)

    # Plot explained variance ratio graph
    pca = PCA().fit(df_tr)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    # Reduce features via PCA
    pca = PCA(n_components=n_components).fit(df_tr)
    reduced_data = pca.transform(df_tr)

    # Plot PCA components composition
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(df.columns)),
               df.columns, rotation=60, ha='left')
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.show()

    range_n_clusters = range(2, 25)
    for n_clusters in range_n_clusters:
        # Initialize the clusterer
        model = SpectralClustering(n_clusters=n_clusters)
        cluster_labels = model.fit_predict(reduced_data)

        # The silhouette_score gives the average value for all the samples.
        silhouette_avg = silhouette_score(reduced_data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)


def affinity(df, n_components=2):
    """Affinity Propagation clustering on PCA-reduced data with silhouette scores."""

    print('Affinity Propagation clustering on PCA-reduced (' +
          str(n_components) + ' components) ' + 'data')

    # Standardize data
    df_tr = StandardScaler().fit_transform(df)

    # Plot explained variance ratio graph
    pca = PCA().fit(df_tr)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    # Reduce features via PCA
    pca = PCA(n_components=n_components).fit(df_tr)
    reduced_data = pca.transform(df_tr)

    # Plot PCA components composition
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(df.columns)),
               df.columns, rotation=60, ha='left')
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.show()

    # Initialize the clusterer
    model = AffinityPropagation()
    cluster_labels = model.fit_predict(reduced_data)

    # The silhouette_score gives the average value for all the samples.
    silhouette_avg = silhouette_score(reduced_data, cluster_labels)
    print("The average silhouette_score is :", silhouette_avg)


def agglomerative(df, n_components=2, v=False):
    """Agglomerative clustering on PCA-reduced data with silhouette scores."""

    print('Agglomerative clustering on PCA-reduced data')

    # Standardize data
    df_tr = StandardScaler().fit_transform(df)

    # Plot explained variance ratio graph
    pca = PCA().fit(df_tr)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    # Reduce features via PCA
    pca = PCA(n_components=n_components).fit(df_tr)
    reduced_data = pca.transform(df_tr)

    # Plot PCA components composition
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(df.columns)),
               df.columns, rotation=60, ha='left')
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.show()

    # linkages = ['ward', 'complete', 'average']
    linkages = ['ward']
    for linkage in linkages:
        print('Linkage: ' + linkage)

        range_n_clusters = range(2, 40)
        for n_clusters in range_n_clusters:
            # Initialize the clusterer
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            cluster_labels = model.fit_predict(reduced_data)

            # The silhouette_score gives the average value for all the samples.
            silhouette_avg = silhouette_score(reduced_data, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            if v and n_clusters == 20:
                vis(model, reduced_data, cluster_labels, n_components, n_clusters, silhouette_avg)


def dbscan(df, n_components=2):
    """Affinity Propagation clustering on PCA-reduced data with silhouette scores."""

    print('Affinity Propagation clustering on PCA-reduced (' +
          str(n_components) + ' components) ' + 'data')

    # Standardize data
    df_tr = StandardScaler().fit_transform(df)

    # Plot explained variance ratio graph
    pca = PCA().fit(df_tr)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    # Reduce features via PCA
    pca = PCA(n_components=n_components).fit(df_tr)
    reduced_data = pca.transform(df_tr)

    # Plot PCA components composition
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(df.columns)),
               df.columns, rotation=60, ha='left')
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.show()

    # Initialize the clusterer
    model = DBSCAN()
    cluster_labels = model.fit_predict(reduced_data)

    # The silhouette_score gives the average value for all the samples.
    silhouette_avg = silhouette_score(reduced_data, cluster_labels)
    print("The average silhouette_score is :", silhouette_avg)


def birch(df, n_components=2):
    """Agglomerative clustering on PCA-reduced data with silhouette scores."""

    print('Agglomerative clustering on PCA-reduced data')

    # Standardize data
    df_tr = StandardScaler().fit_transform(df)

    # Plot explained variance ratio graph
    pca = PCA().fit(df_tr)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    # Reduce features via PCA
    pca = PCA(n_components=n_components).fit(df_tr)
    reduced_data = pca.transform(df_tr)

    # Plot PCA components composition
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(df.columns)),
               df.columns, rotation=60, ha='left')
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.show()

    range_n_clusters = range(2, 40)
    for n_clusters in range_n_clusters:
        # Initialize the clusterer
        model = Birch(n_clusters=n_clusters)
        cluster_labels = model.fit_predict(reduced_data)

        # The silhouette_score gives the average value for all the samples.
        silhouette_avg = silhouette_score(reduced_data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

################################################################################
# Execution
################################################################################


if __name__ == '__main__':
    sns.set()

    df = pickle.load(open("data/ml.p", "rb"))
    dfq = pickle.load(open("data/q.p", "rb"))

    kmeans(df, n_components=5)
    # kmeans_elbow(df)
    # gmm(df)
    # spectral(df)
    # affinity(df)
    # agglomerative(df, v=True)
    # birch(df)

    meanq = pd.Series()

    for cluster in range(1, 37):
        mask = dfq['Cluster Labels'] == cluster
        mask_index = dfq[mask].index

        dfq.loc[mask_index, 'Mean Q-Scores'] = dfq[mask]['Total Q-Score'].mean()
        print('Cluster ' + str(cluster) + ': ' + str(dfq[mask]['Total Q-Score'].mean()))

        meanq = meanq.set_value('Cluster ' + str(cluster), dfq[mask]['Total Q-Score'].mean())
        # meanq.append(dfq[mask]['Total Q-Score'].mean())
