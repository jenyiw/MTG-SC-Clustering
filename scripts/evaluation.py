import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay, auc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def dunn_index(cluster_labels, data, cluster_centroids):
    '''
    cluster_labels: a N x 1 array denoting the cluster label of each data point
    data: a N x 20 matrix of reduced data
    cluster_centroids: a K x 20 matrix of cluster centroids (this isn't used in the calculation, but we can change the formula to use it if computation time is too long.)

    1. Find the highest intra-cluster distance i.e. the largest Euclidean distance between 2 data points in the same cluster.
    2. Find the lowest inter-cluster distance i.e.
    3. DI = intra-cluster distance / inter-cluster distance
    4. return DI
    '''

    # Step 1: Calculate intra-cluster distances
    num_clusters = cluster_centroids.shape[0]
    intra_cluster_distance = 0.0
    for i in range(num_clusters):
        # find indices of data points lying in this cluster
        cluster_indices = np.where(cluster_labels == i)
        # retrieve the data points using the indices
        current_cluster = data[cluster_indices, :][0]
        # find the largest distance within that cluster
        distance = find_largest_distance(current_cluster)
        # if the largest distance is bigger than what has been found so far, replace intra_cluster_distance with that value
        if distance > intra_cluster_distance:
            intra_cluster_distance = distance

    # Step 2: Calculate inter-cluster distances
    inter_cluster_distance = np.inf
    for first in range(num_clusters):
        for second in range(first, num_clusters):
            # range over every pair of clusters. There will be 0.5 x K x (K-1) pairings
            if first == second:
                # first and second cluster should not be the same
                continue
            else:
                # find indices of data points lying in both clusters
                first_cluster_indices = np.where(cluster_labels == first)
                second_cluster_indices = np.where(cluster_labels == second)
                # retrieve the data points using the indices
                first_cluster = data[first_cluster_indices, :][0]
                second_cluster = data[second_cluster_indices, :][0]
                # find the smallest distance between both clusters
                distance = find_smallest_distance(
                    first_cluster, second_cluster)
                # if the smallest distance is smaller than what has been found so far, replace inter_cluster_distance with that value
                if distance < inter_cluster_distance:
                    inter_cluster_distance = distance

    # Step 3: Calculate Dunn Index
    dunn = intra_cluster_distance / inter_cluster_distance
    return round(dunn, 3)


def find_largest_distance(cluster):
    '''
    cluster: a k x 20 matrix containing k data points from a single cluster
    Among all pairs of points in a cluster, find largest Euclidean distance between points in a pair.
    '''
    num_points = cluster.shape[0]
    largest_distance = 0.0
    for first in range(num_points):
        for second in range(first, num_points):
            # range over every pair of points in the cluster
            if first == second:
                # points should not be the same
                continue
            else:
                # calculate Euclidean distance
                distance = np.linalg.norm(
                    cluster[first, :] - cluster[second, :])
                # if the largest distance is bigger than what has been found so far, replace largest_distance with that value
                if distance > largest_distance:
                    largest_distance = distance
    return largest_distance


def find_smallest_distance(cluster_1, cluster_2):
    '''
    cluster_1: a k x 20 matrix containing k data points from one cluster
    cluster_2: a j x 20 matrix containing j data points from another cluster
    Among all pairs of points between one cluster and another, find smallest Euclidean distance between points in a pair.
    '''
    num_points_1 = cluster_1.shape[0]
    num_points_2 = cluster_2.shape[0]
    smallest_distance = np.inf

    for first in range(num_points_1):
        for second in range(num_points_2):
            # calculate Euclidean distance
            distance = np.linalg.norm(
                cluster_1[first, :] - cluster_2[second, :])
            # if the smallest distance is smaller than what has been found so far, replace smallest_distance with that value
            if distance < smallest_distance:
                smallest_distance = distance
    return smallest_distance


def silhouette_coefficient(cluster_labels, data, cluster_centroids):
    '''
    cluster_labels: a N x 1 array denoting the cluster label of each data point
    data: a N x 20 matrix of reduced data
    cluster_centroids: a K x 20 matrix of cluster centroids

    1. Calculate silhouette_value for each data point in each cluster
    2. Take the average silhouette_value for each cluster
    3. Choose largest average silhouette_value as silhouette coefficient
    '''
    num_points = cluster_labels.shape[0]
    num_clusters = cluster_centroids.shape[0]
    cluster_average = np.zeros(num_clusters)
    data_silhouette_values = np.zeros(num_points)

    for point in range(num_points):
        data_point = data[point, :]
        # determine which cluster the point belongs to
        cluster_label = cluster_labels[point]
        # find the Euclidean distance between the data point and its cluster centre
        a_value = np.linalg.norm(
            data_point - cluster_centroids[cluster_label, :])
        # find the smallest Euclidean distance between the data point and all other cluster centres
        b_value = find_closest_centroid(
            data_point, cluster_label, cluster_centroids)
        # calculate silhouette value
        silhouette_value = (b_value - a_value) / max(a_value, b_value)
        # store the value for calculating average later
        data_silhouette_values[point] = silhouette_value

    for cluster in range(num_clusters):
        # find indices of data points lying in this cluster
        cluster_indices = np.where(cluster_labels == cluster)
        # retrieve the silhouette values using the indices
        current_cluster = data_silhouette_values[cluster_indices]
        # Calculate average
        average_SC = np.sum(current_cluster) / current_cluster.shape[0]
        # store the average SC of the cluster
        cluster_average[cluster] = average_SC
    return np.max(cluster_average)


def find_closest_centroid(point, cluster_label, cluster_centroids):
    '''
    point: 1 x 20 array of single data point
    cluster_label: int, label of current point's cluster
    cluster_centroids: k x 20 matrix of centres of every cluster
    '''
    num_clusters = cluster_centroids.shape[0]
    closest_distance = np.inf
    for cluster in range(num_clusters):
        if cluster == cluster_label:
            continue
        else:
            distance = np.linalg.norm(point - cluster_centroids[cluster, :])
            if distance < closest_distance:
                closest_distance = distance
    return closest_distance


def plot_evaluation_metric(array, labels, mode, cluster_method):
    '''
    Array: Given multiple clusterings, find Dunn Index (DI) or Silhoutte Coefficient (SC) of each clustering and store the value for each clustering in a numpy array.
    Labels: Also provide a list of labels specifying what each clustering algorithm is. This could be abbreviated names. The function will output a bar plot of DI/ SC across clustering algorithms.
    Mode: Specify "Dunn Index" or "Silhouette Coefficient" for graph titles
    cluster_method: Specify clustering algorithm.
    '''
    plt.bar(x=labels, height=array)
    plt.title("Comparison of {f} among clusterings".format(f=mode))
    plt.xlabel(cluster_method)
    plt.ylabel(mode)
    plt.savefig("../output/{m}_{c}.png".format(m=mode, c=cluster_method))
    plt.show()
    plt.clf()
    pass


def confusion_matrix(true_labels, predictions):
    '''
    Inputs should be np arrays of equal length (N x 1) and the same set of cluster values.
    1. Create a table of true_labels on rows and predictions on columns. Plot this as a heatmap.
    2. Calculate True Positive, False Negative, False Positive and True Negative. Create a confusion matrix and plot it as a heatmap.
    3. Create ROC curve: Plot individual clusters' ROC using metrics and include AUC.
    '''
    #### Plot confusion matrix for each cluster ####

    cluster_labels = np.unique(true_labels)
    num_clusters = cluster_labels.shape[0]

    # Plot a condensed confusion matrix for each cluster
    condensed_confusion_matrix = create_confusion_matrix(
        true_labels, predictions, cluster_labels, num_clusters)  # outputs a dataframe for each cluster

    x_labels = ['Predicted True', 'Predicted False']
    y_labels = ['Labelled True', 'Labelled False']
    for i in range(num_clusters):
        cluster_res = condensed_confusion_matrix[i]
        sns.heatmap(data=cluster_res, annot=True, xticklabels=x_labels,
                    yticklabels=y_labels)
        plt.title("Confusion Matrix for Cluster {c}".format(
            c=cluster_labels[i]))

        plt.savefig(
            "../output/ConfusionMatrix_{c}.png".format(c=cluster_labels[i]))
        plt.clf()

    # Plot ROC curve using individual clusters (use metrics package)

    label_binarizer = LabelBinarizer().fit(true_labels)
    y_onehot_test_true = label_binarizer.transform(true_labels)
    y_onehot_test_pred = label_binarizer.transform(pred_labels)

    for i in range(num_clusters):
        fpr, tpr, _ = roc_curve(y_onehot_test_true[:, i].ravel(),
                                y_onehot_test_pred[:, i].ravel())
        auroc = auc(fpr, tpr)

        RocCurveDisplay.from_predictions(
            y_onehot_test_true[:, i], y_onehot_test_pred[:, i], label="AUC={n}".format(n=auroc))
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Cluster {c} vs others".format(c=cluster_labels[i]))
        plt.legend()
        plt.savefig("../output/ROC_{c}.png".format(c=i))
        plt.clf()

    pass


def create_confusion_matrix(true, pred, cluster_labels, num_clusters):

    # create a k x k np matrix enumerating the matches between true and predicted labels
    global_matrix = create_global_matrix(
        true, pred, cluster_labels, num_clusters)

    matrices = []  # store each cluster's df here

    for i in range(num_clusters):
        results = np.zeros((2, 2))

        tp = global_matrix[i, i]
        fn = np.sum(global_matrix[i, :]) - tp
        fp = np.sum(global_matrix[:, i]) - tp
        tn = np.sum(global_matrix) - tp - fp - fn

        results[0, 0] = tp
        results[1, 0] = fp
        results[0, 1] = fn
        results[1, 1] = tn

        matrices.append(results)

    return matrices


def create_global_matrix(true, pred, cluster_labels, num_clusters):
    # Create a matrix to store true labels on rows and predicted labels on columns
    matrix = np.zeros((num_clusters, num_clusters))

    for i in range(num_clusters):
        # find all indices of current cluster in original labels
        cluster_idx = np.where(true == cluster_labels[i])

        # find true labels of current cluster
        true_labels = true[cluster_idx]

        # find predicted labels of current cluster
        pred_labels = pred[cluster_idx]

        for j in range(num_clusters):
            # count number of predictions made for each cluster in current true cluster
            values = np.where(pred_labels == cluster_labels[j])
            matrix[i, j] = len(values[0])

    sns.heatmap(data=matrix, annot=True, xticklabels=cluster_labels,
                yticklabels=cluster_labels)
    plt.title("Overall Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("../output/ConfusionMatrixOverall.png")
    plt.clf()
    return matrix


def calculate_cluster_centroids(cluster_labels, data):
    unique_clusters = np.unique(cluster_labels)
    num_clusters = unique_clusters.shape[0]
    centroids = np.zeros((num_clusters, data.shape[1]))
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)
        cluster_data = data[cluster_indices]
        mean_cluster = np.mean(cluster_data, axis=0)
        centroids[i, :] = mean_cluster
    return centroids


### Uncomment below for testing purposes. If ready to submit, just delete this section ###
'''
if __name__ == "__main__":
    # find_closest_centroid(point, cluster_label, cluster_centroids)
    # silhouette_coefficient(cluster_labels, data, cluster_centroids)
    # find_smallest_distance(cluster_1, cluster_2)
    # find_largest_distance(cluster)
    # dunn_index(cluster_labels, data, cluster_centroids)
    # plot_evaluation_metric(array, labels, mode, cluster_method)

    data = np.loadtxt("../output/reduced_sample_20_PCs.csv",
                      delimiter=",", dtype=float)

    attempts = 5
    di = []
    sc = []
    labels = [i for i in range(attempts)]
    for i in range(attempts):

        random_labels = np.random.randint(low=0, high=5, size=data.shape[0])

        centres = calculate_cluster_centroids(random_labels, data)

        di_att = dunn_index(random_labels, data, centres)
        sc_att = silhouette_coefficient(random_labels, data, centres)

        di.append(di_att)
        sc.append(sc_att)

    plot_evaluation_metric(di, labels, "Dunn Index", "Random Clustering")
    plot_evaluation_metric(
        sc, labels, "Silhouette Coefficient", "Random Clustering")
'''
