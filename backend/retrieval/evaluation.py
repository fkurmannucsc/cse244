import numpy as np
import os
import sys
import torch

from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from backend.constants import MINI_BATCH_SIZE

import backend.retrieval.retrieval
import backend.learning.evaluator.evaluators

def show_info(retriever, queries):
    """
    Debugging function to show the results for specific queries.
    """
    # Load the current SharkfinDataset.
    print("Loading dataset and embeddings.")
    dataset = retriever.get_dataset()
    ids = dataset.get_ids()
    filenames = dataset.get_filenames()
    labels = dataset.get_labels().cpu().detach().numpy()
    label_counts = np.unique(labels, return_counts=True)
    label_counts_dict = dict(zip(label_counts[0], label_counts[1]))

    for query in queries:
        query_index = ids.index(query)
        query_name = filenames[query_index]
        print(f"Result for query id {query}, {query_name}:")

        # Show information on the query image and shark.
        query_image = backend.db.media.get_media(query, get_full=True)
        if not isinstance(query_image, backend.db.media.Media):
            print(f"WARNING: Query image with id {query} not found.")
        query_shark_id = query_image.shark["id"]
        query_shark = backend.db.shark.get_shark(query_shark_id, True, True)
        
        # query_image.print(compact=True)
        # query_shark.print(compact=True)
        print("Associated sharks:", len(query_shark.associated_sharks))
        print("Associated medias:", len (query_shark.associated_medias))

        # Use one of the nearest neighbor methods to find neighbors.
        # neighbors = retriever.get_image_neighbors(query)
        # neighbors = retriever.get_image_neighbor_sharks(query)
        neighbors = retriever.get_image_neighbors(query, centroid=True)

        neighbor_ids = neighbors["ids"]
        neighbor_labels = neighbors["labels"]
        neighbor_distances = neighbors["distances"]

        # Sanity check that the nearest neighbor is itself, with it's own label, if this is not the case, we have problems.
        if neighbor_labels[0] != labels[query_index]:
            print(f"WARNING: Neighbor 0 is not of expected label")

        # print(len(neighbor_ids), neighbor_ids[:51])
        # print(len(neighbor_distances), neighbor_distances[:51])
        # print(len(neighbor_labels), neighbor_labels[:51])

def evaluation_prepare(retriever, retrieval_methods, retrieval_method=0, train_test=False):
    """
    Helper function that prepares to run evaluations by selecting the retriever, loading the dataset,
    and selecting the ids to test in case of the train/test split.
    """
    # Load the stored dataset, add up label counts for hits computation.
    print("Loading dataset and embeddings.")
    dataset = retriever.get_dataset()
    ids = dataset.get_ids()
    filenames = dataset.get_filenames()
    labels = dataset.get_labels().cpu().detach().numpy()
    label_counts = np.unique(labels, return_counts=True)
    label_counts_dict = dict(zip(label_counts[0], label_counts[1]))

    # Filter to only the train or test set:
    if train_test:
        print("Filtering dataset for train/test.")
        train_ids = dataset.get_train_ids()
        test_ids = dataset.get_test_ids()
        print("Train/test", len(train_ids), len(test_ids))

        focus_ids = []
        focus_filenames = []
        focus_labels = []
        for index, id in enumerate(ids):
            if id in test_ids: # Filter to train or test here.
                focus_ids.append(id)
                focus_filenames.append(filenames[index])
                focus_labels.append(labels[index])

        ids = focus_ids
        filenames = focus_filenames
        labels = focus_labels

    print("Ids in selected set", len(ids))

    # Find the retrieval function to use.
    retrieval_info = retrieval_methods[retrieval_method]
    retrieval_name = retrieval_info[0]
    retrieval_function = retrieval_info[1]
    median = retrieval_info[2]
    print("Retrieving results with retrieval method: ", retrieval_name)

    return ids, labels, label_counts_dict, retrieval_function, median

def get_neighbor_labels(query, label, retrieval_function, median):
    """
    Helper function that applies the retrieval function to get the neighbor images
    and return the unique neighbor labels, ordered.
    """
    # Get neighbors with chosen retrieval method.
    neighbors = retrieval_function(query, median=median)
    neighbor_ids = neighbors["ids"]
    neighbor_labels = neighbors["labels"]

    # Remove the query from the neighbor results, confirm it matches what is expected.
    # try:
    neighbors_query_index = neighbor_ids.index(query)
    neighbor_ids.pop(neighbors_query_index)
    removed_label = neighbor_labels.pop(neighbors_query_index)
    assert removed_label == label
    # except:
    #     pass

    unique_neighbor_labels = []
    for label in neighbor_labels:
        if label not in unique_neighbor_labels:
            unique_neighbor_labels.append(label)
    
    return unique_neighbor_labels

def cluster_evaluation(retriever, verbose=False):
    """

    """
    # Get image labels and embeddings.
    labels = retriever.labels
    embeddings = retriever.embeddings
    print(len(embeddings))

    # Get centroid labels and embeddings.
    centroid_labels = retriever.centroid_labels
    centroid_embeddings = retriever.centroid_embeddings
    print(len(centroid_embeddings))

    # Fill dict by iterating over centroids, getting average distance of each image with that label, to that centroid.
    item_dict = {}
    for index, centroid_embedding in enumerate(centroid_embeddings):
        centroid_label = centroid_labels[index]
        cluster_distances = []

        for index, embedding in enumerate(embeddings):
            label = labels[index]

            if label == centroid_label:
                cluster_distances.append(np.linalg.norm(centroid_embedding - embedding))

        item_dict[centroid_label] = np.mean(cluster_distances)

        # # Only compute hits when label contains more than 1 image, because with zero images you can't actually find a neighbor.
        # if label_counts_dict[label] == 1:
        #     continue

    # Sort the output dictionary to show highest k_value ids first.
    item_dict = dict(sorted(item_dict.items(), key=lambda item: -1 * item[1]))

    print("Number of clusters evaluated:", len(item_dict))

    return item_dict

def get_merge_suggestions(retriever, threshold=None, verbose=False):
    """
    Using centroid embeddings, find closest centroids. For each label, returns distance of the other centroids, ordered by the closest centrod.
    Note, this will raise false positives with train/test splitting since you're essentially artificially unmerging clusters.

    TODO This can be extended by checking if images assigned to one label are actually closer to another centroid.
    """
    # Get centroid embeddings/labels and their neighbor lists.
    centroid_labels = retriever.centroid_labels
    centroid_embeddings = retriever.centroid_embeddings
    computed_neighbors = NearestNeighbors(n_neighbors=len(centroid_labels), metric='euclidean').fit(centroid_embeddings)
    distances, neighbor_results = computed_neighbors.kneighbors(centroid_embeddings)

    print(len(centroid_labels), len(centroid_embeddings), len(distances), len(neighbor_results))
    print(len(neighbor_results[0]))
    return

    # Present information in dictionary by centroid, first result is self, remove that always.
    centroid_dict = {}
    for index, neighbor_list in enumerate(neighbor_results[1:]):
        label = centroid_labels[index]
        centroid_neighbors = zip(neighbor_list, distances[index])

        # Either add all of the centroid neighbors or just the ones closer than a threshold.
        if threshold is None:
            centroid_dict[label] = centroid_neighbors
        else:
            threshold_neighbors = []
            for neighbor in centroid_neighbors:
                if neighbor[1] <= threshold:
                    threshold_neighbors.append(neighbor)
            if len(threshold_neighbors) > 0:
                centroid_dict[label] = threshold_neighbors

    return centroid_dict

def get_unmatch_suggestions(retriever, retrieval_methods, retrieval_method=0, verbose=False):
    """
    Using results from hits at k evaluation, returns the images who the model struggles most to match to their label.
    In some cases, the model is wrong, but in others, it may provide insight into mis-labeling or a low quality image.
    """
    ids, labels, label_counts_dict, retrieval_function, median = evaluation_prepare(retriever, retrieval_methods, retrieval_method, train_test=False)

    # Fill dict by iterating over dataset for hits at k.
    item_dict = {}
    for index, id in enumerate(ids):
        query = id
        label = labels[index]

        # Only compute hits when label contains more than 1 image, because with zero images you can't actually find a neighbor.
        if label_counts_dict[label] == 1:
            continue
        
        unique_neighbor_labels = get_neighbor_labels(query, label, retrieval_function, median)

        # Compute the k value needed to get a hit.
        k_value = unique_neighbor_labels.index(label)
        item_dict[query] = k_value

    # Sort the output dictionary to show highest k_value ids first.
    item_dict = dict(sorted(item_dict.items(), key=lambda item: -1 * item[1]))

    print("Number of ids evaluated:", len(item_dict))

    return item_dict

def retrieval_hits_at_k(retriever, retrieval_methods, retrieval_method=0, train_test=False, verbose=False):
    """
    Compute the hits at for when an image is queried in shark matcher and shark matches are returned.
    Used to test different retrieval methods.

    Method 1: Image to image nearest neighbors - find nearest neighbors for query image.

    Method 2: Image to centroid - centroid with the lowest distance to the query image.

    Method 3: Query centroid to centroid - centroid with the lowest distance to the query centroid.
        This is harder to evaluate because if centroids are pre - computed, query centroid is always the same as its matched centroid, so need a separate test set. Although the above also has a slight bias in this respect. TODO test set evaluation.

    Then, TODO, make some kind of cluster visualization to see if this is onto something.

    Method 4: TODO Query centroid to image nearest neighbors - find nearest neighbors for query centroid.

    Note: For all these methods, it may be useful/essential to weight images based on: quality, temporal recency, topshot
    vs. underwater.

    This method simply takes the first X unique shark (images with different labels) results, compares the labels with the query, and
    sums up hits at k like that. 

    The idea is to test hits at k for different retrieval techniques.
    """
    ids, labels, label_counts_dict, retrieval_function, median = evaluation_prepare(retriever, retrieval_methods, retrieval_method, train_test)
    
    # Compute hits@k with nearest neighbors.
    average_hit_at_1 = 0
    average_hit_at_5 = 0
    average_hit_at_10 = 0
    average_hit_at_25 = 0
    average_hit_at_50 = 0
    average_hit_at_100 = 0
    num_samples = 0

    # Iterage over dataset for hits at k.
    for index, id in enumerate(ids):
        query = id
        label = labels[index]

        # Only compute hits when label contains more than 1 image, because with zero images you can't actually find a neighbor.
        if label_counts_dict[label] == 1:
            continue
        
        unique_neighbor_labels = get_neighbor_labels(query, label, retrieval_function, median)

        # print(unique_neighbor_labels)

        # Compare the desired label to the label of the closest k neighbors of unique labels.
        hit_at_1, hit_at_5, hit_at_10, hit_at_25, hit_at_50, hit_at_100 = backend.learning.evaluator.evaluators.__hits_at_k__(labels[index], unique_neighbor_labels)

        average_hit_at_1 += hit_at_1
        average_hit_at_5 += hit_at_5
        average_hit_at_10 += hit_at_10
        average_hit_at_25 += hit_at_25
        average_hit_at_50 += hit_at_50
        average_hit_at_100 += hit_at_100
        num_samples += 1

        # if hit_at_50 == 0:
        #     print("First 50 neighbors for query:", query, index)
        #     print(neighbor_ids[:51])
        #     print(neighbor_labels[:51])
        #     print(hit_at_1, hit_at_5, hit_at_10, hit_at_25, hit_at_50)

    average_hit_at_1 /= num_samples
    average_hit_at_5 /= num_samples
    average_hit_at_10 /= num_samples
    average_hit_at_25 /= num_samples
    average_hit_at_50 /= num_samples
    average_hit_at_100 /= num_samples


    print("Number of samples taken:", num_samples)

    return  {
        "average_hit_at_1": average_hit_at_1,
        "average_hit_at_5": average_hit_at_5,
        "average_hit_at_10": average_hit_at_10,
        "average_hit_at_25": average_hit_at_25,
        "average_hit_at_50": average_hit_at_50,
        "average_hit_at_100": average_hit_at_100
    }

def main():
    retriever = backend.retrieval.retrieval.Retriever("lora")

    # Debug, show information for a list of retrieved objects using the retriever object.
    # show_info(test_retriever, [583, 2647, 2674, 2723, 2798, 2799])

    # Reference tables explaining different retrieval methods.
    retrieval_methods = {
        0: ("image_image", retriever.get_image_neighbors, False),
        1: ("image_shark_mean_distance", retriever.get_image_neighbor_sharks, False),
        2: ("image_shark_med_distance", retriever.get_image_neighbor_sharks, True),
        3: ("image_centroid_mean_distance", retriever.get_centroid_neighbors, False), # TODO Requires storing different centroid_neighbors function.
        4: ("image_centroid_med_distance", retriever.get_centroid_neighbors, True), # TODO Requires storing different centroid_neighbors function.
        5: ("random_guess", retriever.get_random_guess, False), 
    }
    # Settings.
    train_test = True
    retrieval_method = 0

    # For getting hits at k results using a specific method.
    result = retrieval_hits_at_k(retriever, retrieval_methods=retrieval_methods, retrieval_method=retrieval_method, train_test=train_test, verbose=False)
    print(result)

    # For getting unmatch suggestions using a specific method.
    # result = get_unmatch_suggestions(retriever, retrieval_methods=retrieval_methods, retrieval_method=retrieval_method)


    # # For getting merge suggestions using a specific method.
    # result = get_merge_suggestions(retriever, threshold=10)

    # For getting cluster average distance results.
    # result = cluster_evaluation(retriever)


    for key, value in result.items():
        print(f"{key},{value}")

if __name__ == "__main__":
    main()