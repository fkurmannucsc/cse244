"""Functions for building and storing embeddings and storing data such as nearest neighbors as pt files in the enbeddings directory.
Image neighbors are stored as a pt file to prevent having to compute at startup, TODO should other neighbor type such as shark neighbors 
also be stored as pt files, with different comprison methods this could eventually get out of hand."""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import tqdm

from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NeighborhoodComponentsAnalysis

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from backend.constants import EMBEDDINGS_DIR
from backend.constants import IMAGES_DIR
from backend.constants import FRONTEND_IMAGES_DIR
from backend.sharkfin_dataset import SharkfinDataset
from backend.constants import FULL_ANNOTATIONS_PATH

import backend.db.media
import backend.db.shark
import repo_utils
import backend.utils
import backend.convert_data


def store_transformed_embeddings(model_name, centroid=False, method="nca"):
    """
    Transform the embeddings with the given method and store the transformed embeddings.
    These techniques can include dimensionality reduction and component analysis for better cluster behavior for 
    nearest neighbors.
    """
    # Method check.
    if method not in ["nca"]:
        print("Invalid transformation method, doing nothing.")
        return
    
    device = repo_utils.get_torch_device()

    # Load the embeddings and labels, either image or centroid.
    if not centroid:
        stored_embeddings = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_embeddings.pt"), weights_only=False, map_location=device).cpu().detach().numpy()
        dataset = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_dataset.pt"), map_location=device, weights_only=False)
        stored_labels = dataset.get_labels().cpu().detach().numpy()
    else:
        stored_embeddings = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_centroid_embeddings.pt"), weights_only=False, map_location=device).cpu().detach().numpy()
        stored_labels = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_centroid_labels.pt"), weights_only=False, map_location=device).cpu().detach().numpy()
    
    # Transform embeddings based on the selected method, then store them in the correctly named file.
    if method == "nca":
        print(f"Transforming embeddings with {method}, centroid: {centroid}")
        nca = NeighborhoodComponentsAnalysis(n_components=3, random_state=1)
        transformed_embeddings = nca.fit_transform(stored_embeddings, stored_labels)
        transformed_embeddings = transformed_embeddings.astype(np.float32)


        transformed_embeddings = torch.tensor(transformed_embeddings, device=device)

        file_name = "stored_transformed_embeddings.pt"
        if centroid:
            file_name = "stored_transformed_centroid_embeddings.pt"

        # Save the embeddings as pt files.
        print("Saving transformed embeddings")
        os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name), exist_ok=True)
        torch.save(transformed_embeddings, os.path.join(EMBEDDINGS_DIR, model_name, file_name))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name, timestamp), exist_ok=True)
        torch.save(transformed_embeddings, os.path.join(EMBEDDINGS_DIR, model_name, timestamp, file_name))

def print_dataset(model_name):
    """
    Debug function to just print the dataset.
    """
    device = repo_utils.get_torch_device()

    # Load the Sharkfin Dataset.
    print("Loading and updating dataset.")
    dataset = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_dataset.pt"), map_location=device, weights_only=False)
    ids = dataset.get_ids()
    labels = dataset.get_labels().cpu().detach().numpy()

    # for index, id in enumerate(ids):
    #     print(id, labels[index])


    print("Train and test indices:")
    train_ids = dataset.get_train_ids()
    test_ids = dataset.get_test_ids()
    print("TRAIN:", len(train_ids))
    print("\n")
    print("TEST:", len(test_ids), test_ids)

def store_dataset(model_name, channels=3, prepare_dataset=False, train_test=True):
    device = repo_utils.get_torch_device()

    # Optionally, prepare the annotations files needed for loading the dataset.
    if prepare_dataset:
        backend.convert_data.prepare_dataset(train_test=train_test, save=True)

    # Load the Sharkfin Dataset, for embeddings generation and embeddings, always load the full dataset, train/test split is saved as part of the dataset.
    print("Loading and saving dataset.")
    dataset = SharkfinDataset(IMAGES_DIR, FULL_ANNOTATIONS_PATH, device=device, channels=channels)

    # Save the dataset as pt file.
    os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name), exist_ok=True)
    torch.save(dataset, os.path.join(EMBEDDINGS_DIR, model_name, "stored_dataset.pt"))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name, timestamp), exist_ok=True)
    torch.save(dataset, os.path.join(EMBEDDINGS_DIR, model_name, timestamp, "stored_dataset.pt"))

def store_embeddings(model_name, image=True, centroid=True, test_split=False, median=False):
    """
    Store all vector embeddings after generating them with the specified trained model on the sharkfin dataset.
    This stores embeddings so must only be run after a new model is trained or data is upated.
    By default stores all image and centroid embeddings, this can be adjusted with the corresponding parameters.
    """
    device = repo_utils.get_torch_device()
    print("Storing embeddings.")

    # Load dataset.
    print("Loading dataset.")
    dataset = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_dataset.pt"), map_location=device, weights_only=False)
    ids = dataset.get_ids()
    labels = dataset.get_labels().cpu().detach().numpy()

    if image:
        print("Storing image embeddings.")

        # Load the pretrained models.
        print("Loading model.")
        projection_head, model_backbone = backend.utils.load_model(model_name)

        # Compute embeddings for each image.
        print("Computing image embeddings.")
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        embeddings = []
        with tqdm.tqdm(loader) as tq:
            for step, batch in enumerate(tq):
                with torch.no_grad():
                    embeddings.append(projection_head(model_backbone(batch[0].to(device))))

        embeddings = torch.cat(embeddings, dim=0)

        # Save the image embeddings as pt files.
        print("Saving image embeddings")
        os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name), exist_ok=True)
        torch.save(embeddings, os.path.join(EMBEDDINGS_DIR, model_name, "stored_embeddings.pt"))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name, timestamp), exist_ok=True)
        torch.save(embeddings, os.path.join(EMBEDDINGS_DIR, model_name, timestamp, "stored_embeddings.pt"))

    if centroid:
        print("Storing centroid embeddings.")

        # Load most recent embeddings.
        print("Loading image embeddings.")
        embeddings = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_embeddings.pt"), map_location=device, weights_only=False).cpu().detach().numpy()

        # Split into the training and testing sets. The centroids will be computed across all points within the same set. From there on everything will be the same.
        # Total number of centroids will now equal the number of unique labels in the training set + the number of unique labels in the testing set.
        if test_split:
            train_ids = dataset.get_train_ids()
            test_ids = dataset.get_test_ids()

            centroid_embeddings, centroid_labels = [], []
            
            for set_ids in [train_ids, test_ids]:
                focus_embeddings, focus_labels, focus_ids = [], [], []
                for index, id in enumerate(ids):
                    if id in set_ids:
                        focus_embeddings.append(embeddings[index])
                        focus_labels.append(labels[index])
                        focus_ids.append(id)

                # Populate lists storing centroid embeddings for each shark.
                print("Computing centroid embeddings. Length focus_embeddings:", len(focus_embeddings))
                focus_centroid_embeddings, focus_centroid_labels = compute_set_embeddings(np.array(focus_embeddings), np.array(focus_labels), focus_ids, median)
                centroid_embeddings = centroid_embeddings + focus_centroid_embeddings
                centroid_labels = centroid_labels + focus_centroid_labels

        else:
            # No split, just compute the centoids for each shark.
            print("Computing centroid embeddings.")
            centroid_embeddings, centroid_labels = compute_set_embeddings(embeddings, labels, ids, median)

        # for index, item in enumerate(centroid_embeddings):
        #     print(centroid_labels[index], np.mean(item))

        centroid_embeddings = torch.tensor(centroid_embeddings, device=device)
        centroid_labels = torch.tensor(centroid_labels, dtype=torch.long, device=device)

        # Save the centroid embeddings and labels as pt files.
        print("Saving centroid embeddings.")
        os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name), exist_ok=True)
        torch.save(centroid_embeddings, os.path.join(EMBEDDINGS_DIR, model_name, "stored_centroid_embeddings.pt"))
        torch.save(centroid_labels, os.path.join(EMBEDDINGS_DIR, model_name, "stored_centroid_labels.pt"))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name, timestamp), exist_ok=True)
        torch.save(centroid_embeddings, os.path.join(EMBEDDINGS_DIR, model_name, timestamp, "stored_centroid_embeddings.pt"))
        torch.save(centroid_labels, os.path.join(EMBEDDINGS_DIR, model_name, timestamp, "stored_centroid_labels.pt"))

def update_embeddings(model_name, transform_method=None):
    """
    Update the vector embeddings by applying the specified trained model on the most current data.
    This checks which embeddings need to be re-created
    and only updates those, ideal for when small dataset changes are made (a few new images),
    not affecting all embeddings.
    """
    device = repo_utils.get_torch_device()
    print("Updating embeddings.")

    # Load the stored dataset and embeddings.
    print("Loading stored dataset.")
    stored_dataset = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_dataset.pt"), weights_only=False, map_location=device)
    stored_filenames = stored_dataset.get_filenames()
    stored_embeddings = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_embeddings.pt"), weights_only=False, map_location=device)

    # Load the pretrained models.
    print("Loading model.")
    projection_head, model_backbone = backend.utils.load_model(model_name)

    # Store a new dataset with the most current database contents, the load this dataset.
    print("Storing updated datset.")
    store_dataset(model_name, prepare_dataset=True, train_test=False)
    current_dataset = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_dataset.pt"), weights_only=False, map_location=device)
    current_filenames = current_dataset.get_filenames()

    print("Stored vs current dataset length", len(stored_filenames), len(current_filenames))

    # Counter for embeddings carried over vs updated.
    carried_over, updated = 0, 0

    current_embeddings = []
    print("Updating embeddings.")
    for index, current_filename in enumerate(current_filenames):
        # Try carrying over embeddings from the stored file
        try:
            stored_index = stored_filenames.index(current_filename)
            stored_tensor = stored_embeddings[stored_index]
            stored_tensor = torch.reshape(stored_tensor, (1,1024))
            current_embeddings.append(stored_tensor)
            carried_over += 1
        except:
            print("Updating embeddings.", index)
            new_embedding = projection_head(model_backbone(current_dataset[index][0].to(device)))

            current_embeddings.append(new_embedding)
            updated += 1

    # Check that you have counted correctly.
    assert carried_over + updated == len(current_embeddings)
    print(f"Embeddings carried over: {carried_over} updated: {updated}.")

    print(f"Saving embeddings, previously: {len(stored_embeddings)} now: {len(current_embeddings)}.")
    current_embeddings = torch.cat(current_embeddings, dim=0)

    # Save the embeddings.
    os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name), exist_ok=True)
    torch.save(current_embeddings, os.path.join(EMBEDDINGS_DIR, model_name, "stored_embeddings.pt"))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name, timestamp), exist_ok=True)
    torch.save(current_embeddings, os.path.join(EMBEDDINGS_DIR, model_name, timestamp, "stored_embeddings.pt"))

    return updated

def compute_set_embeddings(embeddings, labels, ids, median=False):
    """
    Helper function that computes and returns an average embedding for each unique shark among the given ids.
    """
    centroid_embeddings = []
    centroid_labels = []
    for index, id in enumerate(ids):
        # Get the label of the id.
        label = labels[index]

        # For each shark, make one centroid embedding.
        if label not in centroid_labels:
            centroid_embeddings.append(get_average_embedding(embeddings, labels, id, median))
            centroid_labels.append(label)

    # Check that the correct number of embeddings have been made.
    label_counts = np.unique(labels, return_counts=True)
    assert len(centroid_embeddings) == len(label_counts[0])

    return centroid_embeddings, centroid_labels
    
def get_average_embedding(embeddings, labels, query_image_id, median=False):
    """
    Get the average embedding value of a query shark. The query image shark's 
    entire set of embeddings are averaged to generate a central embedding for that shark.

    Note that embeddings and labels lists that are passed may contain only a subset of the 
    total embeddings/labels such as the train/test embeddings/labels, so the average will 
    be computed within the relevant set.
    """    
    # Get the source shark of the query image.
    query_image = backend.db.media.get_media(query_image_id, get_full=True)
    if not isinstance(query_image, backend.db.media.Media):
        print(f"WARNING: Query image with id {query_image_id} not found.")
        return
    query_shark_id = query_image.shark["id"]
    query_shark = backend.db.shark.get_shark(query_shark_id, False, True)
    query_shark_source = query_shark.source

    # Add up al the embeddings matching this shark.
    query_shark_embeddings = []
    for index, embedding in enumerate(embeddings):
        if (labels[index] == query_shark_source):
            query_shark_embeddings.append(embedding)
    
    # Double check that the correct number of embeddings are identified.
    label_counts = np.unique(labels, return_counts=True)
    label_counts_dict = dict(zip(label_counts[0], label_counts[1]))
    assert label_counts_dict[query_shark_source] == len(query_shark_embeddings)

    # Produce an average embedding for the query shark.
    if median:
        average_embedding = np.median(query_shark_embeddings, axis=0)
    else:
        average_embedding = np.mean(query_shark_embeddings, axis=0)
    
    average_embedding = list(average_embedding)

    return average_embedding

def store_image_neighbors(model_name, centroid=False, test_split=False, transformed=False):
    """ 
    Compute nearest neighbors for embeddings stored in the embeddings directory and store in the embeddings directory.
    This stores the nearest neighbors as a file so must only be run after a new model is trained or data is updated. 
    """
    device = repo_utils.get_torch_device()

    # Load dataset.
    print("Loading dataset and embeddings.")
    dataset = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_dataset.pt"), map_location=device, weights_only=False)

    if not centroid:
        print(f"Storing image neighbors. Transformed: {transformed}")

        file_name = "stored_embeddings.pt"
        if transformed:
            file_name = "stored_transformed_embeddings.pt"

        embeddings = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, file_name), map_location=device, weights_only=False).cpu().detach().numpy()
        labels = dataset.get_labels().cpu().detach().numpy()
    else:
        print(f"Storing centroid neighbors. Transformed: {transformed}")

        file_name = "stored_centroid_embeddings.pt"
        if transformed:
            file_name = "stored_transformed_centroid_embeddings.pt"

        embeddings = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, file_name), map_location=device, weights_only=False).cpu().detach().numpy()
        labels = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_centroid_labels.pt"), map_location=device, weights_only=False).cpu().detach().numpy()

    computed_neighbors = NearestNeighbors(n_neighbors=len(labels), metric='euclidean').fit(embeddings)
    distances, neighbor_results = computed_neighbors.kneighbors(embeddings)

    # Preliminary catch for conflicting embeddings and annotations.
    assert len(labels) == len(neighbor_results)

    # Get the neighbors for each id in the dataset in the desired dict format.
    neighbors = computed_neighbors_to_dict(dataset, labels, distances, neighbor_results, centroid=centroid, test_split=test_split)

    os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join(EMBEDDINGS_DIR, model_name, timestamp), exist_ok=True)
    if not centroid:
        print(f"Saving image neighbors. Transformed: {transformed}")
        torch.save(neighbors, os.path.join(EMBEDDINGS_DIR, model_name, "stored_image_neighbors.pt"))
        torch.save(neighbors, os.path.join(EMBEDDINGS_DIR, model_name, timestamp, "stored_image_neighbors.pt"))
    else:
        print(f"Saving centroid neighbors. Transformed: {transformed}")
        torch.save(neighbors, os.path.join(EMBEDDINGS_DIR, model_name, "stored_centroid_neighbors.pt"))
        torch.save(neighbors, os.path.join(EMBEDDINGS_DIR, model_name, timestamp, "stored_centroid_neighbors.pt"))

def computed_neighbors_to_dict(dataset, labels, distances, neighbor_results, centroid=False, test_split=False):
    """
    Helper function that turns the neighbor results from the kneighbors function
    into a dictionary with the query file name as the key and a dict of information
    about that file's nearest neighbors as the value. 
    """
    # Load all information.
    # Note that labels contains the labels of the embeddings while dataset_labels contains the labels of the dataset, which differ when centroid embeddings are handled.
    ids = dataset.get_ids()
    filenames = dataset.get_filenames()
    labels = list(labels)
    labels = [int(label) for label in labels]
    num_unique_labels = len(list(set(labels)))
    dataset_labels = list(dataset.get_labels().cpu().detach().numpy())    

    # Optionally split indices for train/test split.
    # Indices lists store which indices in the dataset belong to the train/test set. (Stored in dataset.)
    # Centroid incices lists store which centroid indices belong to the train/test set. (Centroids always formatted train, then test.)
    if not test_split:
        train_ids = ids
        test_ids = []
        train_centroid_indices = []
        test_centroid_indices = []
        for index in range(len(labels)):
            train_centroid_indices.append(index)
    else:
        train_ids = dataset.get_train_ids()
        test_ids = dataset.get_test_ids()
        train_centroid_indices = []
        test_centroid_indices = []
        for index in range(len(labels)):
            if index < num_unique_labels:
                train_centroid_indices.append(index)
            else:
                test_centroid_indices.append(index)
        
        print(f"Train/test ids {train_ids}, {test_ids}.")

    # Just for centroids, make a dictionary that maps labels to all indices in the dataset that are of that label.
    label_index_dict = {}
    label_tracker_dict = {}
    for label in labels:
        for index, _ in enumerate(ids):
            if dataset_labels[index] == label:
                if label not in label_index_dict:
                    label_index_dict[label] = []
                    label_tracker_dict[label] = []
                if index not in label_tracker_dict[label]:
                    # Boolean for whether this index is in the train or test set.
                    test = index in test_ids
                    label_index_dict[label].append((index, test))
                    label_tracker_dict[label].append(index)

    return_dict = {}
    # For each id in the dataset, find the nearest neighbor results and add it to the formatted dictionary.
    for index, id in enumerate(ids):
        query = id
        query_name = filenames[index]

        # In non centroid cases the neighbor results list is mapped directly to the ids list.
        if not centroid:
            neighbor_result = neighbor_results[index]
            match_ids = [ids[index] for index in neighbor_result]
            match_names = [filenames[index] for index in neighbor_result]
            match_labels = [int(dataset_labels[index]) for index in neighbor_result]
            match_distances = list(distances[index])

        else:
            # In centroid cases, you need to find the neighbor results list that represents results for the centroid this id belongs to.
            dataset_label = dataset_labels[index]
            
            # Store the centroid indices that will be evaluated, for a train image, that's train centroids, for a test image, test centroids.
            focus_centroid_indices = train_centroid_indices

            if test_split:
                if id in train_ids:
                    # print("TRAIN")
                    focus_centroid_indices = train_centroid_indices
                if id in test_ids:
                    # print("TEST")
                    focus_centroid_indices = test_centroid_indices
            
            # Find the index of the centroid we want to evaluate, searching only within the correct set of focus indices.
            # Having the right centroid means we are indeed looking at the centroid that represents the current id we are iterating on.
            for centroid_index in range(len(labels)):
                if centroid_index in focus_centroid_indices and labels[centroid_index] == dataset_label:
                    centroid_index = centroid_index
                    break

            # Get the neighbor result for this centroid and store its value for all ids associated with this centroid. 
            # This means only the ids who's label is the same as that of the centroid and who belong to the same set (test/train) as the centroid.
            neighbor_result = neighbor_results[centroid_index]
            distance_result = distances[centroid_index]
            
            match_ids = []
            match_names = []
            match_labels = []
            match_distances = []

            # print(id, neighbor_result, distance_result)
            # Add neighbor results to the match results list, if they are of the same set, train/test.
            for index, item in enumerate(neighbor_result):
                # This represents the dataset label of this neighbor result (centroid).
                dataset_label = labels[item]
                distance = distance_result[index]
                test_centroid = item in test_centroid_indices
                # Dataset info contains information on (dataset index of the item with this label, set membership of the item with this label).
                for dataset_info in label_index_dict[dataset_label]:
                    dataset_index = dataset_info[0]
                    test_result = dataset_info[1]

                    if test_result and test_centroid:
                        match_ids.append(ids[dataset_index])
                        match_names.append(filenames[dataset_index])
                        match_labels.append(dataset_label)
                        match_distances.append(distance)

                    if not test_result and not test_centroid and ids[dataset_index] not in match_ids:
                        match_ids.append(ids[dataset_index])
                        match_names.append(filenames[dataset_index])
                        match_labels.append(dataset_label)
                        match_distances.append(distance)

        # Store the dictionary entry for this id.
        return_dict[query_name] = {"query": query,
                                    "query_name": query_name,
                                    "matches": match_ids,
                                    "match_names": match_names,
                                    "match_distances": match_distances,
                                    "match_labels": match_labels}
        
    # for key, value in return_dict.items():
    #     print(key, value, "\n\n")

    return return_dict

def main():
    # Store dataset. and image embeddings.
    save = False
    channels = 4
    store_dataset(model_name="lora", channels = channels)

    # Store image embeddings (this takes a while).
    # Note that test split is not relevant to generating image embeddings, only centroid embeddings.
    store_embeddings(model_name="lora", image=True, centroid=False)
    # store_transformed_embeddings(model_name="lora", centroid=False)

    # Store the centroid embeddings. 
    # Either with or without splitting train and test. 
    # Use either mean or median centroid computation.
    test_split = True
    centroid = True
    median = False
    store_embeddings(model_name="lora", image=False, centroid=centroid, test_split=test_split, median=median)
    # store_transformed_embeddings(model_name="lora", centroid=centroid)

    # Store image and centroid neighbors.
    # Use the original or transfomred embeddings for neighbor computation.
    transformed = False
    store_image_neighbors(model_name="lora", centroid=False, test_split=False, transformed=transformed)
    store_image_neighbors(model_name="lora", centroid=centroid, test_split=test_split, transformed=transformed)


    # update_dataset("lora")

    # Print the dataset, useful for checking you have the right fold.
    print_dataset("lora")

if __name__ == "__main__":
    main()
