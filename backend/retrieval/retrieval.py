"""
Handles interaction with the stored embeddings in
order to retrieve relevant data relating to model predictions.
"""

import datetime
import numpy as np
import os
import re
import sqlite3
import sys
import torch
import random

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from backend.constants import EMBEDDINGS_DIR

import backend.db.media
import backend.utils
import repo_utils

class Retriever:
    """
    Retriever object storing parameters to be used in retrieval.
    """
    def __init__(self, model_name):
        """
        Initialize Retriever object.
        """
        self.model_name = model_name

        self.device = repo_utils.get_torch_device()

        # Load image neighbors.
        if model_name == "test":
            pass
        else:
            self.embeddings = torch.load(os.path.join(EMBEDDINGS_DIR, self.model_name, "stored_embeddings.pt"), map_location=self.device, weights_only=False).cpu().detach().numpy()
            self.image_neighbors = torch.load(os.path.join(EMBEDDINGS_DIR, self.model_name, "stored_image_neighbors.pt"), map_location=self.device, weights_only=False)
            self.centroid_embeddings = torch.load(os.path.join(EMBEDDINGS_DIR, self.model_name, "stored_centroid_embeddings.pt"), map_location=self.device, weights_only=False).cpu().detach().numpy()
            self.centroid_neighbors = torch.load(os.path.join(EMBEDDINGS_DIR, self.model_name, "stored_centroid_neighbors.pt"), map_location=self.device, weights_only=False)
            self.centroid_labels = torch.load(os.path.join(EMBEDDINGS_DIR, self.model_name, "stored_centroid_labels.pt"), map_location=self.device, weights_only=False)
            self.dataset = torch.load(os.path.join(EMBEDDINGS_DIR, model_name, "stored_dataset.pt"), map_location=self.device, weights_only=False)
            self.ids = self.dataset.get_ids()
            self.filenames = self.dataset.get_filenames()
            self.labels = self.dataset.get_labels().cpu().detach().numpy()

    def get_dataset(self):
        return self.dataset

    def to_dict(self, compact=False):
        """
        Represent object as dict.
        """
        return {"model_name": self.model_name,
                "device": self.device,
                "image_neighbors": self.image_neighbors}

    def print(self):
        """
        Print object as dict.
        """
        print(self.to_dict())

    def get_random_guess(self, query_image_id, median=False, centroid=False, filter=False):
        """
        Get neighbor image ids and distances for a query image id. Instead of using a 
        proper retrieval technique, just get a raondom guess.
        """
        if centroid:
            neighbor_results = self.get_centroid_neighbors(query_image_id, filter=filter)
        else:
            neighbor_results = self.get_image_neighbors(query_image_id, filter=filter)

        ids = neighbor_results["ids"]
        distances = neighbor_results["distances"]
        labels = neighbor_results["labels"]

        combined = list(zip(ids, distances, labels))
        random.shuffle(combined)
        combined = list(zip(*combined))

        return_dict = {"ids": list(combined[0]),
                       "distances": list(combined[1]),
                       "labels": list(combined[2])}
        
        # for key, value in return_dict.items():
        #     print(key, value[:3])

        return return_dict

    def get_centroid_neighbors(self, query_image_id, median=False, filter=False):
        return self.get_image_neighbors(query_image_id, median=median, centroid=True, filter=filter)

    def get_image_neighbors(self, query_image_id, median=False, centroid=False, filter=False):
        """
        Get neighbor image ids and distances for a query image id.
        """
        # Return random results in case of testing.
        if self.model_name == "test":
            return {"ids": [3,2,1,6,5,4,9,8,7,12,11,10,15,14,13,18,17,16],
                    "distances": [3,2,1,6,5,4,9,8,7,12,11,10,15,14,13,18,17,16],
                    "labels": [3,2,1,6,5,4,9,8,7,12,11,10,15,14,13,18,17,16]}

        # Get query image name.
        query_image = backend.db.media.get_media(query_image_id)
        if not isinstance(query_image, backend.db.media.Media):
            print(f"WARNING: Query image with id {query_image_id} not found.")
            return {"ids": [], "distances": [], "labels": []}
        query_image_file = query_image.name + ".jpeg"

        # Get neighbors from stored neighbors, image or centroid.
        if not centroid:
            neighbor_result = self.image_neighbors.get(query_image_file)
        else:
            neighbor_result = self.centroid_neighbors.get(query_image_file)

        # for key, value in neighbor_result.items():
        #     if type(value) == list:
        #         if key == "matches" or key == "match_distances":
        #             print(len(value), key, value)
        #     else:
        #         print(key, value)

        if neighbor_result is None:
            print(f"WARNING: Query image {query_image_file} not found in neighbor reference dictionary.")
            return {"ids": [], "distances": [], "labels": []}

        # Numpy int to int.
        neighbor_ids = [int(id) for id in neighbor_result["matches"]]
        neighbor_distances = [float(dist) for dist in list(neighbor_result["match_distances"])]
        neighbor_labels = [int(label) for label in neighbor_result["match_labels"]]

        # Filter items from the return list.
        # TODO possible filters can include low image quality, mismatch among image metadata, mismatch among shark metadata.
        # Also note that some of this may be handled by PSL eventuall.
        if filter:
            filtered_neighbor_ids = []
            filtered_neighbor_distances = []
            filtered_neighbor_labels = []
            # Get the query image and neighbor images from the database.
            # Consider, perhaps if some of these constrains are going to be applied always, then it's smarter
            # to actually cache them in the stored dataset file, but for now to test the concept we filter here.
            all_images = backend.db.media.get_medias(ids=None)

            # Store all the images in a dictionary.
            image_dict = {}
            for image in all_images:
                image_dict[image.id] = image

            print("Image dict:", len(image_dict), image_dict[100])

            # Iterate over all the neighbors and filter keep only the desired.
            for id, index in enumerate(neighbor_ids):
                neighbor_image = image_dict[id]

                # Filter out if the image is "unmatchable".
                if neighbor_image.quality == "unmatchable":
                    print("Found unmatcable")
                else:
                    filtered_neighbor_ids.append(id)
                    filtered_neighbor_distances.append(neighbor_distances[index])
                    filtered_neighbor_labels.append(neighbor_labels[index])
            
            neighbor_ids = filtered_neighbor_ids
            neighbor_distances = filtered_neighbor_distances
            neighbor_labels = filtered_neighbor_labels
        
        return_dict = {"ids": neighbor_ids,
                       "distances": neighbor_distances,
                       "labels": neighbor_labels}
        
        # for key, value in return_dict.items():
        #     print(key, value[:3])

        return return_dict

    def get_image_neighbor_sharks(self, query_image_id, median=False):
        """
        Given an image, get the closest sharks by average distance to it and return a list
        of all of the neighbor images, now ranked by centroid closeness to the query.
        """
        # First get the image neighbors.
        image_neighbors = self.get_image_neighbors(query_image_id)

        # Store the image ids and distances for each shark.
        shark_dict = {}
        for index, id in enumerate(image_neighbors["ids"]):
            distance = image_neighbors["distances"][index]
            label = image_neighbors["labels"][index]

            # Add the distance to the running total distance metric for each shark.
            if shark_dict.get(label) is None:
                shark_dict[label] = [(id, distance)]
            else:
                shark_dict[label].append((id, distance))

        # Compute average shark distances for all sharks labeled the same as the query, excluding the query.
        shark_average_distances = {}
        for key, value in shark_dict.items():
            shark_distances = []
            for item in value:
                # Don't include query, where the distance is zero, in the neighbor average computation.
                if (item[0] == query_image_id):
                    pass
                else:
                    shark_distances.append(item[1])
            
            # Compute the mean of the distances for this shark.
            if median:
                shark_average = np.median(np.array(shark_distances))
            else:
                shark_average = np.mean(np.array(shark_distances))
                    
            shark_average_distances[key] = shark_average

        # Sort the average distances.
        shark_average_distances = dict(sorted(shark_average_distances.items(), key=lambda item: item[1]))

        # Build a return dict ordered by shark distances.
        return_dict = {"ids": [],
                       "distances": [],
                       "labels": []}

        # For each close shark, add all of the images associated with it to the results.
        for label, _ in shark_average_distances.items():
            for neighbor in shark_dict[label]:
                # Shark_dict values are already sorted by distance from construction.
                return_dict["ids"].append(neighbor[0])
                return_dict["distances"].append(neighbor[1])
                return_dict["labels"].append(label)

        return return_dict
    
    # TODO, get the centroid of a query shark and compare with that.
    # Practically, you'll want to find the average embedding of the query shark,
    # (this can be done with the (not train/test) centroid embeddings) then
    # order the regular embeddings by closeness to that centroid.
    def get_shark_neighbor_images(self, query_image_id):
        pass

def main():
    """
    Main
    """

    test_retriever = Retriever("lora")
    # print(test_retriever.dataset)
    # print(test_retriever.labels)

    print("Running main")
    print("\nNeighbor Image")
    test_id = 33
    results = test_retriever.get_image_neighbors(test_id, filter=True)
    for key, value in results.items():
            if type(value) == list:
                print(len(value), key, value[:30])
            else:
                print(key, value)
    results = test_retriever.get_image_neighbor_sharks(test_id)
    # print("\nNeighbor Sharks")
    # for key, value in results.items():
    #         if type(value) == list:
    #             print(len(value), key, value[:30])
    #         else:
    #             print(key, value)

    # results = test_retriever.get_image_neighbors(test_id, centroid=True)
    # print("\nNeighbor Centroid")
    # for key, value in results.items():
    #         if type(value) == list:
    #             print(len(value), key, value[:30])
    #         else:
    #             print(key, value)


if __name__ == "__main__":
    main()
