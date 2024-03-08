import cv2
import numpy as np




class Tracker:
    def __init__(self, max_distance=50):
        self.tracked_objects = {}  # Dictionary to store tracked objects with IDs
        self.max_distance = max_distance  # Maximum distance for matching objects
        self.next_object_id = 1  # Counter for assigning unique IDs to objects

    def update(self, new_rectangles):
        # Initialize a dictionary to store updated objects
        updated_objects = {}

        # Iterate over the new detected rectangles
        for new_rect in new_rectangles:
            matched = False

            # Iterate over the existing tracked objects
            for obj_id, obj_rect in self.tracked_objects.items():
                # Calculate the center of the new rectangle
                new_center = (
                    (new_rect[0] + new_rect[2]) / 2,
                    (new_rect[1] + new_rect[3]) / 2,
                )

                # Calculate the center of the existing tracked object's rectangle
                obj_center = (
                    (obj_rect[0] + obj_rect[2]) / 2,
                    (obj_rect[1] + obj_rect[3]) / 2,
                )

                # Calculate the Euclidean distance between the centers
                distance = ((new_center[0] - obj_center[0]) ** 2 +
                            (new_center[1] - obj_center[1]) ** 2) ** 0.5

                # If the distance is within the threshold, update the tracked object
                if distance <= self.max_distance:
                    updated_objects[obj_id] = new_rect
                    matched = True
                    break

            # If no match is found, create a new tracked object
            if not matched:
                updated_objects[self.next_object_id] = new_rect
                self.next_object_id += 1  # Increment the object ID

        # Update the tracked_objects dictionary with the updated objects
        self.tracked_objects = updated_objects

        # Return the updated objects with IDs
        return self.tracked_objects



