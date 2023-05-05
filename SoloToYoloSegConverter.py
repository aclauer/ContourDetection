import numpy as np
import cv2 as cv
import pandas as pd
import os
import json
import shutil

from pysolotools.consumers import Solo

class SoloToYoloSegConverter:
    def __init__(self, solo_path):
        self.solo_path = solo_path

    def convert(self, dataset_name, output_path, train_split=0.8):
        solo = Solo(data_path=self.solo_path)

        # Make output directories
        os.makedirs(os.path.join(output_path, "dataset", "images", "train"))
        os.makedirs(os.path.join(output_path, "dataset", "images", "val"))
        os.makedirs(os.path.join(output_path, "dataset", "labels", "train"))
        os.makedirs(os.path.join(output_path, "dataset", "labels", "val"))

        dirs = os.listdir(self.solo_path)
        sequence_paths = []
        #annotations_path = f'{self.solo_path}/annotation_definitions.json'
        #print(annotations_path)
        print("here")
        annotations_path = os.path.join(self.solo_path, "annotation_definitions.json")
        print(annotations_path)
        print("Now we are here")
        labels = self.get_labels(annotations_path)
        i = 0
        for elem in dirs:    
            if elem[0:9] == 'sequence.':
                print("i: " + str(i))
                #sequence_paths.append(f"{self.solo_path}/{elem[0:10]}")
                sequence_path = os.path.join(self.solo_path, elem[0:10])
                label_output_path = os.path.join(output_path, "dataset", "labels", "train", str(i))
                #print("Label output path: " + label_output_path)
                self.process_sequence(sequence_path, labels, label_output_path)
                image_source_path = os.path.join(self.solo_path, elem[0:10], "step0.camera.png")
                image_destination_path = os.path.join(output_path, "dataset", "images", "train", f"{i}.png")
                shutil.copy(image_source_path, image_destination_path)
                i += 1

        # Move correct number of files to validation set
        training_path = os.path.join(output_path, "dataset", "labels", "train")
        training_samples = os.listdir(training_path)
        num_val_samples = int(len(training_samples) * (1-train_split))
        print(f"Moving {num_val_samples} samples to validation")

        for i in range(len(training_samples) - num_val_samples, len(training_samples)):
            source_file = os.path.join(output_path, "dataset", "labels", "train", str(f'{i}.txt'))
            destination_file = os.path.join(output_path, "dataset", "labels", "val", str(f'{i}.txt'))
            shutil.copy(source_file, destination_file)

            source_file = os.path.join(output_path, "dataset", "images", "train", str(f'{i}.png'))
            destination_file = os.path.join(output_path, "dataset", "images", "val", str(f'{i}.png'))
            shutil.copy(source_file, destination_file)



    def process_sequence(self, sequence_path, labels, output_path):
        segmentation = cv.imread(f'{sequence_path}/step0.camera.SemanticSegmentation.png')
        width, height, depth = segmentation.shape
        frame_data_path = os.path.join(sequence_path, "step0.frame_data.json")
        colors = self.get_colors(frame_data_path)
        id_to_color = {}
        for color in colors:
            labelId = labels[color['labelName']]
            id_to_color[labelId] = color['pixelValue'][0:3]

        for id in id_to_color.keys():
            color = np.array(id_to_color[id])
            mask = self.isolate_color(segmentation, np.flip(color))
            contours = self.get_polygon(mask)
            self.add_label_to_file(f'{output_path}.txt', width, height, id, contours)

    @staticmethod
    def add_label_to_file(path, width, height, label_id, contours):
        f = open(path, 'a')
        for contour in contours:
            # Write class id to a new line
            f.write(f"{label_id}")
            for point in contour:
                f.write(f" {point[0][0]/width} {point[0][1]/height}")
            f.write("\n")
        f.close()

    @staticmethod
    def get_colors(frame_data_path):
        with open(frame_data_path) as json_file:
            data = json.load(json_file)
            capture = data['captures'][0]
            print(type(capture['annotations'][0]))
            for annotation in capture['annotations']:
                print(type(annotation))
                if annotation["id"] == "SemanticSegmentation":
                    instances = annotation['instances']
            return instances
    
    @staticmethod
    def get_labels(annotation_definition_path):
        class_mappings = {}
        with open(annotation_definition_path) as json_file:
            definitions = json.load(json_file)
            annotation_definitions = definitions['annotationDefinitions'][0]
            labels = annotation_definitions['spec']
            for label in labels:
                class_mappings[label['label_name']] = label['label_id']
        return class_mappings

    @staticmethod
    def isolate_color(img, color):
        return cv.inRange(img, color, color)
    
    @staticmethod
    def get_polygon(mask):
        ret, thresh = cv.threshold(mask, 127, 255, 0)
        contours, heirarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return contours