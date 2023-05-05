import numpy as np
import cv2 as cv
import pandas as pd
import os
import json
print("Imported the basics.")

from pysolotools.consumers import Solo
print("Imported pysolotools.")

pd.set_option('display.max_rows', None)

solo = Solo(data_path="/Users/andrewlauer/Downloads/solo")

dirs = os.listdir('/Users/andrewlauer/Downloads/solo')
#print(dirs)

sequences = []
"""
for elem in dirs:
    if elem[0:9] == 'sequence.':
        sequences.append(f'/Users/andrewlauer/Downloads/solo/{elem[0:10]}')
        #print(elem[0:10])
        #print(os.listdir(f'/Users/andrewlauer/Downloads/solo/{elem[0:10]}'))

for sequence in sequences:
    print(sequence)
"""
"""
for frame in solo.frames():
    captures = frame.get_captures_df()
    print(type(captures.loc[0]))
    print(frame.get_file_path(captures.loc[0]))
    annotations = captures['annotations'][0]
    #print("annotations type: " + str(type(annotations)))
    for annotation in annotations:
        if annotation['id'] == "SemanticSegmentation":
            mask_file = annotation['filename']
            colors = annotation['instances']
"""

def process_sequence(sequence_path, labels):
    segmentation = cv.imread(f'{sequence_path}/step0.camera.SemanticSegmentation.png')
    colors = get_colors(f'{sequence_path}/step0.frame_data.json')
    print(colors)
    id_to_color = {}
    for color in colors:
        labelId = labels[color['labelName']]
        id_to_color[labelId] = color['pixelValue'][0:3]
    print(id_to_color)

    for id in id_to_color.keys():
        # Iterate through each class present in the image
        print(id_to_color[id])
        mask = isolate_color(segmentation, np.array(id_to_color[id]))
        contours = get_polygon(mask)
        #print(contours)


def get_colors(frame_data_path):
    with open(frame_data_path) as json_file:
        data = json.load(json_file)
        capture = data['captures'][0]
        annotations = capture['annotations'][0]
        instances = annotations['instances']
        return instances

def get_labels(annotation_definition_path):
    class_mappings = {}
    with open(annotation_definition_path) as json_file:
        definitions = json.load(json_file)
        annotation_definitions = definitions['annotationDefinitions'][0]
        labels = annotation_definitions['spec']
        for label in labels:
            class_mappings[label['label_name']] = label['label_id']
    return class_mappings



def isolate_color(img, color):
    mask = cv.inRange(img, color, color)
    print(mask.shape)
    return mask

def get_polygon(mask):
    #imgray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(mask, 127, 255, 0)
    contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("contour shape: " + str(np.shape(contours[0])))
    print(type(contours))
    return contours

def _extract_info(frame):
    captures = frame.get_captures_df()
    annotations = captures['annotations'][0]
    #print("annotations type: " + str(type(annotations)))
    for annotation in annotations:
        if annotation['id'] == "SemanticSegmentation":
            mask_file = annotation['filename']
            colors = annotation['instances']
            return mask_file, colors

labels = get_labels('/Users/andrewlauer/Downloads/solo/annotation_definitions.json')
process_sequence('/Users/andrewlauer/Downloads/solo/sequence.2', labels)

#def _get_polygon(mask, color):
#    cv.load

#def _process_mask():
"""
i = 0
for frame in solo.frames():
    print(i)

    #print(frame.get_file_path)

    mask, colors = _extract_info(frame)


    #print(annotations[0][1])
    #print(captures['annotations'].at[1])
    #print(type(captures['annotations']))
    #print("here")
    #annotations = captures['annotations']
    #print(annotations.at[0][1])
    print()
    
    i += 1

"""

"""
im = cv.imread('0003.png')
cv.imshow("Image Window", im)
cv.waitKey(0)

imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#print(np.shape(contours))
print(contours)

cv.drawContours(im, contours, -1, (0,255,0), 3)

cv.namedWindow("Image Display")
cv.imshow("Image Window", im)
cv.waitKey(0)"""
"""
cv.namedWindow("Image Display")
img = cv.imread('step0.camera.SemanticSegmentation.png')

print(img.shape)

cv.imshow("Image Window", img)
cv.waitKey(0)
mask = isolate_color(img, np.array([0, 255, 0]))

cv.imshow("Image Window", mask)
cv.waitKey(0)

print(mask.shape)
#mask.reshape([640, 640, 3])
#print(mask.shape)
#
ret, thresh = cv.threshold(mask, 127, 255, 0)
contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#print(np.shape(contours))
print(contours)

cv.drawContours(img, contours, -1, (0,255,255), 3)

cv.namedWindow("Image Display")
cv.imshow("Image Window", img)
cv.waitKey(0)
"""
"""
contours = get_polygon(mask)
cv.drawContours(mask, contours, -1, (0,255,0), 3)
cv.imshow("Image Window", mask)
cv.waitKey(0)"""