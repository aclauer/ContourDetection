import numpy as np
import cv2 as cv
import pandas as pd
import os
import json

from pysolotools.consumers import Solo

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
    print(segmentation.shape)
    width, height, depth = segmentation.shape
    print(width)
    print(height)
    colors = get_colors(f'{sequence_path}/step0.frame_data.json')
    #print(colors)
    id_to_color = {}
    for color in colors:
        labelId = labels[color['labelName']]
        #print(labelId)
        id_to_color[labelId] = color['pixelValue'][0:3]
    #print(id_to_color)

    i = 0
    for id in id_to_color.keys():
        i += 1
        # Iterate through each class present in the image
        print(id)
        print(type(id_to_color[id]))
        color = np.array(id_to_color[id])
        mask = isolate_color(segmentation, np.flip(color))
        cv.imshow("Window", mask)
        cv.waitKey(0)
        
        contours = get_polygon(mask)
        print(len(contours))
        cv.drawContours(segmentation, contours, -1, (127,127,127), 3)
        cv.imshow("Window", segmentation)
        cv.waitKey(0)
        #print(id_to_color[id])
        #print(contours)
        add_label_to_file('label.txt', width, height, id, contours)
    #print(i)

def add_label_to_file(path, width, height, label_id, contours):
    f = open(path, 'a')
    for contour in contours:
        # Write class id to a new line
        f.write(f"{label_id}")
        for point in contour:
            f.write(f" {point[0][0]/width} {point[0][1]/height}")
        f.write("\n")
    f.close()

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
    #print(mask.shape)
    return mask

def get_polygon(mask):
    #imgray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(mask, 127, 255, 0)
    contours, heirarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #print("contour shape: " + str(np.shape(contours[0])))
    #print(contours)
    #print(type(contours))
    #print(len(contours))
    #for instance in contours:
    #    for point in instance:
            #print(point[0])
    #        print(f"{point[0][0]} {point[0][1]}")
        #print(instance)
    #    print(type(instance))
    #    print(np.shape(instance))
        #print(instance[0][0][1])
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
process_sequence('/Users/andrewlauer/Downloads/solo/sequence.0', labels)

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