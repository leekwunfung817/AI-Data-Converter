import pycocotools.mask as mask
import cv2

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='.'
fullPath = '{}/{}.json'.format(dataDir,'Hasty.ai-CompactRLE')
targetFullPath = '{}/{}.json'.format(dataDir,'CocoAnnotator-PolygonFormat')

import json

print('Loading JSON')
dataset = json.load(open(fullPath, 'r'))
print('Info:',dataset['info'])
print('Categories:',len(dataset['categories']))
print('Images:',len(dataset['images']))
print('Annotations:',len(dataset['annotations']))

images={}
for objs in dataset['images']:
	images[objs['id']]=objs

def polygonFromMask(maskedArr): # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
        contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())
        RLEs = mask.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
        RLE = mask.merge(RLEs)
        # RLE = mask.encode(np.asfortranarray(maskedArr))
        area = mask.area(RLE)
        [x, y, w, h] = cv2.boundingRect(maskedArr)
        return segmentation #, [x, y, w, h], area
def convertAnnotationCompressedToMask(dataset, index):
	ann = dataset['annotations'][index]
	# print('Ann:',ann)
	# print('m',m)
	# print('index type:',type(index))
	# print("dataset['annotations']",type(dataset['annotations']))
	# print("dataset['annotations'][index]",type(dataset['annotations'][index]))
	# print("dataset['annotations'][index]['segmentation']",type(dataset['annotations'][index]['segmentation']))
	if type(dataset['annotations'][index]['segmentation']) == list:
		return dataset
	# print("dataset['annotations'][index]['segmentation']['counts']",type(dataset['annotations'][index]['segmentation']['counts']))
	maskedArr = mask.decode(ann['segmentation'])
	m=polygonFromMask(maskedArr)
	dataset['annotations'][index]['segmentation']=m
	ann = dataset['annotations'][index]
	# print('Ann:',ann)
	return dataset
def convertAnnotationsCompressedToMask(dataset):
	for i in range(len(dataset['annotations'])):
		dataset=convertAnnotationCompressedToMask(dataset,i)
	return dataset

dataset = convertAnnotationsCompressedToMask(dataset)
# json.dumps(dataset, indent=4)
with open(targetFullPath, 'w') as outfile:
	json.dump(obj=dataset, fp=outfile)

print('From:',fullPath)
print('To:',targetFullPath)
