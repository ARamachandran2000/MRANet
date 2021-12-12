import json
import os
import time
import datetime
import numpy as np
import matplotlib
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageChops
from shapely import wkt
from shapely.geometry.multipolygon import MultiPolygon
from google.colab.patches import cv2_imshow
import cv2


damage_dict = {
    "no-damage": (0, 255, 0, 255),
    "minor-damage": (0, 0, 255, 255),
    "major-damage": (255, 69, 0, 255),
    "destroyed": (255, 0, 0, 255),
    "un-classified": (255, 255, 255, 255)
}


def get_damage_type(properties):
    if 'subtype' in properties:
        return properties['subtype']
    else:
        return 'no-damage'

def annotate_img(draw, coords):
    wkt_polygons = []

    for coord in coords:
        damage = get_damage_type(coord['properties'])
        wkt_polygons.append((damage, coord['wkt']))

    polygons = []

    for damage, swkt in wkt_polygons:
        polygons.append((damage, wkt.loads(swkt)))

    for damage, polygon in polygons:
        x, y = polygon.exterior.coords.xy
        coords = list(zip(x, y))
        draw.polygon(coords, damage_dict[damage])

    del draw

def read_label(label_path):
    with open(label_path) as json_file:
        image_json = json.load(json_file)
        return image_json


def display_img(json_path, time='post', annotated=True):
    if time == 'pre':
        json_path = json_path.replace('post', 'pre')

    img_path = json_path.replace('Labels', 'Images').replace('json', 'png')

    image_json = read_label(json_path)
    img_name = image_json['metadata']['img_name']

    print(img_name)

    #img = Image.open(img_path)
    img = Image.open("/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/black.png")
    
    draw = ImageDraw.Draw(img, 'RGBA')

    if annotated:
        annotate_img(draw, image_json['features']['xy'])

    return img,img_name

files = [f for f in os.listdir("./Labels_json")]

for i in range(0,len(files)):
  files[i] = os.path.join("./Labels_json",files[i])


for f in files:
  img,img_name = display_img(f)
  img = np.array(img)
  cv2.imwrite("./Labels/" + img_name,img)



