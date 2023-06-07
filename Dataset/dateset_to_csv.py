import csv
import cv2
import os 
import glob
from PIL import Image
from numpy import asarray
import pandas as pd

emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
emotion_map = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5, "surprise": 6}
data_name = {"train" : 'Training', "test" : 'PublicTest'}
dataset = ["train", "test"]

with open('dataset2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["emotion","pixels","Usage"]
    writer.writerow(field)

for data in dataset:
    for emotion in emotions:
        img_dir = f"dataset2//{data}//{emotion}" # Enter Directory of all images  
        data_path = os.path.join(img_dir,'*g') 
        files = glob.glob(data_path) 
        for f1 in files: 
            image = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
            pixel_data = ' '.join(map(str, image.flatten()))
                
            with open("dataset2.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([emotion_map[emotion], pixel_data, data_name[data]])
    
    
