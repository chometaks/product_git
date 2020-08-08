import insightface
import pandas as pd
import numpy as np
import cv2
import os
import base64
from jinja2 import Environment, FileSystemLoader

fa = insightface.app.FaceAnalysis()
fa.prepare(0)

detect_thresh = 0.4
scale = 0.8

#base = cv2.imdecode(np.fromfile("imgHQ00000_00.png", np.uint8), cv2.IMREAD_COLOR)
#faces = fa.get(base, det_thresh=detect_thresh, det_scale=scale)

search_dir = "target_dir/"
ext_list = ["jpg","jpeg","png","bmp"]
img_list = []
for root, dirs, files in os.walk(search_dir):
    for ext in ext_list:
        img_list.extend([os.path.join(root, file) for file in files if ext in file])

#faces_list = []
for filename in img_list:
    target_file = cv2.imdecode(np.fromfile(filename,np.uint8), cv2.IMREAD_COLOR)
    temp_faces = fa.get(target_file, det_thresh=detect_thresh, det_scale=scale)
    for ind, temp_face in enumerate(temp_faces):
        temp = {}
        temp["filename"] = filename
        temp["bbox"] = [int(b) for b in temp_face.bbox.tolist()]
        temp["cvdata"] = target_file[temp["bbox"][1] : temp["bbox"][3] , temp["bbox"][0] : temp["bbox"][2]]
        temp["normed_embedding"] = temp_face.normed_embedding
        print(temp["bbox"])
        print(temp["cvdata"])
        print(temp["filename"])
        cv2.rectangle(target_file, (temp["bbox"][0],temp["bbox"][1]), (temp["bbox"][2], temp["bbox"][3]), (255,0,0), 5)
        cv2.imwrite('/home/taks/mystudy/myproduct/facerecog/result/'+temp["filename"].split('/')[1].split('.')[0]+'_insightface.jpg',target_file)
