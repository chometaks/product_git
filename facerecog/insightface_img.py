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
scale = 1.0

base = cv2.imdecode(np.fromfile("00000200.jpg", np.uint8), cv2.IMREAD_COLOR)
faces = fa.get(base, det_thresh=detect_thresh, det_scale=scale)

#search_dir = "target_dir/"
#ext_list = ["jpg","jpeg","png","bmp"]
#img_list = []
#for root, dirs, files in os.walk(search_dir):
#    for ext in ext_list:
#        img_list.extend([os.path.join(root, file) for file in files if ext in file])

#faces_list = []
#for filename in img_list:
#    target_file = cv2.imdecode(np.fromfile(filename,np.uint8), cv2.IMREAD_COLOR)
#    temp_faces = fa.get(target_file, det_thresh=detect_thresh, det_scale=scale)
#    for ind, temp_face in enumerate(temp_faces):
#        temp = {}
#        temp["filename"] = filename
#        temp["bbox"] = [int(b) for b in temp_face.bbox.tolist()]
#        temp["cvdata"] = target_file[temp["bbox"][1] : temp["bbox"][3] , temp["bbox"][0] : temp["bbox"][2]]
#        temp["normed_embedding"] = temp_face.normed_embedding
#        faces_list.append(temp)

targetname = "00000200.jpg"
target_file = cv2.imdecode(np.fromfile(targetname,np.uint8), cv2.IMREAD_COLOR)
target_faces = fa.get(target_file, det_thresh=detect_thresh, det_scale=scale)
target = {}
target["filename"] = targetname
target["bbox"] = [int(b) for b in target_faces[0].bbox.tolist()]
target["cvdata"] = target_file[target["bbox"][1] : target["bbox"][3] , target["bbox"][0] : target["bbox"][2]]
target["normed_embedding"] = target_faces[0].normed_embedding


#target_embedding = np.array([ f["normed_embedding"] for f in faces_list])

target_embedding = np.array([ target["normed_embedding"] ])


sim = np.dot(faces[0].normed_embedding , target_embedding.T)


#result = []
#for ind , s in enumerate(sim):
#    faces_list[ind]["sim"] = s
#    result.append(s)

#print(result)

print(sim)
