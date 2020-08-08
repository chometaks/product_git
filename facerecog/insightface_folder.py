import insightface
import pandas as pd
import numpy as np
import cv2
import os
import base64
from jinja2 import Environment, FileSystemLoader
from natsort import natsorted

fa = insightface.app.FaceAnalysis()
fa.prepare(0)

detect_thresh = 0.4
scale = 0.8

base = cv2.imdecode(np.fromfile("515*0.jpg", np.uint8), cv2.IMREAD_COLOR)
faces = fa.get(base, det_thresh=detect_thresh, det_scale=scale)

search_dir = "target_dir/"
ext_list = ["jpg","jpeg","png","bmp"]
img_list = []
#for root, dirs, files in os.walk(search_dir):
#    for ext in ext_list:
#        img_list.extend([os.path.join(root, file) for file in files if ext in file])
#natsorted(リスト)で自然なソート　sorted(リスト)だと文字列の扱い
for root, dirs, files in os.walk(search_dir):
    for ext in ext_list:
        img_list.extend([os.path.join(root, file) for file in natsorted(files) if ext in file])

faces_list = []
for filename in img_list:
    target_file = cv2.imdecode(np.fromfile(filename,np.uint8), cv2.IMREAD_COLOR)
    temp_faces = fa.get(target_file, det_thresh=detect_thresh, det_scale=scale)
    for ind, temp_face in enumerate(temp_faces):
        temp = {}
        temp["filename"] = filename.split('/')[1].split('.')[0]
        temp["bbox"] = [int(b) for b in temp_face.bbox.tolist()]
        temp["cvdata"] = target_file[temp["bbox"][1] : temp["bbox"][3] , temp["bbox"][0] : temp["bbox"][2]]
        temp["normed_embedding"] = temp_face.normed_embedding
        faces_list.append(temp)

target_embedding = np.array([ f["normed_embedding"] for f in faces_list])

#print(faces)
#print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
#print(target_embedding)


sim = np.dot(faces[0].normed_embedding , target_embedding.T)

result = []
for ind , s in enumerate(sim):
    faces_list[ind]["sim"] = s
    result.append(s)

resultcsv = pd.DataFrame(img_list, columns = ['filenames'])
resultcsv['result'] = result
resultcsv.to_csv('insightface_result.csv', index=False, encoding='utf-8')

#print(result)


