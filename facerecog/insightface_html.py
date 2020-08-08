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

base = cv2.imdecode(np.fromfile("imgHQ00000_00.png", np.uint8), cv2.IMREAD_COLOR)
faces = fa.get(base, det_thresh=detect_thresh, det_scale=scale)

search_dir = "target_dir/"
ext_list = ["jpg","jpeg","png","bmp"]
img_list = []
for root, dirs, files in os.walk(search_dir):
    for ext in ext_list:
        img_list.extend([os.path.join(root, file) for file in files if ext in file])

faces_list = []
for filename in img_list:
    target_file = cv2.imdecode(np.fromfile(filename,np.uint8), cv2.IMREAD_COLOR)
    temp_faces = fa.get(target_file, det_thresh=detect_thresh, det_scale=scale)
    for ind, temp_face in enumerate(temp_faces):
        temp = {}
        temp["filename"] = filename
        temp["bbox"] = [int(b) for b in temp_face.bbox.tolist()]
        temp["cvdata"] = target_file[temp["bbox"][1] : temp["bbox"][3] , temp["bbox"][0] : temp["bbox"][2]]
        temp["normed_embedding"] = temp_face.normed_embedding
        faces_list.append(temp)


target_embedding = np.array([ f["normed_embedding"] for f in faces_list])

#print(faces)
#print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
#print(target_embedding)


sim = np.dot(faces[0].normed_embedding , target_embedding.T)

for ind , s in enumerate(sim):
	faces_list[ind]["sim"] = s
	if(s > 0.4):
		faces_list[ind]["css_class"] = "honnin"
	else:
		faces_list[ind]["css_class"] = "hoka"

for ind , face in enumerate(faces_list):
	dst_base64 =cv2.imencode('.png', face["cvdata"])[1]
	faces_list[ind]["img"] = "{}".format(base64.b64encode(dst_base64))[2:-1]
env = Environment(loader=FileSystemLoader('./templates/',encoding='utf8'))
tpl = env.get_template('checklist.html')
html = tpl.render({'data':faces_list})
file = open("test.html",'w',encoding='utf8')
file.write(html)
file.close()
