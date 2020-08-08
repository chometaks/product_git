#csvからスコアを読んでファイル名を 識別子_元ファイル名_スコア.jpg にする
import os
from natsort import natsorted
import pandas as pd
import csv

search_dir = "target_dir/"
ext_list = ["jpg","jpeg","png","bmp"]
img_list = []
for root, dirs, files in os.walk(search_dir):
    for ext in ext_list:
        img_list.extend([os.path.join(root, file) for file in natsorted(files) if ext in file])

with open('insightface_result_P1ES1C1_p2.csv') as f:
    reader=csv.reader(f)
    csv_list = [row for row in reader]

for filename in img_list:
    for i,l in enumerate(csv_list):
        if l[0]==filename:
            print(l[0])
            print('aaaaaaaaaaa')
            print(filename)
            os.rename(filename, search_dir+'P1ES1C1p2_'+filename.split('/')[1].split('.')[0]+'_'+l[1]+'.jpg')