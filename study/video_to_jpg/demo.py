from keras.models import load_model
import numpy as np
import cv2
from keras import backend as K
import os
import shutil
import datetime
from PIL import ImageFont, ImageDraw, Image
import subprocess

# 動画関係準備
target = 'mcem0_head2.mp4'
result = target + 'output.m4v'
movie = cv2.VideoCapture(target)
fps = movie.get(cv2.CAP_PROP_FPS)
height = movie.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = movie.get(cv2.CAP_PROP_FRAME_WIDTH)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(result, int(fourcc), fps, (int(width), int(height)))

model = load_model('C:/Users/taks/grastudy/study/mini_model.h5')

score_list = []

def frame_to_3d(video_path, frame_num, result_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(result_path, frame)
    
    subprocess.run("C:/Users/taks/grastudy/jpgto3dtoscore.bat")

    shutil.move('C:/Users/taks/grastudy/PRNet/3d/faces/temp.jpg', 'C:/Users/taks/grastudy/study/video_to_jpg/' + target + '.jpg')
    shutil.move('C:/Users/taks/grastudy/PRNet/3d/models/faces/temp.obj', 'C:/Users/taks/grastudy/study/video_to_jpg/' + target + '.obj')

# 各フレームへの処理
if movie.isOpened() is True:
    ret, frame = movie.read()
    f_h, f_w = frame.shape[:2]
else:
    ret = False
while ret:
    img_width, img_height = 128, 128
    image = cv2.resize(frame, (img_width, img_height))
    #ここの意味？
    image = image.transpose(2, 0, 1)
    image = image / 255
    #4次元の入力にしなきゃいけない(1, 128, 128, 3)
    #1,3である意味？
    image = image.reshape(1, img_width, img_height, 3)
    pred = model.predict(image)

    print('score:',pred[0,0])

    score = str(pred[0,0])
    
    score_list.append(score)
    score_show_number = 0
    if movie.get(cv2.CAP_PROP_POS_FRAMES) % 1 == 0:
        score_show_number = score_show_number + int(movie.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("arial.ttf", 30)
    draw.text((10, 10), score_list[score_show_number], font = font)
    rgb_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # フレーム書き込み
    out.write(rgb_image)
    ret, frame = movie.read()
    # 50フレームごとに経過を出力
    if movie.get(cv2.CAP_PROP_POS_FRAMES) % 1 == 0:
        print(datetime.datetime.now().strftime('%H:%M:%S'),
              '現在フレーム数：' + str(int(movie.get(cv2.CAP_PROP_POS_FRAMES))))

    # 途中で終了する場合コメントイン
    # if movie.get(cv2.CAP_PROP_POS_FRAMES) > 500:
    #     break

frame_num = score_list.index(min(score_list)) + 1

frame_to_3d(target, frame_num, 'C:/Users/taks/grastudy/PRNet/3d/faces/temp.jpg')

#model = load_model('mini_model.h5')
#image = cv2.imread('personne15159+15+0.jpg')
#img_width, img_height = 128, 128
#image = cv2.resize(image, (img_width, img_height))
#ここの意味？
#image = image.transpose(2, 0, 1)
#image = image / 255
#4次元の入力にしなきゃいけない(1, 128, 128, 3)
#1,3である意味？
#image = image.reshape(1, img_width, img_height, 3)
#pred = model.predict(image)
#print('score:',pred)