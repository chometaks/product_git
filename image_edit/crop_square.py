from PIL import Image
import os

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size),min(pil_img.size))

search_dir = "crop_target/"
ext_list = ["jpg","jpeg","png","bmp"]
img_list = []
for root, dirs, files in os.walk(search_dir):
    for ext in ext_list:
        img_list.extend([os.path.join(root, file) for file in files if ext in file])

for filename in img_list:
    imgtemp=Image.open(filename)
    img=crop_max_square(imgtemp)
    img.save('crop_result/'+filename.split('/')[1].split('.')[0]+'.jpg')

