# -*- coding：utf-8 -*- #
# CREATED BY： zhangchang
# CREATED BY:  2024/1/26 下午12:54
# LAST MODIFIED ON:
# AIM:
import sys

from ultralytics import YOLO
import cv2
import os
import tqdm

imgPath = "/home/zhangchang/data/smart_class/Video2.2student/"
file_lst = os.listdir(imgPath)
file_lst.sort(key=lambda x: int(x[:-4]))
# Load a pretrained YOLOv8n model
#
model = YOLO('/home/zhangchang/graduate/yolov5/runs/detect/best23.pt')
lens = len(os.listdir(imgPath))

save_path = "/home/zhangchang/graduate/ultralytics/test/all_json/"
img_name = len(os.listdir(save_path))
cnt = img_name // 5000 + 1
print(cnt)

for i in range(0,lens,5000):
# Define path to video file
    source = f"/home/zhangchang/gitlab/deep-learning-model-factory/cv/test/test_data/out_vedio/outall{cnt}.mp4"
    cnt += 1
    print(cnt-1)
    #test_source = "/home/zhangchang/graduate/ultralytics/mydata/testvedio.mp4"
    #source = "/home/zhangchang/graduate/ultralytics/mydata/images/3157.jpg"

    # Get the video capture object
    results = model(source)
    for r in tqdm.tqdm(results):
        image_name = f"{file_lst[img_name]}"
        data = r.mytojson(image_name=image_name)
        with open(os.path.join(save_path,str(file_lst[img_name])[:-4] + ".json"), 'w') as f:
            f.write(data)
        img_name += 1
    del data,results

