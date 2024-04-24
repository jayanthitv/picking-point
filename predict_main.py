import cv2
from ultralytics import YOLO
import os
import pandas as pd
import subprocess
from step2_picking import *
from hrnet.hrnet import *
import time
from PIL import Image, ImageDraw


def read_txt_files(folder_path, name):
    data_dict = {}
    counter = 1
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                # 读取txt文件的每一行
                for line in file:
                    parts = line.strip().split()
                    category = parts[0]
                    # 检查第一列是否为1
                    if category == '1':
                        # 分离文件名和后缀
                        name_parts = name.rsplit('.', 1)
                        base_name = name_parts[0]
                        extension = name_parts[1] if len(name_parts) > 1 else ''
                        # 在文件名和后缀之间插入计数器
                        key_name = f'{base_name}{counter}.{extension}' if counter > 1 else f'{name}'
                        # 如果键名已存在，增加计数器
                        while key_name in data_dict:
                            counter += 1
                            key_name = f'{base_name}{counter}.{extension}'
                        # 将后四位数据作为键值
                        data_dict[key_name] = list(map(float, parts[1:]))
                        counter += 1
    return data_dict




input_folder_path='3-mixture'
timelist=[]
for image_name1 in os.listdir(input_folder_path):
    # Check if the file is an image
    if image_name1.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # Construct the full path to the image
        source = os.path.join(input_folder_path, image_name1)
        # Run the model prediction
        # ******************* 1 *****************************************************
        # 模型推理
        model = YOLO('weight/improved_8n_best.pt')
        original_image = cv2.imread(source)
        image_name = os.path.basename(source)
        parts = image_name.split(".")
        file_name = parts[0]
        model.predict(source=source, name=file_name, **{'save': True, 'save_txt': True, 'save_crop': True})

        data = {}
        image_size = original_image.shape
        data['Image'] = image_name
        data['Image_address'] = source
        data['Image_output'] = "runs/detect/" + file_name
        data['Width'] = image_size[1]
        data['Height'] = image_size[0]

        txt_path = data['Image_output'] + "/labels"
        stem = read_txt_files(txt_path, image_name)
        print("stem",stem)

        # ******************* 2 *****************************************************
        hrnet = HRnet_Segmentation()
        mode = "dir_predict"
        count = False
        name_classes = ["background", "stem"]
        dir_origin_path = data['Image_output'] + "/crops/stem"
        if not os.path.exists(dir_origin_path):
            create_directory_if_not_exists(dir_origin_path)
        dir_save_path = data['Image_output'] + f"/crops/seg"

        if mode == "dir_predict":
            import os
            from tqdm import tqdm

            img_names2 = os.listdir(dir_origin_path)
            for img_name2 in tqdm(img_names2):
                if img_name2.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path = os.path.join(dir_origin_path, img_name2)
                    image = Image.open(image_path)
                    r_image = hrnet.detect_image(image)
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    r_image.save(os.path.join(dir_save_path, img_name2))
        seg_folder_path = dir_save_path  # 替换为实际的文件夹路径

        if not os.path.exists(seg_folder_path):
            create_directory_if_not_exists(seg_folder_path)

        start_time = time.time()
        points_list = []
        # 第一步：遍历文件夹下的所有图片
        for image_name3 in os.listdir(seg_folder_path):
            if image_name3.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):

                image_path = os.path.join(seg_folder_path, image_name3)
                # 调用处理图像的函数
                target_coords = process_and_apply_morphology(image_path)
                print("first",target_coords)

                # 第二步：检查图片名是否在stems字典中
                try:
                    x_center, y_center, w, h = stem[image_name3]

                    # 第三步：按公式计算pointx和pointy
                    pointx = data['Width'] * x_center - data['Width'] * w / 2 + target_coords[0]
                    pointy = data['Height'] * y_center - data['Height'] * h / 2 + target_coords[1]
                    # 第四步：将生成的(pointx, pointy)添加到列表中
                    points_list.append((pointx, pointy))
                    print("second", points_list)
                except Exception as e:
                    continue


        last_image = Image.open(source)
        draw = ImageDraw.Draw(last_image)

        # 在每个坐标处画一个红点
        for point in points_list:
            x, y = point
            print("third", point)
            # 画一个小圆来表示点，这里的30是圆的直径
            draw.ellipse((x - 15, y - 15, x + 15, y + 15), fill='red', outline='red')

        # 保存图片

        output_folder = "2"
        os.makedirs(output_folder, exist_ok=True)
        output_image_file = os.path.join(output_folder, os.path.basename(source))
        last_image.save(output_image_file)

        end_time = time.time()

        # 计算并打印运行时间
        execution_time = end_time - start_time
        timelist.append(execution_time)
        print(f"程序运行时间：{execution_time}秒")
print(sum(timelist)/len(timelist))


