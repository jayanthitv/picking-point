from PIL import Image
import cv2
import numpy as np
from skimage import morphology
import os
from openpyxl import Workbook
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

import subprocess

def find_blue_extremes(image):
    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 定义蓝色的范围
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # 创建一个蓝色的掩码
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # 找到掩码中非零的坐标
    coords = np.nonzero(mask)
    # 如果没有找到蓝色像素，返回None
    if len(coords[0]) == 0 or len(coords[1]) == 0:
        return None
    # 否则，找到y坐标的最小值和最大值，对应于最低和最高的蓝色像素点
    min_y = np.min(coords[0])
    max_y = np.max(coords[0])
    # 找到对应的x坐标，可能有多个，随机选取一个
    min_x = np.random.choice(coords[1][coords[0] == min_y])
    max_x = np.random.choice(coords[1][coords[0] == max_y])
    # 返回起点和终点的坐标，四舍五入为整数
    return (int(round(min_x)), int(round(min_y))), (int(round(max_x)), int(round(max_y)))


def find_blue_path(image, start, end):
    # 定义一个列表，存储已经访问过的坐标
    visited = []
    # 定义一个队列，存储待访问的坐标和路径
    queue = [(start, [start])]
    # 定义一个变量，存储最短路径
    shortest_path = None
    # 定义一个变量，存储最短路径的长度
    shortest_length = float('inf')
    # 定义一个变量，存储图像的高度和宽度
    height, width = image.shape[:2]
    # 定义一个循环，直到队列为空或找到终点
    while queue and shortest_path is None:
        # 从队列中弹出一个元素，得到当前的坐标和路径
        current, path = queue.pop(0)
        # 如果当前的坐标等于终点，更新最短路径和长度，跳出循环
        if current == end:
            shortest_path = path
            shortest_length = len(path)
            break
        # 否则，遍历当前坐标周围的8个像素
        for i in range(-1, 2):
            for j in range(0, 2):
                # 计算新的坐标
                new_x = current[0] + i
                new_y = current[1] + j
                # 检查新的坐标是否在图像范围内，是否是蓝色像素，是否已经访问过
                if 0 <= new_x < width and 0 <= new_y < height and image[new_y, new_x, 0] > image[new_y, new_x, 1] and image[new_y, new_x, 0] > image[new_y, new_x, 2] and (new_x, new_y) not in visited:
                    # 将新的坐标和路径加入队列
                    queue.append(((new_x, new_y), path + [(new_x, new_y)]))
                    # 将新的坐标加入已访问列表
                    visited.append((new_x, new_y))
    # 返回最短路径
    return shortest_path


def draw_blue_path(image, path):
    # 复制原始图像
    result = image.copy()
    # 如果路径存在，遍历路径中的坐标
    if path:
        for x, y in path:
            # 将对应的像素设为白色
            result[y, x, 0] = 255
            result[y, x, 1] = 255
            result[y, x, 2] = 255
    # 返回结果图像
    return result

def find_largest_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    return max_contour
def process_and_apply_morphology(input_image_path):
    # 读取图片，保留透明度
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # 第一步：检查是否存在白色区域，如果没有直接返回None
    if not np.any(input_image == 255):
        return None

    # 找到最大的白色连通区域并消除其他白色连通区域
    white_mask = (input_image == 255).astype(np.uint8)
    max_white_contour = find_largest_contour(white_mask)

    result_image_1 = np.zeros_like(input_image)
    if max_white_contour is not None:
        cv2.drawContours(result_image_1, [max_white_contour], -1, 255, thickness=cv2.FILLED)

        # 将其他白色连通区域消除
        input_image = cv2.bitwise_and(input_image, result_image_1)

        # 第二步：保留与边界相接触的黑色连通区域，其他转为白色
        black_mask = (input_image == 0).astype(np.uint8)
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_image_2 = np.ones_like(input_image) * 255  # 初始化为白色图像
        for contour in contours:
            # 检查轮廓是否与图像边界相接触
            if any(point[0][0] == 0 or point[0][1] == 0 or point[0][0] == input_image.shape[1]-1 or point[0][1] == input_image.shape[0]-1 for point in contour):
                cv2.drawContours(result_image_2, [contour], -1, 0, thickness=cv2.FILLED)

        # 第三步：对图片进行闭运算
        kernel = np.ones((7, 7), np.uint8)
        result_image_3 = cv2.morphologyEx(result_image_2, cv2.MORPH_CLOSE, kernel)

        # 读取灰度图像
        gray_image = result_image_3

        # 使用自适应阈值二值化，确保输入图像是二值图像
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

        # 进行骨架化
        skeleton_image = morphology.skeletonize(binary_image > 0)

        # 将布尔值图像转换为8位灰度图像
        skeleton_image = skeleton_image.astype(np.uint8) * 255

        # 阈值化图像以提取白色区域
        _, white_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(white_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个三通道彩色图像（RGB），大小和原图一样，背景为黑色
        output_image = np.zeros_like(result_image_3)

        # 将骨架线的坐标在新图像上用蓝色像素点表示出来，蓝色的值为 (255, 0, 0)
        blue_channel = np.zeros_like(input_image)
        blue_channel[skeleton_image == 255] = 255

        output_image = cv2.merge((blue_channel, np.zeros_like(input_image), np.zeros_like(input_image)))

        output_image[skeleton_image == 255] = (255, 0, 0)

        # 找到蓝色像素的最低和最高点
        extremes = find_blue_extremes(output_image)
        # 如果找到了，计算最短路径
        if extremes:
            start, end = extremes
            path = find_blue_path(output_image, start, end)
        # 否则，路径为空
        else:
            path = None
        # 生成一张彩色图像，将最短路径用红色像素表示
        output_image = draw_blue_path(output_image, path)

        # 读取图片并转换为灰度图
        image = output_image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 创建一个掩膜，只保留白色的像素
        mask = cv2.inRange(gray, 255, 255)

        # 找到掩膜中的所有非零像素的坐标
        coords = np.nonzero(mask)

        # 如果没有找到白色像素，就返回None
        if len(coords[0]) == 0 or len(coords[1]) == 0:
            return None

        # 找到最高的白色像素点的坐标
        max_row = np.min(coords[0])
        max_col = coords[1][np.argmin(coords[0])]

        # 找到最低的白色像素点的坐标
        min_row = np.max(coords[0])
        min_col = coords[1][np.argmax(coords[0])]


        mid_row = ( min_row + max_row) // 2
        mid_col = coords[1][np.argmin(np.abs(coords[0] - mid_row))]

        # 返回目标点的坐标
        return mid_col, input_image.shape[0]-mid_row  # 调整坐标顺序为(x, y)即水平方向向右为正，垂直方向向下为正

    # 如果没有白色区域，返回None
    return None


def positioning_picking_point(folder_path):
    images_dict = {}
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            # 调用处理图像的函数
            target_coords = process_and_apply_morphology(image_path)
            # 将图片名和坐标点添加到字典中
            images_dict[filename] = target_coords
    return images_dict



def create_empty_excel_file(path,filename):
    # 创建一个空的DataFrame
    df = pd.DataFrame()

    # 创建完整的文件路径
    full_path = f"{path}/{filename}"

    # 创建目录（如果不存在）
    #os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # 将DataFrame写入Excel文件
    df.to_excel(full_path, index=False)

def create_directory_if_not_exists(directory_path):

    os.makedirs(directory_path, exist_ok=True)




