import os
import cv2
import numpy as np
import time
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from ultralytics import YOLO
import pandas as pd

# 加载YOLOv8模型
detect_weights = 'weight/improved_8n_best.pt'
segment_weights = 'runs/segment/train/weights/best.pt'
detect_model = YOLO(detect_weights)
segment_model = YOLO(segment_weights)

# 确保输出文件夹存在
input_folder = 'test'
output_folder = 'linshi'
os.makedirs(output_folder, exist_ok=True)

# 初始化一个空的DataFrame来存储图片名和处理时间
df = pd.DataFrame(columns=['Image Name', 'Total Processing Time (ms)', 'Detection Time (ms)', 'Segmentation Time (ms)', 'Mask Processing Time (ms)', 'Pathfinding Time (ms)'])

# 遍历文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)

        image = cv2.imread(image_path)

        # 记录开始时间
        start_time = time.time()

        # 执行检测
        detect_start_time = time.time()
        results = detect_model(image)
        detect_end_time = time.time()

        # 获取stem框
        stem_boxes = []
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 1:  # 检查类别是否为stem（类别1）
                    x1, y1, x2, y2 = box.xyxy[0].cpu().int().numpy()
                    stem_boxes.append((x1, y1, x2, y2))

        # 遍历每个stem框
        for box in stem_boxes:
            x1, y1, x2, y2 = box
            crop_img = image[y1:y2, x1:x2]

            segment_results = segment_model(crop_img)


            if isinstance(segment_results, list) and len(segment_results) > 0:
                segment_result = segment_results[0]
                masks = segment_result.masks

                if masks is not None and len(masks.data) > 0:
                    mask_array = masks.data[0].cpu().numpy()
                    mask_image = (mask_array * 255).astype(np.uint8)
                    mask_image_resized = cv2.resize(mask_image, (crop_img.shape[1], crop_img.shape[0]))

                    # 处理mask
                    mask_processing_start_time = time.time()
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image_resized, connectivity=8)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    largest_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                    bool_mask = largest_mask.astype(np.bool_)
                    skeleton = skeletonize(bool_mask)


                    # 寻找骨架的最短路径
                    pathfinding_start_time = time.time()
                    coords = np.column_stack(np.where(skeleton))
                    highest_point = coords[np.argmin(coords[:, 0])]
                    lowest_point = coords[np.argmax(coords[:, 0])]
                    path, cost = route_through_array(1 - skeleton, start=tuple(highest_point), end=tuple(lowest_point), fully_connected=True)
                    shortest_path_skeleton = np.zeros_like(skeleton)
                    for coord in path:
                        shortest_path_skeleton[coord] = 1

                    # 标记中间高度点
                    middle_height = (highest_point[0] + lowest_point[0]) // 2
                    path_coords = np.column_stack(np.where(shortest_path_skeleton > 0))
                    closest_point = path_coords[np.argmin(np.abs(path_coords[:, 0] - middle_height))]
                    closest_point = (closest_point[1] + x1, closest_point[0] + y1)
                    cv2.circle(image, closest_point, 7, (0, 0, 255), -1)

        # 记录结束时间并计算耗时（排除保存图片时间）
        end_time = time.time()
        total_processing_time_ms = (end_time - start_time) * 1000

        # 打印每张图片的处理时间
        print(f'Processing time for {filename}: {total_processing_time_ms:.2f} ms')
        print(f'Detection time: {total_detect_time:.2f} ms')
        print(f'Segmentation time: {total_segment_time:.2f} ms')
        print(f'Mask processing time: {total_mask_processing_time:.2f} ms')
        print(f'Pathfinding time: {total_pathfinding_time:.2f} ms')

        # 将标记后的图像保存到输出文件夹中，保持原文件名
        marked_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(marked_image_path, image)
        print(f'Marked image saved as {marked_image_path}')

        # 将图片名和处理时间添加到DataFrame中
        new_row = pd.DataFrame({
            'Image Name': [filename],
            'Total Processing Time (ms)': [total_processing_time_ms],
            'Detection Time (ms)': [total_detect_time],
            'Segmentation Time (ms)': [total_segment_time],
            'Mask Processing Time (ms)': [total_mask_processing_time],
            'Pathfinding Time (ms)': [total_pathfinding_time]
        })

        # 排除空或全NA的行
        if not new_row.isna().all(axis=1).all():
            df = pd.concat([df, new_row], ignore_index=True)

# 将DataFrame保存到Excel文件
excel_path = 'processing_times2.xlsx'  # 替换为Excel文件路径
df.to_excel(excel_path, index=False)
print(f"处理时间已保存到 {excel_path}")
