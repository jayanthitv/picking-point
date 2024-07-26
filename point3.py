import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from ultralytics import YOLO

def load_yolov8_model(weights_path):
    model = YOLO(weights_path)
    return model

def detect_objects(model, image_path):
    image = cv2.imread(image_path)
    results = model(image)
    return results, image

def get_stem_boxes(results):
    stem_boxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 1:
                x1, y1, x2, y2 = box.xyxy[0].cpu().int().numpy()
                stem_boxes.append((x1, y1, x2, y2))
    return stem_boxes

def yolo_segment(model, image, box):
    x1, y1, x2, y2 = box
    crop_img = image[y1:y2, x1:x2]
    results = model(crop_img)
    return results, crop_img

def process_mask(binary_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    bool_mask = largest_mask.astype(np.bool_)
    skeleton = skeletonize(bool_mask)
    return skeleton

def find_shortest_path_skeleton(skeleton):
    coords = np.column_stack(np.where(skeleton))
    highest_point = coords[np.argmin(coords[:, 0])]
    lowest_point = coords[np.argmax(coords[:, 0])]
    path, cost = route_through_array(1 - skeleton, start=tuple(highest_point), end=tuple(lowest_point),
                                     fully_connected=True)
    shortest_path_skeleton = np.zeros_like(skeleton)
    for coord in path:
        shortest_path_skeleton[coord] = 1
    return shortest_path_skeleton, highest_point, lowest_point

def mark_middle_height_point(orig_image, skeleton, highest_point, lowest_point, offset):
    middle_height = (highest_point[0] + lowest_point[0]) // 2
    path_coords = np.column_stack(np.where(skeleton > 0))
    closest_point = path_coords[np.argmin(np.abs(path_coords[:, 0] - middle_height))]
    closest_point = (closest_point[1] + offset[0], closest_point[0] + offset[1])
    cv2.circle(orig_image, closest_point, 15, (0, 0, 255), -1)
    return orig_image, closest_point

def process_images_in_folder(folder_path, detect_weights, segment_weights, output_folder):
    detect_model = load_yolov8_model(detect_weights)
    segment_model = load_yolov8_model(segment_weights)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            results, image = detect_objects(detect_model, image_path)
            stem_boxes = get_stem_boxes(results)

            if len(stem_boxes) == 0:
                print(f'No stems found in {filename}. Saving original image...')
                original_image_path = os.path.join(output_folder, filename)
                cv2.imwrite(original_image_path, image)
                continue

            for box in stem_boxes:
                segment_results, crop_img = yolo_segment(segment_model, image, box)
                if isinstance(segment_results, list) and len(segment_results) > 0:
                    segment_result = segment_results[0]
                    masks = segment_result.masks
                    if masks is not None and len(masks.data) > 0:
                        mask_array = masks.data[0].cpu().numpy()
                        mask_image = (mask_array * 255).astype(np.uint8)
                        mask_image_resized = cv2.resize(mask_image, (crop_img.shape[1], crop_img.shape[0]))
                        skeleton = process_mask(mask_image_resized)
                        shortest_path_skeleton, highest_point, lowest_point = find_shortest_path_skeleton(skeleton)
                        marked_image, middle_point = mark_middle_height_point(image, shortest_path_skeleton, highest_point, lowest_point, (box[0], box[1]))

                        marked_image_path = os.path.join(output_folder, filename)
                        cv2.imwrite(marked_image_path, marked_image)
                        print(f'Marked image saved as {marked_image_path}')
                    else:
                        print(f'No masks found in {filename}. Saving original image...')
                        original_image_path = os.path.join(output_folder, filename)
                        cv2.imwrite(original_image_path, image)
                else:
                    print(f'No valid segment results found for {filename}. Saving original image...')
                    original_image_path = os.path.join(output_folder, filename)
                    cv2.imwrite(original_image_path, image)
        else:
            continue

if __name__ == "__main__":
    input_folder = 'imagetest'
    detect_weights = 'weight/improved_8n_best.pt'
    segment_weights = r'F:\Desktop\other-method\ultralytics-seg\runs\segment\seg3-ghos-silmneck\weights\best.pt'
    output_folder = 'result2'
    process_images_in_folder(input_folder, detect_weights, segment_weights, output_folder)
