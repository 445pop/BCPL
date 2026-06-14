import os
import cv2
import numpy as np

# ===================== 配置部分 =====================
folder_A = '/root/data1/data/nnl/NNLdata/nianqian/test_data/images'  # 原始图片所在文件夹，请修改为实际路径
folder_B = '/root/data1/data/nnl/NNLdata/nianqian/test_data/labels'  # 标注 .txt 文件夹，请修改为实际路径
output_root_folder = '/root/data1/data/nnl/NNLdata/nianqian/cluster/test'  # 最终输出根目录，会按类别存子文件夹

# 支持的图片格式
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

# ====================================================

def yolo_to_pixel_coords(img_width, img_height, class_id, x_center, y_center, box_width, box_height):
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    box_width_abs = box_width * img_width
    box_height_abs = box_height * img_height

    x_min = int(x_center_abs - box_width_abs / 2)
    y_min = int(y_center_abs - box_height_abs / 2)
    x_max = int(x_center_abs + box_width_abs / 2)
    y_max = int(y_center_abs + box_height_abs / 2)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width - 1, x_max)
    y_max = min(img_height - 1, y_max)

    return x_min, y_min, x_max, y_max

def resize_with_padding(img, target_size=(224, 224), pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded

def ensure_dir_exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def process_images():
    for image_filename in os.listdir(folder_A):
        if not image_filename.lower().endswith(image_extensions):
            continue

        image_path = os.path.join(folder_A, image_filename)
        base_name = os.path.splitext(image_filename)[0]  # 去掉后缀
        txt_filename = base_name + '.txt'
        txt_path = os.path.join(folder_B, txt_filename)

        if not os.path.exists(txt_path):
            print(f"未找到 {image_filename} 对应的标注文件 {txt_filename}，跳过")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片 {image_filename}，跳过")
            continue

        img_height, img_width = img.shape[:2]

        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print(f"{txt_filename} 为空，跳过")
            continue

        # 遍历当前图片的每一个标注目标（每行一个目标）
        for target_idx, line in enumerate(lines):
            line = line.strip()
            parts = line.split()
            if len(parts) < 5:
                print(f"{txt_filename} 第 {target_idx + 1} 行格式错误，应为：class_id x_center y_center width height，跳过")
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])
            except Exception as e:
                print(f"{txt_filename} 第 {target_idx + 1} 行解析出错：{e}，跳过")
                continue

            x_min, y_min, x_max, y_max = yolo_to_pixel_coords(
                img_width, img_height, class_id, x_center, y_center, box_width, box_height
            )

            cropped = img[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                print(f"{image_filename} 第 {target_idx + 1} 个目标裁剪区域为空，可能越界，跳过")
                continue

            # 缩放并填充到 224x224
            resized_padded = resize_with_padding(cropped, target_size=(224, 224))

            # 按类别创建文件夹
            class_folder = os.path.join(output_root_folder, str(class_id))
            ensure_dir_exists(class_folder)

            # 保存图片，文件名包含原图名 + 目标索引，如 resized_image001_0.jpg
            output_filename = f"resized_{base_name}_{target_idx}.jpg"
            output_path = os.path.join(class_folder, output_filename)

            cv2.imwrite(output_path, resized_padded)
            print(f"已保存类别 {class_id} 的目标 {target_idx}：{output_path}")

if __name__ == '__main__':
    process_images()