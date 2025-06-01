import os
import glob
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO
import multiprocessing

def augment_dataset_with_rotations(source_img_dir, source_label_dir, output_dir):
    """
    Augment dataset with 90°, 180°, and 270° rotations.

    Args:
        source_img_dir: Directory containing original images
        source_label_dir: Directory containing original labels
        output_dir: Directory to save augmented dataset
    """
    # Create output directories
    output_img_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Print debugging information
    print(f"Source image directory: {source_img_dir}")
    print(f"Directory exists: {os.path.exists(source_img_dir)}")
    print(f"Source label directory: {source_label_dir}")
    print(f"Directory exists: {os.path.exists(source_label_dir)}")

    # Get all image files (support multiple extensions and casing)
    image_paths = []
    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.webp', '.WEBP']:
        image_paths.extend(glob.glob(os.path.join(source_img_dir, f'*{ext}')))

    print(f"Found {len(image_paths)} images for augmentation")
    if len(image_paths) == 0:
        print("ERROR: No images found! Check the directory path.")
        return

    for img_path in image_paths:
        # Get file information
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]
        img_ext = os.path.splitext(img_filename)[1]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        h, w = img.shape[:2]

        # Find corresponding label file
        label_path = os.path.join(source_label_dir, base_name + '.txt')
        if not os.path.exists(label_path):
            print(f"Warning: No label file for {img_filename}")
            continue

        # Read labels
        with open(label_path, 'r') as f:
            lines = f.read().splitlines()

        # Copy original files
        shutil.copy(img_path, os.path.join(output_img_dir, img_filename))
        shutil.copy(label_path, os.path.join(output_label_dir, base_name + '.txt'))

        # Process rotations (90°, 180°, 270°)
        for angle in [90, 180, 270]:
            # Rotate image
            if angle == 90:
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_img = cv2.rotate(img, cv2.ROTATE_180)
            else:  # 270°
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Save rotated image
            rotated_img_path = os.path.join(output_img_dir, f"{base_name}_rot{angle}{img_ext}")
            cv2.imwrite(rotated_img_path, rotated_img)

            # Transform and save labels
            rotated_labels = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = parts[0]
                if len(parts) >= 5:
                    x_center, y_center, width, height = map(float, parts[1:5])

                    if angle == 90:
                        # For 90° rotation (x,y) -> (1-y, x)
                        new_x = 1.0 - y_center
                        new_y = x_center
                        new_width = height
                        new_height = width
                    elif angle == 180:
                        # For 180° rotation (x,y) -> (1-x, 1-y)
                        new_x = 1.0 - x_center
                        new_y = 1.0 - y_center
                        new_width = width
                        new_height = height
                    else:  # 270°
                        # For 270° rotation (x,y) -> (y, 1-x)
                        new_x = y_center
                        new_y = 1.0 - x_center
                        new_width = height
                        new_height = width

                    # Ensure values are within bounds
                    new_x = max(0, min(1, new_x))
                    new_y = max(0, min(1, new_y))
                    new_width = max(0, min(1, new_width))
                    new_height = max(0, min(1, new_height))

                    rotated_labels.append(f"{class_id} {new_x:.6f} {new_y:.6f} {new_width:.6f} {new_height:.6f}")

            # Save rotated labels
            rotated_label_path = os.path.join(output_label_dir, f"{base_name}_rot{angle}.txt")
            with open(rotated_label_path, 'w') as f:
                f.write('\n'.join(rotated_labels))

    print(
        f"Augmentation complete. Original dataset size: {len(image_paths)}, Total augmented size: {len(image_paths) * 4}")


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, random_seed=42):
    """
    Split a dataset into training, validation, and test sets.

    Args:
        source_dir: Directory containing images and labels subdirectories
        output_dir: Directory to save the split dataset
        train_ratio: Ratio of data for training set
        val_ratio: Ratio of data for validation set (test = 1 - train - val)
        random_seed: Random seed for reproducibility
    """
    # Source directories
    source_img_dir = os.path.join(source_dir, 'images')
    source_label_dir = os.path.join(source_dir, 'labels')

    # Check if directories exist
    print(f"Source image directory for splitting: {source_img_dir}")
    print(f"Directory exists: {os.path.exists(source_img_dir)}")

    # Create output directories
    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, folder, 'labels'), exist_ok=True)

    # Get all image files
    image_files = []
    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.webp', '.WEBP']:
        image_files.extend([os.path.basename(f) for f in glob.glob(os.path.join(source_img_dir, f'*{ext}'))])

    print(f"Found {len(image_files)} images for splitting")
    if len(image_files) == 0:
        print("ERROR: No images found in the augmented directory!")
        return

    # Make sure we have corresponding label files
    valid_files = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + '.txt'
        if os.path.exists(os.path.join(source_label_dir, label_file)):
            valid_files.append(img_file)
        else:
            print(f"Warning: No label file for {img_file}")

    print(f"Found {len(valid_files)} valid image-label pairs for splitting")

    # Split dataset
    train_files, remain_files = train_test_split(valid_files, train_size=train_ratio, random_state=random_seed)

    # Calculate validation ratio from remaining files
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_files, test_files = train_test_split(remain_files, train_size=val_ratio_adjusted, random_state=random_seed)

    # Copy files to their respective directories
    for files, folder in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        for file in files:
            # Get base name without extension
            base_name = os.path.splitext(file)[0]

            # Copy image
            src_img = os.path.join(source_img_dir, file)
            dst_img = os.path.join(output_dir, folder, 'images', file)
            shutil.copy(src_img, dst_img)

            # Copy corresponding label file
            label_file = base_name + '.txt'
            src_label = os.path.join(source_label_dir, label_file)
            dst_label = os.path.join(output_dir, folder, 'labels', label_file)
            shutil.copy(src_label, dst_label)

    # Print summary
    print(f"Dataset split complete:")
    print(f"  Training set: {len(train_files)} images")
    print(f"  Validation set: {len(val_files)} images")
    print(f"  Test set: {len(test_files)} images")


# # Define paths - CORRECTED TO MATCH YOUR ACTUAL DIRECTORY STRUCTURE
# base_dir = r"E:\ppt\图论\大作业"
# training_dir = os.path.join(base_dir, "训练集")  # Your main training directory
# original_data_dir = training_dir  # This now points to where your JPEGImages and Annotations are
# augmented_data_dir = os.path.join(base_dir, "增强数据")
# final_dataset_dir = os.path.join(base_dir, "最终数据集")
#
# # 1. First run the data augmentation
# augment_dataset_with_rotations(
#     source_img_dir=os.path.join(original_data_dir, 'JPEGImages'),
#     source_label_dir=os.path.join(original_data_dir, 'Annotations'),
#     output_dir=augmented_data_dir
# )
#
# # 2. Then split the augmented dataset
# split_dataset(
#     source_dir=augmented_data_dir,
#     output_dir=final_dataset_dir,
#     train_ratio=0.7,
#     val_ratio=0.2
# )
#
# # 3. Create data.yaml file
# data_yaml = {
#     'path': final_dataset_dir,
#     'train': 'train/images',
#     'val': 'val/images',
#     'test': 'test/images',
#     'nc': 9,
#     'names': ['voltage_source', 'controlled_voltage_source', 'current_source',
#               'controlled_current_source', 'resistor', 'inductor', 'capacitor',
#               'diode', 'ground']
# }
#
# # Save the YAML file
# yaml_path = os.path.join(final_dataset_dir, 'data.yaml')
# with open(yaml_path, 'w') as file:
#     yaml.dump(data_yaml, file, sort_keys=False)
#
# print(f"Created data.yaml file at {yaml_path}")
# print("Ready to train YOLOv8 model.")
#
# # 4. Train YOLOv8 model
# model = YOLO('yolov8n.pt')  # Load a pretrained model (recommended for training)
#
# # Train the model
# results = model.train(
#     data=yaml_path,
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     workers=4,
#     patience=20,
#     device='0',  # Use specific GPU. Use 'cpu' if no GPU available
#     project=os.path.join(base_dir, 'runs'),
#     name='circuit_components_detector',
#     verbose=True
# )
#
# # 5. Evaluate the model
# model.val()

# Move the execution code into a main function
def main():
    # Define paths
    base_dir = r"E:\ppt\图论\大作业"
    training_dir = os.path.join(base_dir, "训练集")
    original_data_dir = training_dir
    augmented_data_dir = os.path.join(base_dir, "增强数据")
    final_dataset_dir = os.path.join(base_dir, "最终数据集")

    # 1. First run the data augmentation
    augment_dataset_with_rotations(
        source_img_dir=os.path.join(original_data_dir, 'JPEGImages'),
        source_label_dir=os.path.join(original_data_dir, 'Annotations'),
        output_dir=augmented_data_dir
    )

    # 2. Then split the augmented dataset
    split_dataset(
        source_dir=augmented_data_dir,
        output_dir=final_dataset_dir,
        train_ratio=0.7,
        val_ratio=0.2
    )

    # 3. Create data.yaml file
    data_yaml = {
        'path': final_dataset_dir,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 9,
        'names': ['voltage_source', 'controlled_voltage_source', 'current_source',
                  'controlled_current_source', 'resistor', 'inductor', 'capacitor',
                  'diode', 'ground']
    }

    # Save the YAML file
    yaml_path = os.path.join(final_dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as file:
        yaml.dump(data_yaml, file, sort_keys=False)

    print(f"Created data.yaml file at {yaml_path}")
    print("Ready to train YOLOv8 model.")

    # 4. Train YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load a pretrained model

    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,  # Try reducing workers if still having issues
        patience=20,
        device='0',  # Use specific GPU. Use 'cpu' if no GPU available
        project=os.path.join(base_dir, 'runs'),
        name='circuit_components_detector',
        verbose=True
    )

    # 5. Evaluate the model
    model.val()

# This is the critical part for fixing the multiprocessing issue
if __name__ == '__main__':
    # Add freeze support for Windows
    multiprocessing.freeze_support()
    main()