import os
import glob
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def verify_keypoint_annotations(image_dir, label_dir, num_samples=5):
    """Verify keypoint annotations by visualizing them on images"""

    # Get list of images
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # Process a subset of images
    samples_to_check = min(num_samples, len(image_files))

    print(f"Verifying keypoint annotations on {samples_to_check} sample images...")

    for img_file in image_files[:samples_to_check]:
        base_name = os.path.splitext(img_file)[0]
        label_file = os.path.join(label_dir, base_name + '.txt')

        # Skip if label file doesn't exist
        if not os.path.exists(label_file):
            print(f"Warning: No label file for {img_file}")
            continue

        # Load image
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Load annotations
        with open(label_file, 'r') as f:
            annotations = [line.strip() for line in f if line.strip()]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)

        component_types = ['voltage_source', 'current_source', 'controlled_voltage_source',
                           'controlled_current_source', 'resistor', 'inductor', 'capacitor',
                           'diode', 'ground']

        # Process each annotation
        for annotation in annotations:
            parts = annotation.split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            class_name = component_types[class_id] if class_id < len(component_types) else f"Class {class_id}"

            # Get bbox coordinates (normalized)
            x_center, y_center, width_norm, height_norm = map(float, parts[1:5])

            # Convert to pixel coordinates
            x_center_px = x_center * w
            y_center_px = y_center * h
            width_px = width_norm * w
            height_px = height_norm * h

            # Calculate box corners
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)

            # Draw bounding box
            rect = patches.Rectangle((x1, y1), width_px, height_px,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Add label
            ax.text(x1, y1 - 5, class_name, color='white', fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.7))

            # Check for keypoints (parts length > 5 indicates keypoints)
            if len(parts) > 5:
                # Collect keypoints
                keypoints = []
                for i in range(5, len(parts), 3):
                    if i + 2 < len(parts):
                        kp_x = float(parts[i]) * w  # Denormalize
                        kp_y = float(parts[i + 1]) * h  # Denormalize
                        visibility = int(float(parts[i + 2]))  # Visibility flag
                        keypoints.append((kp_x, kp_y, visibility))

                # Draw each keypoint
                for idx, (kp_x, kp_y, vis) in enumerate(keypoints):
                    if vis > 0:  # Only draw visible keypoints
                        # Draw point
                        ax.plot(kp_x, kp_y, 'go', markersize=8)

                        # Label the keypoint
                        if idx == 0:
                            point_label = "+" if class_id in [0, 2] else "Start"
                        else:
                            point_label = "-" if class_id in [0, 2] else "End"

                        ax.text(kp_x + 5, kp_y + 5, point_label, fontsize=9, color='white',
                                bbox=dict(facecolor='green', alpha=0.7))

                # Draw line connecting keypoints
                if len(keypoints) >= 2:
                    # Draw arrow from first to second keypoint
                    ax.arrow(keypoints[0][0], keypoints[0][1],
                             keypoints[1][0] - keypoints[0][0],
                             keypoints[1][1] - keypoints[0][1],
                             head_width=10, head_length=10, fc='g', ec='g')

        plt.title(f"Image: {img_file}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    print("Verification complete")


# def augment_dataset_with_rotations(source_img_dir, source_label_dir, output_dir):
#     """
#     Augment dataset with 90°, 180°, and 270° rotations.
#     Also handles keypoint rotation for directional components.
#
#     Args:
#         source_img_dir: Directory containing original images
#         source_label_dir: Directory containing original labels
#         output_dir: Directory to save augmented dataset
#     """
#     # Create output directories
#     output_img_dir = os.path.join(output_dir, 'images')
#     output_label_dir = os.path.join(output_dir, 'labels')
#     os.makedirs(output_img_dir, exist_ok=True)
#     os.makedirs(output_label_dir, exist_ok=True)
#
#     # Print debugging information
#     print(f"Source image directory: {source_img_dir}")
#     print(f"Directory exists: {os.path.exists(source_img_dir)}")
#     print(f"Source label directory: {source_label_dir}")
#     print(f"Directory exists: {os.path.exists(source_label_dir)}")
#
#     # Get all image files (support multiple extensions and casing)
#     image_paths = []
#     for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.webp', '.WEBP']:
#         image_paths.extend(glob.glob(os.path.join(source_img_dir, f'*{ext}')))
#
#     print(f"Found {len(image_paths)} images for augmentation")
#     if len(image_paths) == 0:
#         print("ERROR: No images found! Check the directory path.")
#         return
#
#     for img_path in image_paths:
#         # Get file information
#         img_filename = os.path.basename(img_path)
#         base_name = os.path.splitext(img_filename)[0]
#         img_ext = os.path.splitext(img_filename)[1]
#
#         # Load image
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"Warning: Could not read image {img_path}")
#             continue
#
#         h, w = img.shape[:2]
#
#         # Find corresponding label file
#         label_path = os.path.join(source_label_dir, base_name + '.txt')
#         if not os.path.exists(label_path):
#             print(f"Warning: No label file for {img_filename}")
#             continue
#
#         # Read labels
#         with open(label_path, 'r') as f:
#             lines = f.read().splitlines()
#
#         # Copy original files
#         shutil.copy(img_path, os.path.join(output_img_dir, img_filename))
#         shutil.copy(label_path, os.path.join(output_label_dir, base_name + '.txt'))
#
#         # Process rotations (90°, 180°, 270°)
#         for angle in [90, 180, 270]:
#             # Rotate image
#             if angle == 90:
#                 rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#             elif angle == 180:
#                 rotated_img = cv2.rotate(img, cv2.ROTATE_180)
#             else:  # 270°
#                 rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#             # Save rotated image
#             rotated_img_path = os.path.join(output_img_dir, f"{base_name}_rot{angle}{img_ext}")
#             cv2.imwrite(rotated_img_path, rotated_img)
#
#             # Transform and save labels
#             rotated_labels = []
#             for line in lines:
#                 parts = line.strip().split()
#                 if not parts:
#                     continue
#
#                 class_id = parts[0]
#                 if len(parts) >= 5:
#                     x_center, y_center, width, height = map(float, parts[1:5])
#
#                     if angle == 90:
#                         # For 90° rotation (x,y) -> (1-y, x)
#                         new_x = 1.0 - y_center
#                         new_y = x_center
#                         new_width = height
#                         new_height = width
#                     elif angle == 180:
#                         # For 180° rotation (x,y) -> (1-x, 1-y)
#                         new_x = 1.0 - x_center
#                         new_y = 1.0 - y_center
#                         new_width = width
#                         new_height = height
#                     else:  # 270°
#                         # For 270° rotation (x,y) -> (y, 1-x)
#                         new_x = y_center
#                         new_y = 1.0 - x_center
#                         new_width = height
#                         new_height = width
#
#                     # Ensure values are within bounds
#                     new_x = max(0, min(1, new_x))
#                     new_y = max(0, min(1, new_y))
#                     new_width = max(0, min(1, new_width))
#                     new_height = max(0, min(1, new_height))
#
#                     # Format new label with correct position data
#                     new_label = f"{class_id} {new_x:.6f} {new_y:.6f} {new_width:.6f} {new_height:.6f}"
#
#                     # Handle keypoints if present
#                     if len(parts) > 5:
#                         keypoints = []
#                         for i in range(5, len(parts), 3):
#                             if i + 2 < len(parts):
#                                 kp_x, kp_y, vis = float(parts[i]), float(parts[i + 1]), parts[i + 2]
#
#                                 # Apply same rotation to keypoints
#                                 if angle == 90:
#                                     new_kp_x = 1.0 - kp_y
#                                     new_kp_y = kp_x
#                                 elif angle == 180:
#                                     new_kp_x = 1.0 - kp_x
#                                     new_kp_y = 1.0 - kp_y
#                                 else:  # 270°
#                                     new_kp_x = kp_y
#                                     new_kp_y = 1.0 - kp_x
#
#                                 # Ensure points are within bounds
#                                 new_kp_x = max(0, min(1, new_kp_x))
#                                 new_kp_y = max(0, min(1, new_kp_y))
#
#                                 # Add rotated keypoints
#                                 keypoints.extend([new_kp_x, new_kp_y, vis])
#
#                         # Add keypoints to new label
#                         for kp in keypoints:
#                             if isinstance(kp, float):
#                                 new_label += f" {kp:.6f}"
#                             else:
#                                 new_label += f" {kp}"
#
#                     rotated_labels.append(new_label)
#
#             # Save rotated labels
#             rotated_label_path = os.path.join(output_label_dir, f"{base_name}_rot{angle}.txt")
#             with open(rotated_label_path, 'w') as f:
#                 f.write('\n'.join(rotated_labels))
#
#     print(
#         f"Augmentation complete. Original dataset size: {len(image_paths)}, Total augmented size: {len(image_paths) * 4}")
def augment_dataset_with_rotations(source_img_dir, source_label_dir, output_dir):
    """
    Augment dataset with 90°, 180°, and 270° rotations.
    Also handles keypoint rotation for directional components.
    Includes improved error handling for malformed label files.
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

        # Read labels - check for formatting issues
        try:
            with open(label_path, 'r') as f:
                lines = f.read().splitlines()

            # Validate each line in the label file
            fixed_lines = []
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"Warning: Line {i + 1} in {label_path} has too few elements. Skipping.")
                    continue

                # Try parsing the bbox coordinates to validate format
                try:
                    class_id = parts[0]
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # Validate that values are in range [0,1]
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                            0 <= width <= 1 and 0 <= height <= 1):
                        print(f"Warning: Line {i + 1} in {label_path} has out-of-range coordinates. Normalizing.")
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                    # Reconstruct the line with validated values
                    fixed_line = f"{class_id} {x_center} {y_center} {width} {height}"

                    # Add keypoints if they exist
                    if len(parts) > 5:
                        keypoints = []
                        for j in range(5, len(parts), 3):
                            if j + 2 < len(parts):
                                try:
                                    kp_x = float(parts[j])
                                    kp_y = float(parts[j + 1])
                                    vis = parts[j + 2]

                                    # Validate keypoint coordinates
                                    kp_x = max(0, min(1, kp_x))
                                    kp_y = max(0, min(1, kp_y))

                                    keypoints.extend([kp_x, kp_y, vis])
                                except ValueError:
                                    print(
                                        f"Warning: Keypoint value error in line {i + 1}, value: {parts[j:j + 3]} in {label_path}")
                                    continue

                        # Add validated keypoints
                        for kp in keypoints:
                            if isinstance(kp, float):
                                fixed_line += f" {kp:.6f}"
                            else:
                                fixed_line += f" {kp}"

                    fixed_lines.append(fixed_line)
                except ValueError as e:
                    print(f"Error in {label_path}, line {i + 1}: {e}")
                    print(f"Problematic line content: '{line}'")
                    print(f"After splitting into parts: {parts}")
                    print("Skipping this line")
                    continue

            # Replace original lines with fixed ones
            lines = fixed_lines

            # If no valid lines, skip this image
            if not lines:
                print(f"Warning: No valid annotations in {label_path}. Skipping image.")
                continue

        except Exception as e:
            print(f"Error processing label file {label_path}: {e}")
            continue

        # Copy original files
        shutil.copy(img_path, os.path.join(output_img_dir, img_filename))
        with open(os.path.join(output_label_dir, base_name + '.txt'), 'w') as f:
            f.write('\n'.join(lines))

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
                    try:
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

                        # Format new label with correct position data
                        new_label = f"{class_id} {new_x:.6f} {new_y:.6f} {new_width:.6f} {new_height:.6f}"

                        # Handle keypoints if present
                        if len(parts) > 5:
                            keypoints = []
                            for i in range(5, len(parts), 3):
                                if i + 2 < len(parts):
                                    try:
                                        kp_x = float(parts[i])
                                        kp_y = float(parts[i + 1])
                                        vis = parts[i + 2]

                                        # Apply same rotation to keypoints
                                        if angle == 90:
                                            new_kp_x = 1.0 - kp_y
                                            new_kp_y = kp_x
                                        elif angle == 180:
                                            new_kp_x = 1.0 - kp_x
                                            new_kp_y = 1.0 - kp_y
                                        else:  # 270°
                                            new_kp_x = kp_y
                                            new_kp_y = 1.0 - kp_x

                                        # Ensure points are within bounds
                                        new_kp_x = max(0, min(1, new_kp_x))
                                        new_kp_y = max(0, min(1, new_kp_y))

                                        # Add rotated keypoints
                                        keypoints.extend([new_kp_x, new_kp_y, vis])
                                    except ValueError:
                                        # Skip this keypoint if there's a conversion error
                                        print(
                                            f"Warning: Error converting keypoint values in rotation. Skipping keypoint.")
                                        continue

                            # Add keypoints to new label
                            for kp in keypoints:
                                if isinstance(kp, float):
                                    new_label += f" {kp:.6f}"
                                else:
                                    new_label += f" {kp}"

                        rotated_labels.append(new_label)
                    except ValueError as e:
                        print(f"Error processing line during rotation: {e}")
                        print(f"Problematic line: {line}")
                        continue

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


# Move the execution code into a main function
def main():
    # Define paths
    base_dir = r"E:\ppt\图论\大作业"
    training_dir = os.path.join(base_dir, "训练集")
    original_data_dir = training_dir
    keypoint_label_dir = os.path.join(training_dir, "Annotations_keypoint")  # Your labels with keypoints
    augmented_data_dir = os.path.join(base_dir, "增强数据1")
    final_dataset_dir = os.path.join(base_dir, "最终数据集1")

    # Step 1: Verify some of the keypoint annotations (optional but recommended)
    print("Step 1: Verifying keypoint annotations (displaying first 3 images)...")
    verify_keypoint_annotations(
        image_dir=os.path.join(original_data_dir, 'JPEGImages'),
        label_dir=keypoint_label_dir,  # Use your already-enhanced labels
        num_samples=3  # Display first 3 images for verification
    )

    # Step 2: Run the data augmentation with the keypoint annotations
    print("Step 2: Augmenting dataset with rotations...")
    augment_dataset_with_rotations(
        source_img_dir=os.path.join(original_data_dir, 'JPEGImages'),
        source_label_dir=keypoint_label_dir,  # Use your labels with keypoints
        output_dir=augmented_data_dir
    )

    # Step 3: Split the augmented dataset
    print("Step 3: Splitting dataset into train/val/test...")
    split_dataset(
        source_dir=augmented_data_dir,
        output_dir=final_dataset_dir,
        train_ratio=0.7,
        val_ratio=0.2
    )

    # Step 4: Create data.yaml file
    data_yaml = {
        'path': final_dataset_dir,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 9,
        'names': ['voltage_source', 'controlled_voltage_source', 'current_source',
                  'controlled_current_source', 'resistor', 'inductor', 'capacitor',
                  'diode', 'ground'],
        # Add keypoint configuration for YOLOv8-pose
        'kpt_shape': [2, 3]  # 2 keypoints per object, 3 values per keypoint (x,y,visibility)
    }

    # Save the YAML file
    yaml_path = os.path.join(final_dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as file:
        yaml.dump(data_yaml, file, sort_keys=False)

    print(f"Created data.yaml file at {yaml_path}")
    print("Ready to train YOLOv8-pose model.")

    # Step 5: Train YOLOv8-pose model
    print("Step 5: Training YOLOv8-pose model...")
    model = YOLO('yolov8n-pose.pt')  # Load a pretrained pose model

    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=8,  # Reduced batch size for pose model (more memory intensive)
        workers=4,
        patience=20,
        device='0',  # Use specific GPU. Use 'cpu' if no GPU available
        project=os.path.join(base_dir, 'runs'),
        name='circuit_components_detector_pose',
        verbose=True
    )

    # Step 6: Evaluate the model
    print("Step 6: Evaluating the trained model...")
    model.val()

    print("Training and evaluation complete!")


# This is the critical part for fixing the multiprocessing issue
if __name__ == '__main__':
    # Add freeze support for Windows
    multiprocessing.freeze_support()
    main()