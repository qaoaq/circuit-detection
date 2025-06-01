import os
import glob
import shutil


def add_keypoints_to_yolo_files(input_dir, output_dir):
    """Add directional keypoints to YOLO annotation files"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each .txt file
    for txt_path in glob.glob(os.path.join(input_dir, "*.txt")):
        filename = os.path.basename(txt_path)
        output_path = os.path.join(output_dir, filename)

        with open(txt_path, 'r') as f:
            annotations = [line.strip() for line in f if line.strip()]

        new_annotations = []

        for annotation in annotations:
            parts = annotation.split()
            if len(parts) < 5:  # Skip invalid annotations
                continue

            # Extract class ID and bbox info
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])

            # Check if class needs directional keypoints
            # Class IDs for voltage sources, current sources, diodes (adjust as needed)
            directional_classes = [0, 1, 2, 3, 7]  # 0-indexed

            if class_id in directional_classes:
                # Add directional keypoints based on component type
                if class_id in [0, 2]:  # Voltage sources - vertical direction
                    kp1_x = x_center
                    kp1_y = y_center - 0.3 * height  # Top (positive terminal)
                    kp2_x = x_center
                    kp2_y = y_center + 0.3 * height  # Bottom (negative terminal)

                    # Create new annotation with keypoints
                    # Format: class_id x y w h kp1_x kp1_y v1 kp2_x kp2_y v2
                    new_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {kp1_x:.6f} {kp1_y:.6f} 2 {kp2_x:.6f} {kp2_y:.6f} 2"

                elif class_id in [1, 3]:  # Current sources - horizontal direction
                    kp1_x = x_center - 0.3 * width  # Left
                    kp1_y = y_center
                    kp2_x = x_center + 0.3 * width  # Right
                    kp2_y = y_center

                    new_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {kp1_x:.6f} {kp1_y:.6f} 2 {kp2_x:.6f} {kp2_y:.6f} 2"

                elif class_id == 7:  # Diodes
                    kp1_x = x_center - 0.3 * width  # Anode (left)
                    kp1_y = y_center
                    kp2_x = x_center + 0.3 * width  # Cathode (right)
                    kp2_y = y_center

                    new_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {kp1_x:.6f} {kp1_y:.6f} 2 {kp2_x:.6f} {kp2_y:.6f} 2"

                new_annotations.append(new_annotation)
            else:
                # For non-directional components, keep as is
                new_annotations.append(annotation)

        # Write out updated annotations
        with open(output_path, 'w') as f:
            f.write('\n'.join(new_annotations))

        print(f"Processed: {filename}")

# Example usage
add_keypoints_to_yolo_files("E:\ppt\图论\大作业\训练集\Annotations", "E:\ppt\图论\大作业\训练集\Annotations_keypoint")