import numpy as np
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt
import imageio
import os

ia.seed(1)

# Function to read bounding boxes from YOLO format file
def read_bounding_boxes(yolo_file_path, image_shape):
    bbs = []
    with open(yolo_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * image_shape[1]
            y_center = float(parts[2]) * image_shape[0]
            width = float(parts[3]) * image_shape[1]
            height = float(parts[4]) * image_shape[0]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=class_id))
    return BoundingBoxesOnImage(bbs, shape=image_shape)

# Directories
output_annotation_dir = "output_annotations"
output_image_dir = "output_image_annotations"
result_preview_dir = "result_preview"
os.makedirs(result_preview_dir, exist_ok=True)

# Process each image and its corresponding annotation
for image_filename in os.listdir(output_image_dir):
    if image_filename.endswith("-augmented.jpg"):
        # Load the zoomed image
        image_path = os.path.join(output_image_dir, image_filename)
        image_zoom = imageio.imread(image_path)

        # Load the corresponding bounding boxes
        annotation_filename = image_filename.replace("-augmented.jpg", "-augmented.txt")
        annotation_path = os.path.join(output_annotation_dir, annotation_filename)
        bbs_zoom_clipped = read_bounding_boxes(annotation_path, image_zoom.shape)

        # Draw the bounding boxes on the zoomed image
        image_zoom_bbs = bbs_zoom_clipped.draw_on_image(image_zoom, size=2)

        # Save the result image
        result_image_path = os.path.join(result_preview_dir, image_filename)
        imageio.imwrite(result_image_path, image_zoom_bbs)

        print(f"Processed {image_filename} and saved result to {result_image_path}")
