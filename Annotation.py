import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt
import imageio
import os

ia.seed(1)

# Pad image with a 1px white and (BY-1)px black border
def pad(image, by):
    image_border1 = ia.pad(image, top=1, right=1, bottom=1, left=1,
                           mode="constant", cval=255)
    image_border2 = ia.pad(image_border1, top=by-1, right=by-1,
                           bottom=by-1, left=by-1,
                           mode="constant", cval=0)
    return image_border2

# Draw BBs on an image and extend the image plane by BORDER pixels.
# Mark BBs inside the image plane with green color, those partially inside
# with orange, and those fully outside with red.
def draw_bbs(image, bbs, border):
    image_border = pad(image, border)
    for bb in bbs.bounding_boxes:
        if bb.is_fully_within_image(image.shape):
            color = [0, 255, 0]  # GREEN
        elif bb.is_partly_within_image(image.shape):
            color = [255, 140, 0]  # ORANGE
        else:
            color = [255, 0, 0]  # RED
        image_border = bb.shift(left=border, top=border)\
                         .draw_on_image(image_border, size=2, color=color)
    return image_border

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

# Function to convert bounding boxes to YOLO format
def bounding_boxes_to_yolo(bbs, image_shape):
    yolo_bbs = []
    for bb in bbs.bounding_boxes:
        class_id = bb.label
        x_center = (bb.x1 + bb.x2) / 2 / image_shape[1]
        y_center = (bb.y1 + bb.y2) / 2 / image_shape[0]
        width = (bb.x2 - bb.x1) / image_shape[1]
        height = (bb.y2 - bb.y1) / image_shape[0]
        yolo_bbs.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_bbs

# Define the zoom augmenter
zoom = iaa.Affine(scale=(2.0, 2.0))

# Directories
image_dir = "20_times_dataset_picture"
annotation_dir = "obj_train_data"
output_annotation_dir = "output_annotations"
output_image_dir = "output_image_annotations"
os.makedirs(output_annotation_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)

# Process each image and its corresponding annotation
for image_filename in os.listdir(image_dir):
    if image_filename.endswith(".jpg"):
        # Load the image
        image_path = os.path.join(image_dir, image_filename)
        image = imageio.imread(image_path)

        # Convert grayscale image to RGB
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=-1)

        # Load the bounding boxes
        annotation_filename = image_filename.replace(".jpg", ".txt")
        annotation_path = os.path.join(annotation_dir, annotation_filename)
        bbs = read_bounding_boxes(annotation_path, image.shape)

        # Apply the zoom to the image and bounding boxes
        image_zoom, bbs_zoom = zoom(image=image, bounding_boxes=bbs)

        # Clip bounding boxes that are partially outside the image
        bbs_zoom_clipped = bbs_zoom.remove_out_of_image().clip_out_of_image()

        # Convert clipped bounding boxes to YOLO format
        yolo_bbs_zoom_clipped = bounding_boxes_to_yolo(bbs_zoom_clipped, image_zoom.shape)

        # Save the new bounding boxes to a file
        output_annotation_path = os.path.join(output_annotation_dir, annotation_filename.replace(".txt", "-augmented.txt"))
        with open(output_annotation_path, 'w') as file:
            for bb in yolo_bbs_zoom_clipped:
                file.write(bb + '\n')

        # Save the zoomed image
        output_image_path = os.path.join(output_image_dir, image_filename.replace(".jpg", "-augmented.jpg"))
        imageio.imwrite(output_image_path, image_zoom)

        print(f"Processed {image_filename} and saved new annotations to {output_annotation_path}")
        print(f"Saved zoomed image to {output_image_path}")
