import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt
import imageio

ia.seed(1)

GREEN = [0, 255, 0]
ORANGE = [255, 140, 0]
RED = [255, 0, 0]

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
            color = GREEN
        elif bb.is_partly_within_image(image.shape):
            color = ORANGE
        else:
            color = RED
        image_border = bb.shift(left=border, top=border)\
                         .draw_on_image(image_border, size=2, color=color)
    return image_border

# Load the example image
image_path = r"20_times_dataset_picture\aEcL526-20x-3h-211217R1-7.jpg"
image = imageio.imread(image_path)

# Convert grayscale image to RGB
if len(image.shape) == 2:  # Grayscale image
    image = np.stack([image] * 3, axis=-1)

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

# Load the bounding boxes
yolo_file_path = r"obj_train_data\aEcL526-20x-3h-211217R1-7.txt"
bbs = read_bounding_boxes(yolo_file_path, image.shape)

# Define the zoom augmenter
zoom = iaa.Affine(scale=(2.0, 2.0))

# Apply the zoom to the image and bounding boxes
image_zoom, bbs_zoom = zoom(image=image, bounding_boxes=bbs)

# Clip bounding boxes that are partially outside the image
bbs_zoom_clipped = bbs_zoom.remove_out_of_image().clip_out_of_image()

# Draw the bounding boxes on the images
image_bbs = bbs.draw_on_image(image, size=2)
image_zoom_bbs_clipped = bbs_zoom_clipped.draw_on_image(image_zoom, size=2)

# Display the original and zoomed images with bounding boxes
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_bbs)
axes[0].set_title('Original Image with Bounding Boxes')
axes[1].imshow(image_zoom_bbs_clipped)
axes[1].set_title('Zoomed Image with Clipped Bounding Boxes')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

