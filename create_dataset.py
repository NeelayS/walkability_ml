import numpy as np
import pandas as pd
from PIL import Image
import os
import pickle

from fastseg import MobileV3Large

model = MobileV3Large(num_classes=19, use_aspp=True, num_filters=256)
model = model.from_pretrained(num_filters=256)

class_names = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

img_class_counts = {}
img_dir = "../data/utube"

for img_path in sorted(os.listdir(img_dir)):
    print(f"Processing image {img_path}")
    img = Image.open(os.path.join(img_dir, img_path))
    labels = model.predict_one(img)
    _, counts = np.unique(labels, return_counts=True)
    n_pixels = labels.size
    counts = counts / n_pixels
    img_class_counts[img_path] = counts

with open("img_class_counts.pkl", "wb") as f:
    pickle.dump(img_class_counts, f)

# with open('img_class_counts.pkl', 'wb') as f:
#     img_class_counts = pickle.load(f)
