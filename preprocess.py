from torchvision import io
import os


img_dir = "../data/utube/cities"

for img_path in os.listdir(img_dir):

    print(f"Processing image {img_path}")

    img = io.read_image(os.path.join(img_dir, img_path))
    os.remove(os.path.join(img_dir, img_path))

    img = img[:3, :, :]

    io.write_png(img, os.path.join(img_dir, img_path.split(".")[0] + ".png"))
