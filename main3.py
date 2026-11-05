from dataset.curator.preprocessing import resize_image
from pathlib import Path
import cv2

images_path = Path("Datasets/LISTV2/images/train/src")
destination_path = Path("Datasets/LISTV3/images/train/src")
if not destination_path.exists():
    destination_path.mkdir(parents=True, exist_ok=True)

for image_file in images_path.glob("*.png"):
    image = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
    resized_image = resize_image(image, target_size=(512, 512))
    destination_file = destination_path / image_file.name
    cv2.imwrite(str(destination_file), resized_image)