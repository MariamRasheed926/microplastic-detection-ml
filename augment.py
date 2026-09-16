import cv2
import os
import albumentations as A

input_folder = "original_images"
output_folder = "augmented_images"
os.makedirs(output_folder, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.GaussNoise(p=0.2)
])

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        if image is not None:
            augmented = transform(image=image)["image"]
            new_name = "aug_" + filename
            cv2.imwrite(os.path.join(output_folder, new_name), augmented)
