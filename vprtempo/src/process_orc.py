import os
from PIL import Image
from tqdm import tqdm
import image
from camera_model import CameraModel

# Set this to the directory containing the images
base_path = '/media/adam/vprdatasets/data/orc'
# Modify if evaluating different ORC datasets
datasets = [
            '2015-08-12-15-04-18',
            '2014-11-21-16-07-03',
            '2015-10-29-12-18-17'
        ]
# Set up output path names
processed_path = 'demosaiced'

for dataset in datasets:
    # Define the dataset path
    dataset_path = os.path.join(base_path, dataset)

    # Create the folders
    os.makedirs(os.path.join(dataset_path, processed_path), exist_ok=True)

    # file path for the robotcar-dataset-sdk models folder
    model_dir = '/home/adam/repo/robotcar-dataset-sdk/models/'
    # file path for the left stereo images
    images_path = os.path.join(dataset_path, 'stereo/left')
    # Create a camera model object
    model = CameraModel(model_dir,images_path)
    # Get sorted list of PNG images
    images = sorted([os.path.join(images_path, image) for image in os.listdir(images_path) if image.endswith('.png')])

    # Process each image with a progress bar
    for img in tqdm(images, desc="Processing images", unit="image"):
        # Load the image
        processed_img = image.load_image(img, model=model)
        # Create the output file path
        output_img = os.path.join(dataset_path, processed_path, os.path.basename(img))
        processed_img = Image.fromarray(processed_img, mode="RGB")
        # Save the processed image as PNG
        processed_img.save(output_img, "PNG")

    print("Images processed and saved successfully.")