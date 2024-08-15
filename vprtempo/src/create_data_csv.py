import os
import csv

def create_csv_from_images(folder_path, csv_file_path):
    files = os.listdir(folder_path)
    png_files = sorted([f for f in files if f.endswith('.png')])

    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image_name', 'index'])
        
        for index, image_name in enumerate(png_files):
            writer.writerow([image_name, index])

# Name of the dataset to create .csv for
dataset_name = 'nordland-fall'

# Generate paths
folder_path = ''
csv_file_path = os.path.join('./VPRTempo/vprtempo/dataset', f'{dataset_name}.csv')

# Create .csv file
create_csv_from_images(folder_path, csv_file_path)