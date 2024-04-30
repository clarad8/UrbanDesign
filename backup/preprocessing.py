from PIL import Image
import os
import random
import shutil

#https://www.kaggle.com/datasets/sniafas/vyronas-database
#https://www.kaggle.com/datasets/danushkumarv/indian-monuments-image-dataset
#https://www.kaggle.com/datasets/tumanovalexander/home-bro-images
# delete all folders except the one called "buildings"

# all files recursed

count = 0

# Function to recursively traverse directories and resize images
def resize_images_recursive(input_dir, output_dir, target_size=(224, 224), count=count):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all items in the input directory
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)

        # If item is a directory, recursively call the function
        if os.path.isdir(item_path):
            resize_images_recursive(item_path, output_dir, target_size)

        # If item is a file and ends with .jpg or .png, resize and save
        elif item.endswith(".jpg") or item.endswith(".png") or item.endswith(".JPEG"):
            try:
                # Open the image file
                image = Image.open(item_path)
                
                # Resize the image with anti-aliasing
                resized_image = image.resize(target_size, resample=Image.LANCZOS)
                
                # Save the resized image
                file_path = ''.join(['church__', str(count), '.jpg'])
                resized_image.save(os.path.join(output_dir, file_path))
                print(f"Resized {item_path} and saved to {output_dir}")
                count += 1
            
            except Exception as e:
                print(f"Error resizing {item_path}: {e}")

# put each dataset through the function, for indian monuments do test and train images separately
# in the end, i had 4 folders: resized apartments, resized buildings, resized test monuments, resized train monuments

# input_dir = "C:\\Users\\Emily Shao\\Downloads\\churches"
# output_dir = "C:\\Users\\Emily Shao\\Downloads\\Resized_Church_Test"

# resize_images_recursive(input_dir, output_dir)


def split_data(input_dir, train_dir, test_dir, train_percentage=0.8):

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    files = os.listdir(input_dir)

    num_files_train = int(len(files) * train_percentage)

    files_train = random.sample(files, num_files_train)

    for file in files_train:
        src = os.path.join(input_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.move(src, dst)

    for file in os.listdir(input_dir):
        src = os.path.join(input_dir, file)
        dst = os.path.join(test_dir, file)
        shutil.move(src, dst)

        
input_directory = "C:\\Users\\Emily Shao\\Downloads\\Church_80"
train_directory = "C:\\Users\\Emily Shao\\Downloads\\Church_Train"
test_directory = "C:\\Users\\Emily Shao\\Downloads\\Church_Test"
split_data(input_directory, train_directory, test_directory)