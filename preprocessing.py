from PIL import Image
import os

#https://www.kaggle.com/datasets/sniafas/vyronas-database
#https://www.kaggle.com/datasets/danushkumarv/indian-monuments-image-dataset
#https://www.kaggle.com/datasets/tumanovalexander/home-bro-images
# delete all folders except the one called "buildings"

# Function to recursively traverse directories and resize images
def resize_images_recursive(input_dir, output_dir, target_size=(224, 224)):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all items in the input directory
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)

        # If item is a directory, recursively call the function
        if os.path.isdir(item_path):
            resize_images_recursive(item_path, output_dir, target_size)
        # If item is a file and ends with .jpg or .png, resize and save
        elif item.endswith(".jpg") or item.endswith(".png"):
            try:
                # Open the image file
                image = Image.open(item_path)
                
                # Resize the image with anti-aliasing
                resized_image = image.resize(target_size, resample=Image.LANCZOS)
                
                # Save the resized image
                resized_image.save(os.path.join(output_dir, item))
                print(f"Resized {item_path} and saved to {output_dir}")
            except Exception as e:
                print(f"Error resizing {item_path}: {e}")

# put each dataset through the function, for indian monuments do test and train images separately
# in the end, i had 4 folders: resized apartments, resized buildings, resized test monuments, resized train monuments

input_dir = "/Users/liaseo/Desktop/neural networks/UrbanDesign/Indian-monuments/images/test"
output_dir = "/Users/liaseo/Desktop/neural networks/UrbanDesign/resized test monuments"
resize_images_recursive(input_dir, output_dir)
