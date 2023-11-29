import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def create_img_dir(df, original_dir, new_dir):
    """Copies given images into a new given directory"""
    errors = 0
    os.makedirs(new_dir, exist_ok=True)   
    for file_name in df['ImageId']:
        source_file = os.path.join(original_dir, file_name)
        destination_file = os.path.join(new_dir, file_name)
        # Check if the source file exists and then copy it
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
        else:
            errors += 1
            print(f"File not found: {source_file}")

    if errors == 0:
        print("~"*30)
        print("FILES COPIED SUCCESSFULLY")
        print("~"*30)
    else:
        print("~"*30)
        print(f"FILES COPIED WITH {errors} errors")
        print("~"*30)
       
        
def rle_to_mask(rle_string, shape):
    """Convert rle-encoded image into a binary mask"""
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    return img.reshape(shape).T


def combine_masks(rle_list, shape):
    """Combine multiple RLEs into one mask"""
    combined_mask = np.zeros(shape, dtype=np.uint8)
    for rle in rle_list:
        if isinstance(rle, str):  # Check if RLE is valid (not NaN or similar)
            mask = rle_to_mask(rle, shape)
            combined_mask = np.maximum(combined_mask, mask)
            
    return combined_mask


def create_masks(df, img_dir, masks_dir):
    """Create masks for given images and save them in given directory"""
    os.makedirs(masks_dir, exist_ok=True)
    df = df.set_index("ImageId")
    for file_name in os.listdir(img_dir):
        rle = df.loc[file_name, 'EncodedPixels']
        if isinstance(rle, pd.Series):
            rle = rle.tolist()
        else:
            rle = [rle]
                
        mask = combine_masks(rle, (768, 768))
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_image_path = os.path.join(masks_dir, f"{os.path.splitext(file_name)[0]}.png")
        mask_image.save(mask_image_path)