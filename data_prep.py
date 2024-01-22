import os
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def create_img_dir(df:pd.DataFrame, original_dir:str, new_dir:str):
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


def rle_to_mask(rle_string:str, shape:tuple):
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to the original image


def combine_masks(rle_list:list, shape:tuple):
    # Combine multiple RLEs into one mask
    combined_mask = np.zeros(shape, dtype=np.uint8)
    for rle in rle_list:
        if isinstance(rle, str):  # Check if RLE is valid (not NaN or similar)
            mask = rle_to_mask(rle, shape)
            combined_mask = np.maximum(combined_mask, mask)
    return combined_mask


def create_masks(df:pd.DataFrame, img_dir:str, masks_dir:str):
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


def split_data(img_dir:str, rle_dir:str, blank_percentage:float):
    NEW_DATA_DIR = "./airbus_ships_data"
    os.makedirs(NEW_DATA_DIR, exist_ok=True)
    
    df = pd.read_csv(rle_dir)
    df_unique = df.drop_duplicates(subset='ImageId', keep='first')
     
    #splitting dataset into p% of blank images and (100 - p)% of representable images
    imgs_with_ships = df_unique.dropna(subset=['EncodedPixels'])
    N_BLANK_SAMPLES = (imgs_with_ships.shape[0] * blank_percentage) / (1 - blank_percentage)
    N_BLANK_SAMPLES = int(N_BLANK_SAMPLES)
    
    imgs_without_ships = df_unique[df_unique['EncodedPixels'].isna()]
    imgs_without_ships = imgs_without_ships.sample(n=N_BLANK_SAMPLES, random_state=1)
    
    #creating final dataframe and shuffling it
    df_final = pd.concat([imgs_without_ships, imgs_with_ships])
    df_final = df_final.sample(frac=1).reset_index(drop=True)

    print("="*30)
    print("Saving rle encodings...")
    df_final.to_csv(os.path.join(NEW_DATA_DIR, 'rle_encodings.csv'), index=False)
    
    #splitting data into datasets
    print("Splitting data...")
    df_train, df_test = train_test_split(df_final, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42)
    
    dataset_nan_percentage = lambda dataset: (dataset['EncodedPixels'].isna().sum() / dataset.shape[0]) * 100
    initial_nan_percentage = dataset_nan_percentage(df_final)
    train_nan_percentage = dataset_nan_percentage(df_train)
    val_nan_percentage = dataset_nan_percentage(df_val)
    test_nan_percentage = dataset_nan_percentage(df_test)
    
    print("\nShapes of splitted data: ")
    print("--"*30)
    print(f"Initial dataset: shape: {df_final.shape}, nan: {initial_nan_percentage}%")
    print(f"Train dataset: shape: {df_train.shape}, nan: {train_nan_percentage}%")
    print(f"Validation dataset: shape: {df_val.shape}, nan: {val_nan_percentage}")
    print(f"Test dataset: shape: {df_test.shape}, nan: {test_nan_percentage}")
    print("--"*30)
    
    new_train_dir = os.path.join(NEW_DATA_DIR, 'train_images')
    new_val_dir = os.path.join(NEW_DATA_DIR, 'val_images')
    new_test_dir = os.path.join(NEW_DATA_DIR, 'test_images')

    print("\nCreating directories with splitted data...")
    create_img_dir(df_train, img_dir, new_train_dir)
    create_img_dir(df_val, img_dir, new_val_dir)
    create_img_dir(df_test, img_dir, new_test_dir)
    
    print("\Creating masks...")
    train_masks_folder = os.path.join(NEW_DATA_DIR, "train_masks")
    val_masks_folder = os.path.join(NEW_DATA_DIR, "val_masks")
    test_masks_folder = os.path.join(NEW_DATA_DIR, "test_masks")
    create_masks(df, os.path.join(NEW_DATA_DIR, "train_images"), train_masks_folder)
    create_masks(df, os.path.join(NEW_DATA_DIR, "val_images"), val_masks_folder)
    create_masks(df, os.path.join(NEW_DATA_DIR, "test_images"), test_masks_folder)
    
    print("--"*20)
    print("DATA SPLIT DONE")
    print("--"*20)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splits images into train, val and test datasets, datasets are represented in the form of directories")
    parser.add_argument("img_dir", type=str, help="Directory for the images.")
    parser.add_argument("rle_dir", type=str, help="Directory for the RLE data.")
    parser.add_argument("blank_percentage", type=float, help="Percentage of blank images.")
    
    args = parser.parse_args()
    split_data(args.img_dir, args.rle_dir, args.blank_percentage)