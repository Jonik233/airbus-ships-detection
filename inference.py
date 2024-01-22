import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from data import Dataset
import segmentation_models as sm
import matplotlib.pyplot as plt
from preprocessing import get_preprocessing
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def predict(image_path):

    n_classes = 1
    activation = 'sigmoid'
    
    BACKBONE = "resnet50"
    preprocess_input = sm.get_preprocessing(BACKBONE)
    preprocessing_fn = get_preprocessing(preprocess_input)
    
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    model.load_weights("weights\weights50.h5")
    
    if os.path.exists(image_path):
        bgr_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = preprocessing_fn(image=rgb_image)["image"]
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)[0]
        return prediction
    else:
        raise FileNotFoundError(f"Could not find {image_path}")


def evaluate(test_imgs_path, test_masks_path):
    
    if os.path.exists(test_imgs_path) and os.path.exists(test_masks_path):
        n_classes = 1
        activation = 'sigmoid'
        BACKBONE50 = 'resnet50'
        
        preprocessing_input = sm.get_preprocessing(BACKBONE50)
        model = sm.Unet(BACKBONE50, classes=n_classes, activation=activation)
            
        dataset = Dataset(test_imgs_path, test_masks_path, classes=["ships"], preprocessing=get_preprocessing(preprocessing_input))
       
        scores = []
        
        print(f"\n\nNumber of samples: {len(dataset)}\n\n")
        st = time.monotonic()
        for i, (image, mask) in enumerate(dataset):
            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image)[0]
            avg_prediction = np.squeeze(prediction, axis=-1)
            avg_prediction = tf.where(avg_prediction >= 0.5, 1.0, 0.0).numpy()
            intersection = (avg_prediction == mask).sum()
            union = 2 * avg_prediction.shape[0] * avg_prediction.shape[1]
            dice_score = (2 * intersection) / union
            scores.append(dice_score)
            print(f"Test sample #{i}: score: {dice_score:.2f}\n")
     
        et = time.monotonic()
        total_time = et - st
        dice_score = sum(scores) / len(scores)
        print("\n" + "="*30)
        print(f"Dice score: {dice_score:.2f}")
        print("="*30)
        print(f"Time passed: {total_time:.2f}s")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference function")
    parser.add_argument("img_path", type=str, help="Image file path")
    args = parser.parse_args()
    prediction = predict(args.img_path)
    bgr_image = cv2.imread(args.img_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(rgb_image)
    axs[0].set_title("Ground truth")
    axs[0].axis("off")
    axs[1].imshow(prediction)
    axs[1].set_title("Prediction")
    axs[1].axis("off")
    plt.show()