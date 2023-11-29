import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import segmentation_models as sm
from data import Dataset, DataLoader
from preprocessing import get_preprocessing, get_validation_augmentation, get_training_augmentation

DATA_DIR = './data/airbus-ships-data'
test_features_dir = os.path.join(DATA_DIR, 'test_images')
test_masks_dir = os.path.join(DATA_DIR, 'test_masks')

BACKBONE = "resnet34"
BATCH_SIZE = 16
LR = 4e-5
preprocess_input = sm.get_preprocessing(BACKBONE)

n_classes = 1
activation = 'sigmoid'
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

optim = tf.keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss()
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile(optim, dice_loss, metrics)

test_dataset = Dataset(test_features_dir, test_masks_dir, classes=["ship"], preprocessing=get_preprocessing(preprocess_input))
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.load_weights('./weights/best_model.h5')
scores = model.evaluate(test_dataloader)

print(f"Loss: {scores[0]:.5}")
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))