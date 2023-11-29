import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import albumentations as A
import tensorflow as tf
import segmentation_models as sm
from .data import Dataset, DataLoader
from .preprocessing import get_preprocessing, get_training_augmentation, get_validation_augmentation

DATA_DIR = './data/airbus-ships-data'

train_features_dir = os.path.join(DATA_DIR, 'train_images')
train_masks_dir = os.path.join(DATA_DIR, 'train_masks')

val_features_dir = os.path.join(DATA_DIR, "val_images")
val_masks_dir = os.path.join(DATA_DIR, "val_masks")

BACKBONE = "resnet34"
BATCH_SIZE = 16
CLASSES = 1
LR = 4e-5
EPOCHS = 10

preprocess_input = sm.get_preprocessing(BACKBONE)

n_classes = 1
activation = 'sigmoid'
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

optim = tf.keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss()
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile(optim, dice_loss, metrics)

train_dataset1 = Dataset(train_features_dir, train_masks_dir, classes=["ship"], preprocessing=get_preprocessing(preprocess_input))
train_dataset2 = Dataset(train_features_dir, train_masks_dir, classes=["ship"], augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocess_input))

val_dataset1 = Dataset(val_features_dir, val_masks_dir, classes=["ship"], preprocessing=get_preprocessing(preprocess_input))
val_dataset2 = Dataset(val_features_dir, val_masks_dir, classes=["ship"], augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocess_input))

train_dataloader1 = DataLoader(train_dataset1, batch_size=BATCH_SIZE, shuffle=False)
train_dataloader2 = DataLoader(train_dataset2, batch_size=BATCH_SIZE, shuffle=False)

valid_dataloader1 = DataLoader(val_dataset1, batch_size=BATCH_SIZE, shuffle=False)
valid_dataloader2 = DataLoader(val_dataset2, batch_size=BATCH_SIZE, shuffle=False)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./wieghts/best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3)
]

history1 = model.fit(
    train_dataloader1, 
    steps_per_epoch=len(train_dataloader1), 
    epochs=4, 
    callbacks=callbacks, 
    validation_data=valid_dataloader1, 
    validation_steps=len(valid_dataloader1),
)

history2 = model.fit(
    train_dataloader2, 
    steps_per_epoch=len(train_dataloader2), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader2, 
    validation_steps=len(valid_dataloader2),
)