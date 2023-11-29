import os
import cv2
import numpy as np
import tensorflow as tf

class Dataset:
    """
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
        preprocessing (albumentations.Compose): data preprocessing 
        
    """
    CLASSES = ["ship"]
    
    def __init__(self,
                 images_dir, 
                 masks_dir, 
                 classes=None, 
                 preprocessing=None, 
                 augmentation=None):
        
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, img_id) for img_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id.split(".")[0] + ".png") for mask_id in self.ids]
        
        if classes is None:
            self.classes = self.CLASSES
        else:
            self.classes = [cls.lower() for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        
        # Convert mask to binary format (1 where the mask is 255, 0 elsewhere)
        mask = (mask / 255).astype(np.float32)
        
        #apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        
        #apply preprocessin
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
            
        return image, mask
    
    def __len__(self):
        return len(self.ids)


class DataLoader(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        
        self.on_epoch_end()
    
    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   