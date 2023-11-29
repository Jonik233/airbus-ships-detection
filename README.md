# Airbus Ship Detection Challenge Solution

## Introduction
This project presents my solution to the Airbus Ship Detection Challenge. The challenge involves detecting ships in satellite images, a task that requires careful data handling and sophisticated image segmentation techniques.

## Dataset Analysis
An initial analysis of the dataset revealed a significant imbalance: the number of images without ships far exceeded those with ships. To address this, I curated a balanced dataset by randomly selecting 30,000 images without ships and retaining all images with ships. This approach was crucial for preventing the model from becoming biased toward images without ships.

## Dataset Composition
The dataset was divided into three subsets:
- **Training Dataset:** 52,239 samples
- **Validation Dataset:** 5,805 samples
- **Test Dataset:** 14,512 samples

## Preprocessing
Each image in the dataset was processed to create corresponding masks, which are essential for the image segmentation task. These masks serve as the ground truth for training the segmentation model.

## Model Architecture
I employed the `segmentation_models` library and chose U-Net as the primary architecture for this task. U-Net is renowned for its effectiveness in image segmentation problems, particularly in biomedical image segmentation.

## Hyperparameter Tuning
The model's performance was fine-tuned by adjusting the learning rate and batch size. After experimentation, the optimal hyperparameters were identified as:
- **Learning Rate (LR):** 4e-5
- **Batch Size:** 16

## Training and Augmentation
The model was initially trained for 20 epochs. Towards the end of this training phase, signs of overfitting were observed. To mitigate this, I applied image augmentation techniques and retrained the model for an additional 20 epochs.

## Results
- **Best Validation Result:** F1-score: 0.8785
- **Evaluation Result:** Mean F1-score: 0.87307

The training and evaluation results indicate the model's capability to effectively identify ships in satellite imagery. The use of data balancing, careful hyperparameter tuning, and image augmentation were key to enhancing the model's performance.