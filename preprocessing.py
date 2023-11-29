import albumentations as A

def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.HorizontalFlip(p=0.5),
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    
    _transform = [
        A.Resize(height=224, width=224, always_apply=True, p=1),
        A.Lambda(image=preprocessing_fn)
    ]
    return A.Compose(_transform)