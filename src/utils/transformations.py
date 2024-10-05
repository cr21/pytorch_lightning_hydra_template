from torchvision.transforms import transforms

def normalize_transforms():
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )