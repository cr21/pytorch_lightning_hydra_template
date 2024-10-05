import argparse
import os
import torch
from model_builder.food_classifier import FoodClassifier
from torchvision import datasets
from torchvision import transforms
from utils.transformations import normalize_transforms
from pathlib import Path
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from torchvision.io import read_image
from PIL import Image

def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    results_dir:str,
    class_names: List[str] = None,
    image_size: Tuple[int, int] = (224, 224),
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Makes a prediction on a target image with a trained model and plots the image.


        # Get a random list of image paths from test set
        import random
        num_images_to_plot = 3
        test_image_path_list = list(Path(test_path).glob("*/*.jpg")) # get list all image paths from test data
        test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                               k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

        # Make predictions on and plot the images
        for image_path in test_image_path_sample:
            pred_and_plot_image(model=model,
                                image_path=image_path,
                                class_names=class_names,
                                # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                                image_size=(224, 224))
            plt.show()
    """
    # 1. Load in image and convert the tensor values to float32
    #target_image = read_image(str(image_path)).type(torch.float32)
    img = Image.open(image_path).convert("RGB")
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    # target_image = img / 255.0
    # 3. Transform if necessary
    if transform:
        target_image = transform(img)
        #print(target_image)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

     # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    #print(image_path)
    if class_names:
        title = f"Actual_label : {image_path.parent.stem},  Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Actual_label : {image_path.parent.stem},   Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    result_path = f"{results_dir}/{image_path.stem}_{class_names[target_image_pred_label.cpu()]}.png"
    plt.savefig(result_path)

def infer(args):
    print(args)
    if not os.path.isfile(f'./model/best_model_food_classifier.ckpt'):
        print("Model does not exists at location")
    else:
        print("Checkpoint found loading from checkpoint")
        model  = FoodClassifier.load_from_checkpoint(checkpoint_path=
                                                     f'{args.save_dir}/model/best_model_food_classifier.ckpt')
        print("Model loaded from checkpoints")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        results_dir = Path(args.save_dir)/"results"
        infernce_image_path_list = list(Path(Path(args.save_dir)/"data"/"val").glob("*/*.jpg"))
        results_dir.mkdir(parents=True, exist_ok=True)
        test_image_path_samples = random.sample(population=infernce_image_path_list, 
                                                k=int(args.num_samples))
        
        test_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            # normalize_transforms()
        ])
        class_names=['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi']
        for image_path in test_image_path_samples:
            pred_and_plot_image(model=model,
                                image_path=image_path,
                                results_dir=results_dir,
                                class_names=class_names,
                                transform=test_transforms, # optionally pass in a specified transform from our pretrained model weights
                                image_size=(IMG_SIZE, IMG_SIZE))
            plt.show()

        print("finished")
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Perform inference on images")
    parser.add_argument(
        "--ckpt_path", type=str, default='./model/best_model_food_classifier.ckpt', required=False, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--save-dir", default=".", help="checkpoint will be saved in this directory"
    )

    parser.add_argument(
        "--data-dir", default=".", help="data-path for inference data"
    )
    parser.add_argument(
        "--num-samples", default="10", help="data-path for inference data"
    )
    IMG_SIZE=224
    args = parser.parse_args()
    print(args)
    infer(args)