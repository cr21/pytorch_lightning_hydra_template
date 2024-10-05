import os
import argparse
from model_builder.food_classifier import FoodClassifier
import torch
from lightning import Trainer
from datamodules.food_data_module import FoodDataModule
from torchvision.transforms import transforms
from utils.transformations import normalize_transforms
import json
from pathlib import Path

def evaluate(ckpt_path:str):
    """
    Load Model from checkpoint path and then evaluate
    
    """
    try:
        model  = FoodClassifier.load_from_checkpoint(checkpoint_path=ckpt_path)
    except Exception as exp:
        print(f"Exception while loading model from checkpoint {exp}")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    trainer=Trainer(accelerator='auto',
                        min_epochs=1,
                        max_epochs=1,
                        enable_progress_bar=True)
    
    train_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transforms()
        ]
    )
    
    # 2. test transform
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize_transforms()
    ])

    food_data_module = FoodDataModule(transforms=train_transforms,
                                      test_transforms=test_transforms,
                                      data_dir='./data',
                                      batch_size=32,
                                      num_workers=2)
    
    validation_output = trainer.validate(model=model,ckpt_path=ckpt_path,
                     datamodule=food_data_module)
    
    print(f"validation_output => {validation_output}") 

    try:
        with (Path(args.save_dir) / "model" / "eval_results.json").open("w") as f:
            json.dump(validation_output, f)
    except Exception as exp:
        print("Exception while writing json results")
        print(exp)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Perform inference on images")
    parser.add_argument(
        "--ckpt_path", type=str, default='./model/best_model_food_classifier.ckpt', required=False, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--save-dir", default="./", help="checkpoint will be saved in this directory"
    )
    IMG_SIZE=224
    args = parser.parse_args()
    print(args)
    evaluate(args.ckpt_path)