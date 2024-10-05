from lightning import Trainer
from model_builder.food_classifier import FoodClassifier
from datamodules.food_data_module import FoodDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.transforms import transforms
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.custom_callbacks import PrintCallBack
import os
import json




def normalize_transforms():
    return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

def train():
    # 1. train_transform
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
    # 3. data module
    food_data_module = FoodDataModule(transforms=train_transforms,
                                      test_transforms=test_transforms,
                                      data_dir='./data',
                                      batch_size=16,
                                      num_workers=2)

    # 4. model building
    food_classifier = FoodClassifier(num_classes=NUM_CLASSES,
                                     lr=1e-3,
                                    #  model_name='mobilevitv2_100'
                                    model_name='efficientnet_b3'
                                       )
    # 5. callbacks
    model_checkpoint = ModelCheckpoint(dirpath='./model',
                                       monitor='val_loss',
                                       filename='best_model_food_classifier',
                                       #filename='efficientnet_b0_{epoch}-{val_accuracy:.2f}-{val_loss:.2f}',
                                       save_top_k=1)

    print_callbacks = PrintCallBack()

    # train

    trainer = Trainer(
                    # fast_dev_run=True,
                    #   limit_train_batches=10,
                    #   limit_val_batches=10,
                        accelerator='auto',
                        min_epochs=MIN_EPOCH,
                        max_epochs=MAX_EPOCH,
                        enable_progress_bar=True,
                        callbacks=[
                            print_callbacks,
                            model_checkpoint
                        ]
                      )
    
    trainer.fit(food_classifier, food_data_module)
    print(f"class_names {food_data_module.class_names}")
    print(f"print_callbacks {print_callbacks.collections}")
    # try:
    #     with open("./model/trainer_loss_acc_results.json",'w') as f:
    #         json.dump(print_callbacks.collections, f)
    # except Exception as exp:
    #     print("Exception while writing json results")
    #     print(exp)


if __name__ =='__main__':
    IMG_SIZE = 224
    NUM_CLASSES = 10
    MIN_EPOCH=1
    MAX_EPOCH=20
    train()
    
