from lightning import LightningModule, Trainer
from lightning.pytorch import Callback


class PrintCallBack(Callback):
    def __init__(self):
        super().__init__()
        self.collections = []

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f"starting Training epoch:  {trainer.current_epoch}")
        print('+'*50)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f"Finished Training Epoch  !!! {trainer.current_epoch}")
        print("+"*50)
        elogs = trainer.logged_metrics # access it here
        self.collections.append({f"{trainer.current_epoch}":elogs,'stg':'training'})
    

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f"starting Validation epoch:  {trainer.current_epoch}")
        print('+'*50)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f"Finished Validation Epoch  !!! {trainer.current_epoch}")
        print("+"*50)
        elogs = trainer.logged_metrics # access it here
        self.collections.append({f"{trainer.current_epoch}":elogs,'stg':'validation'})
