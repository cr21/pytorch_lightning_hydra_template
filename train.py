
class Trainer:
    def __init__(self, min_epochs: int, max_epochs: int, accelerator: str='cpu', **kwargs   ):
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.kwargs = kwargs

    def train(self):
        print(f"Training with {self.min_epochs} to {self.max_epochs} epochs and {self.accelerator} accelerator and {self.kwargs}")

    def __repr__(self):
        return f"Trainer + str(self)"







