import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
# import tqdm.notebook as tqdm

from dataset import Dataset
from model import SpectrogramModel

# F.cross_entropy keeps giving negative numbers
# so this hack should fix that I guess.
class spicy_loss:
    def __init__(self, device) -> None:
        self.device = device
    def __call__(self, y_hat, y) -> float:
        return (y - y_hat).abs().sum()
        # offset = torch.ones(y_hat.shape) * 1.2
        # offset = offset.to(self.device)
        # y_hat, y = y_hat + offset, y + offset
        # return F.cross_entropy(y_hat, y)

class Trainer:
    dataset: Dataset
    model: SpectrogramModel
    def __init__(self,
        dataset=None,
        model=None,
        device=None,
        checkpoint=None,
        checkpoint_dir="checkpoints",
        model_checkpoint_pattern="model{}.pth",
        load_from_checkpoint=False,
    ) -> None:
        if device is None:
            device = torch.device("cpu")
            print(f"Using default device: {device}")

        if dataset is None:
            dataset = Dataset("tracks")

        if model is None:
            model = SpectrogramModel(len(dataset.label_encoder.classes_))
            print("Using default model")

        if load_from_checkpoint:
            if checkpoint is None:
                raise ValueError("Checkpoint can't be None when loading")
            print(f"Loading models from checkpoint {checkpoint}")
            Trainer.load_from(model, model_checkpoint_pattern.format(checkpoint), dir=checkpoint_dir)
            print("Models loaded")

        if checkpoint is None:
            checkpoint = 0
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.model_checkpoint_pattern = model_checkpoint_pattern
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
        self.loss_func = nn.BCELoss()

    def train_step(self, entry):
        y, spectros = entry
        spectros = spectros.to(self.device)
        y_pred = self.model(spectros)
        
        loss = self.loss_func(y_pred, y.float())
        
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        return loss


    def train(
        self,
        epochs=10,
        save_every_n=10,
        log_every_n=1,
        batch_size=5,
        run_name="runs/run1"
    ):
        print(f"Beginning training {epochs} epochs, logging every {log_every_n}, saving every {save_every_n}, batch size {batch_size}.")
        
        import math

        import torch.utils.tensorboard
        writer = torch.utils.tensorboard.SummaryWriter(run_name)

        cpu = torch.device("cpu")

        j = 0
        entries_in_epoch = 0
        for epoch in range(epochs):
            for i, entry in enumerate(self.dataset.batched(batch_size)):
                if epoch == 0: entries_in_epoch += 1
                j += 1
                self.checkpoint += 1
                loss = self.train_step(entry)
                loss = loss.detach().to(cpu).numpy()
                if (i+1) % log_every_n == 0:
                    Trainer.show_loss(
                        epoch,
                        f"{i+1}/{entries_in_epoch}",
                        loss,
                    )
                    writer.add_scalar('loss', loss, j)
                if (i+1) % save_every_n == 0:
                    self.save()
        self.checkpoint+=1
        self.save()
        writer.close()
    
    def save(self):
        try: # ignore pipe errors when ctrl+c used
            print(f"Saving checkpoint {self.checkpoint}")
        except BrokenPipeError:
            pass
        Trainer.save_as(self.model, self.model_checkpoint_pattern.format(self.checkpoint), dir=self.checkpoint_dir)
        try:
            print("Saved")
        except BrokenPipeError:
            pass
            

    @classmethod
    def show_loss(self, epoch, batch, loss):
        print(
            f"epoch={epoch}",
            f"batch={batch}",
            f"loss={loss:.5f}",
        )

    @classmethod
    def save_as(self, model, name, dir="checkpoints"):
        import pathlib
        import os.path
        save_dir = pathlib.Path(dir)
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / name
        if os.path.exists(save_path):
            raise Exception("Won't overwrite existing models")
        else:
            torch.save(model.state_dict(), save_path)

    @classmethod
    def load_from(self, model, name, dir="checkpoints"):
        import pathlib
        import os.path
        save_dir = pathlib.Path(dir)
        save_path = save_dir / name
        if not os.path.exists(save_path):
            raise Exception(f"Can't find checkpoint file {save_path}")
        else:
            model.load_state_dict(torch.load(save_path))