from pytorch_trainer import Trainer, ModelGetter
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

class TestModel(nn.Module):
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.lin(x))
    
        
class TestTrainer(Trainer):
    def model_forward(self, batch) -> Tuple[Tensor, int]:
        batch = batch.to(self.device)
        B, _ = batch.shape
        
        x_hat = self.model(batch)
        loss = self.loss_fn(x_hat, batch)
        
        return loss, B


class TestDataset(Dataset):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.data = [torch.ones(10) for _ in range(100)]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    
    
    
if __name__ == '__main__':
    model_getter = ModelGetter(
        {
            'test_model': TestModel
        }
    )
    loader = DataLoader(TestDataset(), batch_size=5, shuffle=True)
    trainer = TestTrainer('./tests/test_config.yaml', model_getter)
    
    trainer(loader, loader)
    