import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


class TestModel(nn.Module):
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.lin(x))


class TestDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.data = [torch.ones(10) for _ in range(10000)]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def main():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    dataset = TestDataset()

    print(device)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    )
    train_loader = DataLoader(
        dataset, batch_size=10, sampler=sampler, num_workers=0, pin_memory=True
    )
    model = TestModel(10, 10).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(100)):
        print(device)
        sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            x_hat = model(batch)
            loss = criterion(batch, x_hat)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
