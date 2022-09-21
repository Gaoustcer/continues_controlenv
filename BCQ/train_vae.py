from torch.utils.data import DataLoader
from data.offlinerldataset import OfflineRL
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch
from model.VAE_BCQ import VAE_actiongenerate
def _train():
    EPOCH = 4
    loader = DataLoader(OfflineRL(),batch_size=256)
    msefunc = nn.MSELoss()
    lossindex = 1
    writer = SummaryWriter('./log/vaeloss')
    vae = VAE_actiongenerate().cuda()
    optim = torch.optim.Adam(vae.parameters(),lr=0.001)
    from tqdm import tqdm
    for epoch in (range(EPOCH)):
        for obs,action in tqdm(loader):
            action = action.cuda()
            pred_action = vae(obs)
            loss = msefunc(pred_action,action)
            writer.add_scalar("vaeloss",loss,lossindex)
            lossindex += 1
            optim.zero_grad()
            loss.backward()
            optim.step()
    torch.save(vae,"./data/VAEnet")

if __name__ == "__main__":
    _train()
