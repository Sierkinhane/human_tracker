"""
  author: Sierkinhane
  since: 2019-2-22 11:12:23
  description:simple human tracker -- based on center loss
"""

lr = 0.001
MAX_EPOCH = 200
DISPLAY = 1
BATCH_SIZE = 128
SHUFFLE = True
NUM_WORKERS = 3
RESUME = ''

import torch
from dataset import *
from face_models import Resnet18FaceModel, Resnet50FaceModel
from trainer import Trainer

# thanks to
def model_info(model):  # Plots a line-by-line description of a PyTorch model

    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def train(model, dataloader, device):

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr, momentum=0.9)
    model_info(model)

    trainloader, validloader = dataloader
    trainer = Trainer(
        optimizer,
        model,
        trainloader,
        validloader,
        max_epoch=MAX_EPOCH,
        resume=RESUME,
        device=device,
    )
    trainer.train()


if __name__ == '__main__':
    
    dataset = Data(data_dir='D:/00-Data/MARS/Market-1501-v15.09.15/Market-1501-v15.09.15/train_for_center_loss_market_cuhk')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataloader, num_classes = dataset.getDataloader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=SHUFFLE)
    model = Resnet18FaceModel(num_classes).to(device)
    print("loaded {} classes".format(num_classes))
    train(model, dataloader, device)