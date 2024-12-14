import warnings
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import  SummaryWriter
summaryWriter = SummaryWriter('./log')
from torch.utils.data import DataLoader
import logging


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.embedding = nn.Embedding(1000, 8)
        self.fc = nn.Linear(100, 2)

    def forward(self, x_wide, x_deep, x_dense):
        x_wide = self.embedding(x_wide.long())
        x_wide = x_wide.view(-1, 64)
        x_deep = self.embedding(x_deep.long())
        x_deep = x_deep.view(-1, 32)

        X = torch.cat([x_wide,x_deep,x_dense], dim=1)
        logits = self.fc(X)
        target_classes = torch.argmax(logits, dim=1, keepdim=True)
        probabilites = torch.nn.functional.softmax(logits, dim=1)

        return {"logits":logits,
                "target_classes": target_classes,
                "probabilites": probabilites}


def train(epoch, model, train_loader, optimizer, early_stop_step, log_interval):
    model.train()
    running_loss = 0.0
    for current_step, sample in enumerate(train_loader):
        if current_step==0:
            logging.info(sample)
        logging.info(f"current_step: {current_step}")
        if current_step >5:
            break
        wide = sample[:, 0:8].long()
        deep = sample[:, 8:12].long()
        dense = sample[:, 12:-1]
        loss_fct = CrossEntropyLoss()

        pred = model(wide, deep, dense)
        pred = pred['logits']

        label = torch.round(sample[:, -1]).long()
        loss = loss_fct(pred.view(-1, 2), label.view(-1))
        optimizer.zero_grad()
        running_loss += loss.item()
        if current_step % log_interval ==0:
            loss = loss.item()
            pred = pred.view(-1,2).argmax(dim=1, keepdims=True)
            correct = pred.eq(label.view(-1).data).sum()
            logging.info(f"loss: {loss:.4f}  acc_train: {correct/len(train_loader.dataset):.4f}")
            logging.info(f"current epoch : [{epoch}]  current_step: {current_step}")
            if current_step == early_stop_step:
                break


def val(model, eval_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for current_step, sample in enumerate(eval_loader):
            logging.info(f"current_step: {current_step}")
            if current_step >5:
                break

            loss_fct = CrossEntropyLoss()
            wide = sample[:,0:8]
            deep = sample[:, 8:12]
            dense = sample[:,12:-1]
            pred = model(wide, deep, dense)
            pred = pred['logits']
            label = torch.round(sample[:,-1]).long()

            test_loss += loss_fct(pred.view(-1, 2), label.view(-1))
            pred = pred.view(-1,2).argmax(dim=1, keepdims=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss = test_loss / len(eval_loader)
    logging.info(f"Val set: Average loss: {test_loss:.4f}")
    logging.info(f"Val set: correct: {correct}")
    logging.info(f"Val set: sample numbers: {len(eval_loader.dataset)}")
    logging.info(f"Val set: acc: {100*correct/len(eval_loader.dataset): .4f}")


if __name__ == '__main__':
    epoch = 1
    early_stop_step = 2000
    log_interval = 1
    lr = 1e-3
    torch.manual_seed(1)
    model = Net()
    X_dense = torch.rand(2000, 4)
    X_wide = torch.randint(0,9,(2000, 8))
    X_deep = torch.randint(0, 9, (2000, 4))
    y_label = torch.rand(2000, 1)

    training_data = torch.cat((X_wide, X_deep, X_dense, y_label), 1)
    test_data = torch.cat((X_wide, X_deep, X_dense, y_label), 1)
    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15])

    for epoch in range(1, epoch + 1):
        train(epoch, model, train_dataloader, optimizer, early_stop_step, log_interval)
        val(model, test_dataloader)
        scheduler.step()

    logging.info("save integrated_gradients")
    checkpoint_path = "torchapi_ig_wnd_20221118.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print("Trianing is finished!")




