import warnings
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients, configure_interpretable_embedding_layer
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


# def sequential_forward(X_wide, X_deep, X_dense):
#     return model.forward(X_wide, X_deep, X_dense)
#
#
# def sal(X_wide, X_deep, X_dense, node_index):
#     sal = IntegratedGradients(sequential_forward)
#     grads = sal.attribute((X_wide, X_deep, X_dense), target=node_index)
#     return grads


class Explain_IntegratedGradients(object):
    def __init__(self, node_index):
        super(Explain_IntegratedGradients, self).__init__()
        self.model = Net().to(gpu)
        self.node_index = node_index

    def sequential_forward(self,X_wide, X_deep, X_dense):
        return self.model.forward(X_wide, X_deep, X_dense)['probabilites']

    def sal(self, X_wide, X_deep, X_dense, node_index, n_steps):
        sal = IntegratedGradients(self.sequential_forward)
        grads = sal.attribute((X_wide, X_deep, X_dense), target=node_index, n_steps=n_steps, return_convergence_delta=True)
        return grads


if __name__ == '__main__':
    gpu = torch.device('cpu')
    epchos = 1
    early_stop_step = 2000
    log_interval = 1
    lr = 1e-3
    n_steps = 50
    torch.manual_seed(1)
    model = Net()
    exp_ig = Explain_IntegratedGradients(1)

    model_path = 'torchapi_ig_wnd_20221118.pt'
    wnd_model = torch.load(model_path)

    exp_ig.model.load_state_dict(wnd_model)
    exp_ig.model = exp_ig.model.to(gpu)

    X_dense = torch.rand(2000, 4)
    X_wide = torch.randint(0, 9, (2000, 8))
    X_deep = torch.randint(0, 9, (2000, 4))
    y_label = torch.rand(2000, 1)
    test_data = torch.cat((X_wide, X_deep, X_dense, y_label),1)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb_net = configure_interpretable_embedding_layer(model)

    for current_step, sample in enumerate(test_dataloader):
        X_wide = sample[:,0:8]
        X_deep = sample[:, 8:12]
        X_dense = sample[:,12:-1]
        attr = exp_ig.sal(X_wide, X_deep, X_dense, 0,50)


