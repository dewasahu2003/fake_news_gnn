import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import UPFD
import pandas as pd
from torch_geometric.nn import global_max_pool, GATConv
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

# step:0 data processing
train_data = UPFD(
    root="./sample-data", name="gossipcop", feature="content", split="train"
)
test_data = UPFD(
    root="./sample-data", name="gossipcop", feature="content", split="test"
)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


class GNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()

        # Graph Convolution
        self.conv1 = GATConv(input_size, hidden_size)
        self.conv2 = GATConv(hidden_size, hidden_size)
        self.conv3 = GATConv(hidden_size, hidden_size)

        # Readout
        self.l_news = nn.Linear(input_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x, edge_index, batch):
        # Graph Conv..
        keep_going = self.conv1(x, edge_index).relu()
        keep_going = self.conv2(keep_going, edge_index).relu()
        keep_going = self.conv3(keep_going, edge_index).relu()

        # pooling
        keep_going = global_max_pool(keep_going, batch)

        # readout

        keep_going = self.l1(keep_going).relu()
        # some changes
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat((root.new_zeros(1), root + 1), dim=0)

        # news
        news = x[root]
        news = self.l_news(news).relu()
        keep_going = self.l2(torch.cat((keep_going, news), dim=-1))
        return torch.sigmoid(keep_going)


model = GNN(train_data.num_features, 128, 1)
optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
loss_fun = nn.BCELoss()

epochs = 40

for epoch in range(epochs):

    total_loss = 0

    for i, data in enumerate(train_loader):

        output = model(data.x, data.edge_index, data.batch)
        loss = loss_fun(torch.reshape(output, (-1,)), data.y.float())

        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += float(loss) * data.num_graphs

        if epoch % 10 == 0:
            print(f"epoch:{epoch} || loss:{loss} ")

print(f"total-loss-mean:{total_loss/len(train_loader.dataset)}")


def matrix(preds, gts):
    preds = torch.round(torch.cat(preds))
    gts = torch.cat(gts)
    acc = accuracy_score(preds, gts)
    f1 = f1_score(preds, gts)
    return acc, f1


with torch.no_grad():
    total_loss = 0
    all_pred = []
    all_label = []
    for i, data in enumerate(test_loader):
        output = model(data.x, data.edge_index, data.batch)
        loss = loss_fun(torch.reshape(output, (-1,)), data.y.float())
        total_loss += float(loss) * data.num_graphs
        all_pred.append(torch.reshape(output, (-1,)))
        all_label.append(data.y.float())

    acc, f1 = matrix(all_pred, all_label)
    print(
        f"total_mean-loss:{total_loss/len(test_loader.dataset)} || acc:{acc} || f1:{f1}"
    )

with torch.no_grad():

    for i, data in enumerate(test_loader):

        output = model(data.x, data.edge_index, data.batch)
        df = pd.DataFrame()
        df["pred_logit"] = output.detach().numpy()[:, 0]
        df["pred"] = torch.round(output).detach().numpy()[:, 0]
        df["true"] = data.y.numpy()

print(df.head())
