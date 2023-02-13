import os
import torch
from torch_geometric.data import InMemoryDataset, Data

from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool

from Dataset_gen import MyOwnDataset

dataset = MyOwnDataset("MYdata")
#print(b.data.num_features)
#print(b.data.num_nodes)
#print(b.data.num_edges)
#print(b.data.y)

torch.manual_seed(1)
dataset = dataset.shuffle()

train_dataset = dataset[:10000]
test_dataset = dataset[10000:]

#print(f'Number of training graphs: {len(train_dataset)}')
#print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# for step, data in enumerate(train_loader):
    # print(f'Step {step + 1}:')
    # print('=======')
    # print(f'Number of graphs in the current batch: {data.num_graphs}')
    # print(data)
    # print()


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(1)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        
    def forward(self, x, edge_index, batch):
        # 1. 获得节点嵌入
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        # 2. Readout layer
        x = global_mean_pool(x, batch)   # [batch_size, hidden_channels]
        
        # 3. 分类器
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GNN(hidden_channels=64)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    
    for data in train_loader:
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    
    correct = 0
    for data in loader:                            # 批遍历测试集数据集。
        out = model(data.x, data.edge_index, data.batch) # 一次前向传播
        pred = out.argmax(dim=1)                         # 使用概率最高的类别
        correct += int((pred == data.y).sum())           # 检查真实标签
    return correct / len(loader.dataset)


for epoch in range(1, 101):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    #if(100==epoch):
        #torch.save(model,'model.pt')
