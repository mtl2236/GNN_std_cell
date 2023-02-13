import os
import torch
from torch_geometric.data import InMemoryDataset, Data

from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv

from torch_geometric.nn import global_mean_pool

from Dataset_gen_PDP import MyOwnDataset
#from sklearn.metrics import mean_absolute_percentage_error
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')# 这句话放在开头 必不可少 #
dataset = MyOwnDataset("MYdata_PDP")
#print(b.data.num_features)
#print(b.data.num_nodes)
#print(b.data.num_edges)
#print(b.data.y)

torch.manual_seed(1)
dataset = dataset.shuffle()

train_dataset = dataset[:8568]
test_dataset = dataset[8569:]

#print(f'Number of training graphs: {len(train_dataset)}')
#print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# for step, data in enumerate(train_loader):
    # print(f'Step {step + 1}:')
    # print('=======')
    # print(f'Number of graphs in the current batch: {data.num_graphs}')
    # print(data)
    # print()


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(1)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, batch):
        # 1. 获得节点嵌入
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)   # [batch_size, hidden_channels]
        # 3. 分类器
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

model = GCN(hidden_channels=64)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#criterion = torch.nn.MSELoss()
criterion=torch.nn.MSELoss()

def train():
    model.to(device)
    model.train()
    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        out=torch.squeeze(out)
        loss = criterion(out,data.y)
        loss.backward()
        optimizer.step()

def test(loader):
    model.to(device)
    model.eval()
    #percent_all=0
    mean_all=0
    pred_err_all=0
    variance_all=0
    output=[]
    for data in loader:                            # 批遍历测试集数据集。
        #error_percent = 0
        data.to(device)
        out = model(data.x, data.edge_index, data.batch) # 一次前向传播
        out=torch.squeeze(out)
        output.append(out)
        #err_percent=0
        pred_err=0
        mean=0
        for i in range(len(out)):
            #err_percent+=(torch.abs(out[i]-data.y[i])/(data.y[i]))
        #percent_all+=err_percent
    #percent_all=percent_all/len(loader.dataset)
    #return percent_all
            new_yi=0.0021+0.067*data.y[i]
            mean+=new_yi
            pred_err+=torch.square(0.067*out[i]-0.067*data.y[i])
        mean_all+=mean
        pred_err_all+=pred_err
    mean_all=mean_all/len(loader.dataset)
    
    #print(mean_all)
    #print(pred_err_all)
    for data in loader:
        #print(len(data))
        variance=0
        for i in range(len(data)):
            variance+=torch.square(0.0021+0.067*data.y[i]-mean_all)
        variance_all+=variance
    #print(variance_all)    
    #R2 calculation
    r2=1-(pred_err_all/variance_all)
    return r2


for epoch in range(1, 10001):
    train()
    #print(len(train_loader))
    #print(len(test_loader))
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    #if(100==epoch):
    	#torch.save(model,'model_PDP.pt')
