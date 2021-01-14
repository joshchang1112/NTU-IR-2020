import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RankingDataset
from model import BPR_MF, BCE_MF
import numpy as np
import random
from sklearn.model_selection import train_test_split

def set_seed(SEED):
    SEED = 0
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


data_path = 'data/train.csv'
data = pd.read_csv(data_path)

d = []
total_item = 0
user = []
for i in range(len(data)):
    user.append(i)
    d.append([])
    tmp_list = data['ItemId'][i].split()
    for j in range(len(tmp_list)):
        d[i].append(int(tmp_list[j]))
    
    max_item = max(d[i])+1
    if total_item < max_item:
        total_item = max_item

total_user = len(data)
print("Total User: {}".format(total_user))
print("Total Item: {}".format(total_item))

set_seed(208)
X_train = []
X_test = []
for i in range(total_user):
    x_train, x_test = train_test_split(d[i], test_size=0.11, random_state=208)
    X_train.append(x_train)
    X_test.append(x_test)

train_set = RankingDataset(X_train, total_user, d, 5)
valid_set = RankingDataset(X_test, total_user, d, 5)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=train_set.collate_fn)
val_loader = DataLoader(valid_set, batch_size=32, shuffle=False, collate_fn=valid_set.collate_fn)

model = BPR_MF(total_user, total_item).cuda()

optimizer = optim.SGD(model.parameters(), lr=2e-2, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)
EPOCH = 50
criterion = nn.BCELoss()

accumulation_steps = 1
best_MAP = 0
best_val_loss = 100

# Training

for epoch in range(EPOCH):
    print('EPOCH:{}'.format(epoch+1))
    total_loss = 0.0
    model.train()
    for param_group in optimizer.param_groups:
        print("lr: {}".format(param_group['lr']))
    
    for i, batch in enumerate(train_loader):
        data = batch['data'].cuda()
        neg = batch['neg'].cuda()
        user = batch['user'].cuda()
        loss = model(user, data, neg)
        
        loss.backward()
        
        if((i+1)%accumulation_steps)==0:
            # optimizer the net
            optimizer.step()        # update parameters of net
            #scheduler.step()
            optimizer.zero_grad()   # reset gradient
        total_loss += loss.item()
    
    print("Train Loss:{}".format(total_loss/len(train_loader)))
    model.eval()
    MAP = 0
    if epoch+1 == EPOCH:
        pred = []
    
    total_loss = 0.0
    for i, batch in enumerate(val_loader):
        #data_lens = batch['data_lens'].cuda()
        
        data = batch['data'].cuda()
        user = batch['user'].cuda()
        
        neg = batch['neg'].cuda()

        with torch.no_grad():
            loss = model(user, data, neg)#.squeeze(2).squeeze(1)
        
        total_loss += loss.item()

    print("Valid Loss:{}".format(total_loss/len(val_loader)))
    scheduler.step(total_loss/len(val_loader))
    
    if total_loss/len(val_loader) < best_val_loss:
        best_loss = total_loss/len(val_loader)
        torch.save(model.state_dict(), 'model.pkl')
