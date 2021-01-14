import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import sys
from model import BPR_MF, BCE_MF

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

device = torch.device('cpu')

# Load Model
model_1 = BPR_MF(total_user, total_item)
model_2 = BPR_MF(total_user, total_item)
model_3 = BPR_MF(total_user, total_item)
model_4 = BPR_MF(total_user, total_item)
model_5 = BPR_MF(total_user, total_item)

model_1.load_state_dict(torch.load('model/model_1.pkl', map_location=device))
model_2.load_state_dict(torch.load('model/model_2.pkl', map_location=device))
model_3.load_state_dict(torch.load('model/model_3.pkl', map_location=device))
model_4.load_state_dict(torch.load('model/model_4.pkl', map_location=device))
model_5.load_state_dict(torch.load('model/model_5.pkl', map_location=device))

# Predicting & ensemble
pred = []
model_1.eval()
model_2.eval()
model_3.eval()
model_4.eval()
model_5.eval()

total_item = []
for i in range(3260):
    total_item.append(i)
    
for i in range(total_user):
    ap = 0
    correct = 0
    count = 0
    interact_items = d[i]
    item = torch.LongTensor(total_item)
    with torch.no_grad():
        output_1 = model_1(torch.LongTensor([i]), item.unsqueeze(0), val=True)
        output_2 = model_2(torch.LongTensor([i]), item.unsqueeze(0), val=True)
        output_3 = model_3(torch.LongTensor([i]), item.unsqueeze(0), val=True)
        output_4 = model_4(torch.LongTensor([i]), item.unsqueeze(0), val=True)
        output_5 = model_5(torch.LongTensor([i]), item.unsqueeze(0), val=True)
    
    output = output_1 + output_2 + output_3 + output_4 + output_5
    ranking, idx = torch.topk(output, 500)
    idx = idx.squeeze(1).squeeze(0).tolist()
    
    count = 0
    pred_idx = []
    for j in range(500):
        if count == 50:
            break
        if idx[j] not in interact_items:
            count += 1
            pred_idx.append(idx[j])
    pred.append(pred_idx)

# Write pred to output csv
with open(sys.argv[1], 'w') as f:
    f.write('UserId,ItemId\n')
    for i, user_pred in enumerate(pred):
        f.write('{},'.format(i))
        first = 0
        for item in user_pred:
            if first == 0:
                first = 1
            else:
                f.write(' ')
            f.write('{}'.format(item))
        f.write('\n')


