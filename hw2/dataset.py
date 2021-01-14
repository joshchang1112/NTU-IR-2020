import torch
import torch.nn as nn

from torch.utils.data import Dataset
from random import sample 
from sklearn.model_selection import train_test_split

class RankingDataset(Dataset):
    """
    Args:
        data (list): List of samples.
        n_negative (int): Number of false samples used as negative samples to
            train. Set to -1 to use all false options.
        n_positive (int): Number of true samples used as positive samples to
            train. Set to -1 to use all true options.
        shuffle (bool): Do not shuffle options when sampling.
            **SHOULD BE FALSE WHEN TESTING**
    """
    def __init__(self, data, user, groundtruth, negative_sample=0):
        #self.label = []
        self.origin_data = data
        self.groundtruth = groundtruth

        self.data = []
        self.total_item = []
        self.user = []
        self.negative_items = []
        self.answer = []
        self.n = []
        for i in range(3260):
            self.total_item.append(i)

        if negative_sample != 0:
            self.n_negative = []
        
        for i in range(len(data)):
            self.data.extend(self.origin_data[i])
            self.user.extend([i] * len(self.origin_data[i]))

            n_negative = negative_sample*len(self.origin_data[i])
            self.negative_items.append(list(set(self.total_item) - set(self.groundtruth[i])))
            self.n.extend(sample(self.negative_items[i], n_negative))
            #self.answer.extend([1] * len(self.origin_data[i]))
            for j in range(negative_sample - 1):
                self.data.extend(self.origin_data[i]) 
                self.user.extend([i] * len(self.origin_data[i])) 
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #if self.answer[idx] == 0:
        self.n[idx] = sample(self.negative_items[self.user[idx]], 1)[0]
        
        return {
                'data': self.data[idx],
                'user': self.user[idx],
                #'label': self.label[self.user[idx]],
                #'answer': self.answer[idx],
                'neg': self.n[idx]
            }
        '''
        return {
            'data': self.data[idx],
            'user': self.user[idx],
            #'label': self.label[self.user[idx]],
            #'answer': self.answer[idx]
            'neg': self.n[idx]
            }
        '''

    def collate_fn(self, samples):
        batch = {}
        #batch['data_lens']= [len(sample['data']) for sample in samples]
        #padded_len = max(batch['data_lens'])
        batch['data'] = torch.LongTensor([sample['data'] for sample in samples])
        batch['neg'] = torch.LongTensor([sample['neg'] for sample in samples])

        #batch['answer_lens']= [len(sample['answer']) for sample in samples]
        #padded_len = max(batch['answer_lens'])
        #batch['answer'] = torch.Tensor([sample['answer'] for sample in samples])

        batch['user'] = torch.LongTensor([sample['user'] for sample in samples])
        #batch['negative'] = torch.LongTensor([sample['negative'] for sample in samples])

        #batch['label'] = [sample['label'] for sample in samples]

        return batch

def pad_to_len(arr, padded_len):

    #arr = np.array(arr, dtype= np.float32)
    new_arr = arr.copy()
    length_arr = len(arr)
    if length_arr < padded_len: 
        for i in range(padded_len - len(arr)):
            new_arr.append(-1)

    return new_arr
