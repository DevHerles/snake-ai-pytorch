import os
from model import Linear_QNet

import torch
import torch.nn as nn
import torch.nn.functional as F

input_size = 11
hidden_size = 256
output_size = 3

model_folder_path = './model'


# class Linear_QNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         x = self.linear2(x)
#         return x

#     def save(self, file_name='model.pth'):
#         if not os.path.exists(model_folder_path):
#             os.makedirs(model_folder_path)

#         file_name = os.path.join(model_folder_path, file_name)
#         torch.save(self.state_dict(), file_name)
#         print(self.state_dict())
#         print(self.state_dict()['linear1.weight'].shape)
#         print('xxxx')
#         quit()

def load_model():
  try:
    model = Linear_QNet(input_size, hidden_size, output_size)
    file_name = os.path.join(model_folder_path, 'model.pth')
    model.load_state_dict(torch.load(file_name))
    model.train()
    return model
  except Exception as e:
    print(e)
