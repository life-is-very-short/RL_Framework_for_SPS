# Example of target with class indices
import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss(reduction='none')
input = torch.randn(2, 3, 16, 100, requires_grad=True)
target = torch.empty(2, 3, 16, dtype=torch.long).random_(3)
output = loss(input, target)
print(output.unsqueeze(dim=-1))
