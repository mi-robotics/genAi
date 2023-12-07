import torch
import numpy as np

x = np.random.randn(5**4)

y = torch.from_numpy(x.reshape((5,5,5,5)))

print(y.size())