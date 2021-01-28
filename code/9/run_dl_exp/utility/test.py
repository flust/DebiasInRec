import numpy as np
import matplotlib.pyplot as plt
import recommend

num_batch = 3
num_item = 5
k = 3
x = np.random.uniform(0, 1, num_batch*num_item)
y = np.empty(num_batch*k, dtype=np.int32)

print(x.reshape(num_batch, num_item))
recommend.get_top_k_by_greedy(x, num_batch, num_item, k, y)
print(y.reshape(num_batch, k))
