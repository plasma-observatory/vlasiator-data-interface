import numpy as np


data = np.load("boxed_vdf_data_1610_BBF.npz")

print(data)

for k,v in data.items():
    print(k,v)
