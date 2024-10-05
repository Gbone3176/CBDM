import numpy as np

# 加载 .npz 文件
with np.load('../stats/cifar10.train.npz') as data:
    # 打印文件中所有的数组名称
    print("Arrays in the NPZ file:")
    for key in data:
        print(key)

    # 遍历文件中的每个数组并打印内容
    for key in data:
        print(f"\nContents of '{key}':")
        print(data[key].shape)