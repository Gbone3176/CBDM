import os
import numpy as np
import torch
from numpy.core.defchararray import endswith
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from inception import InceptionV3

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_statistics(images, num_images=None, batch_size=50, use_torch=False,
                   verbose=False, parallel=False):
    """when `images` is a python generator, `num_images` should be given"""

    if num_images is None:
        try:
            num_images = len(images)
        except:
            raise ValueError(
                "when `images` is not a list like object (e.g. generator), "
                "`num_images` should be given")

    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx1]).to(device)
    model.eval()

    if parallel:
        model = torch.nn.DataParallel(model)

    if use_torch:
        fid_acts = torch.empty((num_images, 2048)).to(device)
    else:
        fid_acts = np.empty((num_images, 2048))

    iterator = iter(tqdm(images, total=num_images,
                          dynamic_ncols=True, leave=False, disable=not verbose,
                          desc="get_inception_and_fid_score"))

    start = 0
    while True:
        batch_images = []
        # get a batch of images from iterator
        try:
            for _ in range(batch_size):
                batch_images.append(next(iterator))
        except StopIteration:
            if len(batch_images) == 0:
                break
            pass
        batch_images = np.stack(batch_images, axis=0)
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
            if use_torch:
                fid_acts[start: end] = pred.view(-1, 2048)
            else:
                fid_acts[start: end] = pred.view(-1, 2048).cpu().numpy()
        start = end

    if use_torch:
        m1 = torch.mean(fid_acts, axis=0)
        s1 = torch.cov(fid_acts, rowvar=False)
    else:
        m1 = np.mean(fid_acts, axis=0)
        s1 = np.cov(fid_acts, rowvar=False)
    return m1, s1


def load_images_from_folder(folder):
    """从文件夹加载所有图像并返回 NumPy 数组"""
    images = []
    for filename in os.listdir(folder):
        if not filename.endswith('.jpg'):  # 使用 endswith() 方法检查文件扩展名
            continue
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')  # 确保是 RGB 图像
        img = img.resize((299, 299))  # Inception v3 输入尺寸
        img = np.array(img)
        images.append(img)
    return np.array(images)


def save_statistics_to_npz(m1, s1, output_file):
    """将均值和协方差保存为 .npz 文件"""
    np.savez_compressed(output_file, mean=m1.cpu().numpy(), covariance=s1.cpu().numpy())
    print(f"Statistics saved to {output_file}")


if __name__ == "__main__":
    folder_path = r"F:\data\ISIC2018\val_data"  # 替换为您图像的文件夹路径
    output_file = "ISIC2018.train.npz"  # 输出文件名

    # 从文件夹加载图像
    print('loading pics from folder {} ...'.format(folder_path))
    images = load_images_from_folder(folder_path)
    print('calculating statistics...')
    # 计算 Inception 特征
    m1, s1 = get_statistics(images, num_images=len(images), use_torch=True)
    print(f'm1={m1}, s1={s1}')
    # 保存统计信息
    save_statistics_to_npz(m1, s1, output_file)
