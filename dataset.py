from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import torch
import h5py


def downscale_load_profile(high_res_load, low_res_length):
    high_res_load = np.asarray(high_res_load)
    high_res_length = high_res_load.shape[0]
    scaling_factor = high_res_length / low_res_length
    # Compute the start indices for each block
    indices = (np.arange(low_res_length) * scaling_factor).astype(int)
    # Use np.add.reduceat to compute the sum for each block
    block_sums = np.add.reduceat(high_res_load, indices)
    # Compute the count of elements in each block
    block_counts = np.diff(np.append(indices, high_res_length))
    # Calculate the average for each block
    low_res_load = block_sums / block_counts
    return low_res_load


class SR(Dataset):
    def __init__(self, hdf5_path, split, scale, input_size=360, std_min=0, std_max=0.2,
                 train_mean=None, train_std=None):
        """
        Args:
            hdf5_path (str): HDF5 文件路径。
            split (str): 使用的数据集划分 ('train', 'val' 或 'test')。
            scale (float): 下采样比例。
            input_size (int): 基础输入尺寸。
            std_min (float): 添加噪声的最小标准差（在标准化后的数据上）。
            std_max (float): 添加噪声的最大标准差（在标准化后的数据上）。
            train_mean (float): 训练集计算得到的均值。如果为 None 则自动计算。
            train_std (float): 训练集计算得到的标准差。如果为 None 则自动计算。
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.scale = scale
        self.input_size = input_size
        self.std_min = std_min
        self.std_max = std_max

        # 延迟初始化：在 __getitem__ 中打开 HDF5 文件
        self.h5_file = None
        self.cache = {}

        # 暂时打开 HDF5 文件，获取当前 split 的所有 key
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[self.split]
            self.keys = list(group.keys())

        # 如果没有提供训练集均值和标准差，则对训练集数据进行计算
        if train_mean is None or train_std is None:
            with h5py.File(self.hdf5_path, 'r') as f:
                train_group = f['train']
                all_values = []
                for key in train_group.keys():
                    data = train_group[key][()]
                    all_values.append(data)
                # 将所有数据拼接成一维数组
                all_values = np.concatenate(all_values)
                self.train_mean = float(np.mean(all_values))
                self.train_std = float(np.std(all_values))
            print(f"Computed training mean: {self.train_mean}, std: {self.train_std}")
        else:
            self.train_mean = train_mean
            self.train_std = train_std

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # 如果文件还未打开，则打开 HDF5 文件
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        key = self.keys[idx]

        # 尝试使用缓存
        if key in self.cache:
            data = self.cache[key]
        else:
            data = self.h5_file[self.split][key][()]
            self.cache[key] = data

        # 标准化数据：使得数据分布具有统一的均值和标准差（例如均值0，方差1）
        segment = (data - self.train_mean) / self.train_std

        # 利用自定义函数对标准化后的数据进行下采样处理
        sr = downscale_load_profile(segment, int(self.input_size * self.scale))
        hr = downscale_load_profile(segment, self.input_size)
        lr = downscale_load_profile(hr, int(self.input_size / self.scale))

        # 生成噪声，并创建额外的变体
        std_range = (self.std_min, self.std_max)
        mean_val = 0.0
        noise_std = np.random.uniform(*std_range)  # 对所有噪声生成使用相同的 std

        # 根据一定概率决定是否添加噪声
        if np.random.rand() < 0.1:
            lr_son = lr  # 10% 的概率直接返回干净的 lr
        else:
            noise1 = np.random.normal(mean_val, noise_std, lr.shape)
            lr_son = lr + noise1

        noise2 = np.random.normal(mean_val, noise_std, lr_son.shape)
        gen = noise2 + lr

        noise3 = np.random.normal(mean_val, noise_std, hr.shape)
        hr_noise = noise3 + hr

        # 转换为 torch 张量并增加通道维度
        lr_son_tensor = torch.as_tensor(lr_son).float().unsqueeze(0)  # 带噪 lr2
        hr_tensor = torch.as_tensor(hr).float().unsqueeze(0)          # 干净 lr
        sr_tensor = torch.as_tensor(sr).float().unsqueeze(0)          # 干净 hr
        gen_tensor = torch.as_tensor(gen).float().unsqueeze(0)        # 另一种带噪 lr2 版本
        lr_tensor = torch.as_tensor(lr).float().unsqueeze(0)          # 干净 lr2
        hr_noise_tensor = torch.as_tensor(hr_noise).float().unsqueeze(0)  # 带噪 lr

        return lr_son_tensor, hr_tensor, sr_tensor, gen_tensor, lr_tensor, hr_noise_tensor


def reverse(normalized_data, train_min=0.0, train_max=21.51300048828125):
    """
    Reverts the normalization of data.

    Parameters:
        normalized_data (np.array): Normalized data array.
        train_min (float): Global minimum value from training data.
        train_max (float): Global maximum value from training data.

    Returns:
        np.array: The data in the original scale.
    """
    return normalized_data * (train_max - train_min) + train_min


class SR_fix_H5(Dataset):
    def __init__(self, hdf5_path, split, scale, input_size=360, fix=0.005, train_mean=None, train_std=None):
        """
        Args:
            hdf5_path (str): HDF5 文件路径。
            split (str): 数据集划分，例如 'train', 'val' 或 'test'。
            scale (float): 下采样比例。
            input_size (int): 基础输入尺寸。
            fix (float): 添加噪声的标准差；若 fix==0，则不添加噪声。
            train_mean (float): 训练集均值；如果为 None，则自动计算。
            train_std (float): 训练集标准差；如果为 None，则自动计算。
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.scale = scale
        self.input_size = input_size
        self.fix = fix

        # 延迟初始化：在 __getitem__ 中打开 HDF5 文件
        self.h5_file = None

        # 获取当前 split 下的所有 key
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[self.split]
            self.keys = list(group.keys())

        # 如果没有提供训练集均值和标准差，则自动计算
        if train_mean is None or train_std is None:
            with h5py.File(self.hdf5_path, 'r') as f:
                train_group = f['train']
                all_values = []
                for key in train_group.keys():
                    data = train_group[key][()]
                    all_values.append(data)
                # 拼接所有数据并计算均值和标准差
                all_values = np.concatenate(all_values)
                self.train_mean = float(np.mean(all_values))
                self.train_std = float(np.std(all_values))
            print(f"Computed training mean: {self.train_mean}, std: {self.train_std}")
        else:
            self.train_mean = train_mean
            self.train_std = train_std

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # 确保每个 worker 都有自己的文件句柄
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        key = self.keys[idx]
        data = self.h5_file[self.split][key][()]

        # 对数据进行标准化（零均值，单位标准差）
        segment = (data - self.train_mean) / self.train_std

        # 对标准化后的数据进行下采样处理
        sr = downscale_load_profile(segment, int(self.input_size * self.scale))
        hr = downscale_load_profile(segment, self.input_size)

        # 在标准化后的数据上添加噪声
        mean_val = 0.0
        noise1 = np.random.normal(mean_val, self.fix, hr.shape)
        hr_son = hr + noise1
        if self.fix == 0:
            hr_son = hr

        # 转换为 torch 张量并添加通道维度
        hr_son_tensor = torch.as_tensor(hr_son).float().unsqueeze(0)
        sr_tensor = torch.as_tensor(sr).float().unsqueeze(0)

        return hr_son_tensor, sr_tensor


class SRDataset(Dataset):

    def __init__(self, hdf5_path, split, scale, input_size=360, train_min=None, train_max=None):
        """
        Args:
            hdf5_path (str): Path to the HDF5 file.
            split (str): Which split to use (e.g., 'train', 'val', or 'test').
            scale (float): Scale factor for downscaling.
            input_size (int): Base input size.
            fix (float): Noise level; if fix==0, no noise is added.
            train_min (float): Global minimum value computed from training data. If None, it is computed.
            train_max (float): Global maximum value computed from training data. If None, it is computed.
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.scale = scale
        self.input_size = input_size

        # Lazy initialization: we'll open the HDF5 file in __getitem__.
        self.h5_file = None

        # Create a simple cache to store raw data (per worker)
        self.cache = {}

        # Open the HDF5 file once (temporarily) to fetch dataset keys for the specified split.
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[self.split]
            self.keys = list(group.keys())

        # If normalization parameters are not provided, and we're using the training set,
        # compute global min and max from all training data.
        if train_min is None or train_max is None:
            with h5py.File(self.hdf5_path, 'r') as f:
                train_group = f['train']
                all_values = []
                for key in train_group.keys():
                    data = train_group[key][()]
                    all_values.append(data)
                # Concatenate all data into one long array
                all_values = np.concatenate(all_values)
                self.train_min = float(np.min(all_values))
                self.train_max = float(np.max(all_values))
            print(f"Computed training min: {self.train_min}, max: {self.train_max}")
        else:
            self.train_min = train_min
            self.train_max = train_max


    def __len__(self):
        return len(self.keys)


    def __getitem__(self, idx):
        # Open the HDF5 file if not already opened (each worker gets its own handle)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        key = self.keys[idx]

        # Use cache if available
        if key in self.cache:
            data = self.cache[key]
        else:
            data = self.h5_file[self.split][key][()]
            self.cache[key] = data

        # Normalize using the training set min and max
        segment = (data - self.train_min) / (self.train_max - self.train_min)
        # Downscale the load profile using your custom function.
        hr = downscale_load_profile(segment, self.input_size)

        std_range = (0.001, 0.015)
        mean = 0.0
        std = np.random.uniform(*std_range)

        noise1 = np.random.normal(mean, std, hr.shape)
        noise2 = np.random.normal(mean, std, hr.shape)

        crop1_lr = torch.as_tensor(hr).float() + torch.as_tensor(noise1).float()
        crop2_lr = torch.as_tensor(hr).float() + torch.as_tensor(noise2).float()

        return crop1_lr.unsqueeze(0), crop2_lr.unsqueeze(0)


if __name__ == "__main__":

    train_dataset = SR(
        hdf5_path='./HDF5/output_data.h5',
        split='train',
        scale=4,  # Example scale factor
        input_size=360,
    )

    # Create a DataLoader for batching, shuffling, and parallel loading.
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Adjust batch size as needed
        shuffle=True,
        num_workers=6,  # Use more workers for faster loading if possible
        prefetch_factor=2  # can help with prefetching batches
    )

    for lr_son, hr, sr, gen, lr in train_loader:
        # Now each of these is a tensor batch, and you can pass them to your model.
        # Example: output = model(lr_son)
        print(lr_son.shape, hr.shape, sr.shape, gen.shape, lr.shape)
