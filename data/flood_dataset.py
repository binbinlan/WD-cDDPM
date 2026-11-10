import os
import torch
from torch.utils.data import Dataset
import numpy as np


class FloodDataset(Dataset):
    def __init__(self, data_root, split='train', skip_start=0, skip_end=0, total_samples=5000,
                 max_depth_threshold=30.0, rain_dir_name='cumulative_rainfall_npy',
                 expected_rain_channels=1, max_rain_mm=300.0,
                 require_all_inputs=False):
        """
        data_root: LISFLOOD results 文件夹路径
        max_depth_threshold: 水深异常值剔除阈值（默认 30.0 米）
        """
        self.data_root = data_root
        self.split = split
        self.expected_rain_channels = int(expected_rain_channels)
        self.max_rain_mm = float(max_rain_mm)
        self.require_all_inputs = bool(require_all_inputs)
        if self.expected_rain_channels < 1:
            raise ValueError('expected_rain_channels must be at least 1')
        if self.max_rain_mm <= 0:
            raise ValueError('max_rain_mm must be positive')

        # 定位同级的降雨和地形文件夹
        parent_dir = os.path.dirname(self.data_root)
        self.rain_dir = os.path.join(parent_dir, *rain_dir_name.replace('\\', '/').split('/'))
        self.terrain_dir = os.path.join(parent_dir, 'terrain')

        # 1. 确定我们要扫描的物理基准索引 i 的范围
        start_i = skip_start
        end_i = total_samples - skip_end - 1

        # 2. 动态扫描并进行【质量控制 (QC)】
        valid_indices = []
        missing_indices = []
        print(f"[{split}] Scanning & QC checking dataset from index {start_i} to {end_i}...")

        for i in range(start_i, end_i + 1):
            rain_id = i
            dem_id = f"{i:02d}"
            res_id = i + 1

            # 拼接三个文件的真实路径
            rain_path = os.path.join(self.rain_dir, f"cumulative_rain_seed_{rain_id}.npy")
            dem_path = os.path.join(self.terrain_dir, f"dem_{dem_id}.dem")

            folder_path = os.path.join(self.data_root, f"results_{res_id}")
            depth_path = os.path.join(folder_path, f"seed_{res_id}.max")

            # 当 3 个文件都存在时，才进行数值检查
            if os.path.exists(dem_path) and os.path.exists(depth_path) and os.path.exists(rain_path):

                # === 🎯 核心新增：质量控制 (QC) 拦截异常水深 ===
                # 调用读取函数快速读取 max 文件 (已处理-9999空值)
                depth_data = self._read_ascii_raster(depth_path)
                max_d = depth_data.max()

                if max_d > max_depth_threshold:
                    print(f"  -> [QC 拦截] 剔除样本 {i} (results_{res_id}): 最大水深异常，达 {max_d:.2f} m")
                    continue  # 跳过这个样本，不加入有效列表
                # ================================================

                # 如果安全，才加入合法名单
                valid_indices.append(i)
            else:
                missing_indices.append(i)

            # 加个简单的进度提示，因为读文件需要一点点时间
            if i % 1000 == 0 and i > start_i:
                print(f"  ... 已扫描至 {i}, 当前有效样本: {len(valid_indices)}")

        if self.require_all_inputs and missing_indices:
            preview = ', '.join(str(i) for i in missing_indices[:10])
            suffix = ' ...' if len(missing_indices) > 10 else ''
            raise FileNotFoundError(
                f'[{split}] Missing one or more DEM/rain/depth inputs for '
                f'{len(missing_indices)} requested samples: {preview}{suffix}'
            )

        print(f"[{split}] Scan Complete! Found {len(valid_indices)} perfectly matched and clean sample pairs.")

        self.indices = valid_indices

    def _read_ascii_raster(self, filepath):
        """读取含有 6 行头文件的 LISFLOOD 结果"""
        data = np.loadtxt(filepath, skiprows=6)
        data = np.where(data == -9999.0, 0.0, data)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 拿到物理基准索引 i
        i = self.indices[idx]

        # 🎯 核心校正：生成对应的三种 ID
        rain_id = i
        dem_id = f"{i:02d}"
        res_id = i + 1

        # 拼接读取路径
        rain_path = os.path.join(self.rain_dir, f"cumulative_rain_seed_{rain_id}.npy")
        dem_path = os.path.join(self.terrain_dir, f"dem_{dem_id}.dem")

        folder_path = os.path.join(self.data_root, f"results_{res_id}")
        depth_path = os.path.join(folder_path, f"seed_{res_id}.max")

        # 1. 读取数据 [256, 256]
        dem_data = self._read_ascii_raster(dem_path)
        depth_data = self._read_ascii_raster(depth_path)
        rain_data = np.load(rain_path, allow_pickle=False)
        if rain_data.ndim == 2:
            rain_data = np.expand_dims(rain_data, axis=0)
        if rain_data.ndim != 3 or rain_data.shape[0] != self.expected_rain_channels:
            raise ValueError(
                f'Rainfall array {rain_path} has shape {rain_data.shape}; expected '
                f'[{self.expected_rain_channels}, height, width]'
            )
        if rain_data.shape[1:] != dem_data.shape:
            raise ValueError(
                f'Rainfall grid {rain_data.shape[1:]} does not match DEM grid '
                f'{dem_data.shape}: {rain_path}'
            )
        if not np.isfinite(rain_data).all() or np.any(rain_data < 0.0):
            raise ValueError(f'Rainfall array contains invalid values: {rain_path}')

        depth_data[depth_data < 0.05] = 0.0 # 将小于 0.05m 的极浅水（地表润湿/数值残差）直接清零

        # 2. 动态计算坡度
        dy, dx = np.gradient(dem_data)
        slope_data = np.sqrt(dx ** 2 + dy ** 2)

        # 3. 归一化处理（扩散模型需要 [-1, 1] 的分布）

        # DEM: 局部归一化
        dem_min, dem_max = dem_data.min(), dem_data.max()
        if dem_max > dem_min:
            dem_norm = (dem_data - dem_min) / (dem_max - dem_min) * 2.0 - 1.0
        else:
            dem_norm = np.zeros_like(dem_data)

        # Slope: 局部归一化
        slope_min, slope_max = slope_data.min(), slope_data.max()
        if slope_max > slope_min:
            slope_norm = (slope_data - slope_min) / (slope_max - slope_min) * 2.0 - 1.0
        else:
            slope_norm = np.zeros_like(slope_data)

        # 降雨归一化 (使用全局最大阈值，确保模型能区分大小雨)
        rain_norm = (rain_data / self.max_rain_mm) * 2.0 - 1.0
        rain_norm = np.clip(rain_norm, -1.0, 1.0)

        # 水深: 全局归一化
        depth_sqrt = np.sqrt(depth_data)
        depth_norm = (depth_sqrt / 3.0) * 2.0 - 1.0
        depth_norm = np.clip(depth_norm, -1.0, 1.0)

        # 4. 增加 Channel 维度
        dem_norm = np.expand_dims(dem_norm, axis=0)
        slope_norm = np.expand_dims(slope_norm, axis=0)
        # === 将 DEM、Slope 和 Rain 拼在一起 ===
        cond_image = np.concatenate([dem_norm, slope_norm, rain_norm], axis=0)

        # 目标水深 (1个通道)
        gt_image = np.expand_dims(depth_norm, axis=0)

        return {
            'cond_image': torch.FloatTensor(cond_image),
            'gt_image': torch.FloatTensor(gt_image),
            # 以物理索引 i 命名保存的结果图片
            'path': f"sample_idx_{i}.png"
        }
