# import os
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision.transforms import (
#     Compose,
#     InterpolationMode,
#     Resize,
#     ToTensor,
# )


# def ImageTransform(loadSize):
#     return {
#         "train": Compose(
#             [
#                 Resize(loadSize, interpolation=InterpolationMode.BILINEAR),
#                 ToTensor(),
#             ]
#         ),
#         "test": Compose(
#             [
#                 Resize(loadSize, interpolation=InterpolationMode.BILINEAR),
#                 ToTensor(),
#             ]
#         ),
#     }


# class UIEData(Dataset):
#     def __init__(
#         self, path_img, path_gt, path_gt_depth, path_img_hist, loadSize, mode=1
#     ):
#         super().__init__()
#         self.path_img = path_img
#         self.path_gt = path_gt
#         self.path_gt_depth = path_gt_depth
#         self.path_img_hist = path_img_hist

#         self.loadsize = loadSize  # e.g. (336, 336) or 336
#         self.crop_pad_size = (
#             loadSize[0] if isinstance(loadSize, (tuple, list)) else loadSize
#         )
#         self.mode = mode

#         self.data_img = os.listdir(self.path_img)
#         self.data_gt = os.listdir(self.path_gt)
#         self.data_gt_depth = os.listdir(self.path_gt_depth)
#         self.data_img_hist = os.listdir(self.path_img_hist)
#         if mode == 1:
#             self.ImgTrans = ImageTransform(loadSize)["train"]
#         else:
#             self.ImgTrans = ImageTransform(loadSize)["test"]

#     def __len__(self):
#         return len(self.data_gt)

#     def __getitem__(self, idx):
#         img = Image.open(os.path.join(self.path_img, self.data_img[idx])).convert("RGB")
#         gt = Image.open(os.path.join(self.path_gt, self.data_img[idx])).convert("RGB")
#         label_depth = Image.open(
#             os.path.join(self.path_gt_depth, self.data_img[idx])
#         ).convert("RGB")
#         img_hist = Image.open(
#             os.path.join(self.path_img_hist, self.data_img[idx])
#         ).convert("RGB")

#         name = self.data_img[idx]
#         h, w = img.size

#         if self.mode == 1:
#             seed = torch.random.seed()
#             torch.random.manual_seed(seed)
#             img = self.ImgTrans(img)
#             torch.random.manual_seed(seed)
#             gt = self.ImgTrans(gt)
#             torch.random.manual_seed(seed)
#             label_depth = self.ImgTrans(label_depth)
#             torch.random.manual_seed(seed)
#             img_hist = self.ImgTrans(img_hist)

#         else:
#             img = self.ImgTrans(img)
#             gt = self.ImgTrans(gt)
#             label_depth = self.ImgTrans(label_depth)
#             img_hist = self.ImgTrans(img_hist)

#         return img, gt, label_depth, img_hist, name, (h, w)



import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Resize,
    ToTensor,
)
import logging


def ImageTransform(loadSize):
    return {
        "train": Compose(
            [
                Resize(loadSize, interpolation=InterpolationMode.BILINEAR),
                ToTensor(),
            ]
        ),
        "test": Compose(
            [
                Resize(loadSize, interpolation=InterpolationMode.BILINEAR),
                ToTensor(),
            ]
        ),
    }


class UIEData(Dataset):
    def __init__(
        self, path_img, path_gt, path_gt_depth, path_img_hist, loadSize, mode=1
    ):
        super().__init__()
        self.path_img = path_img
        self.path_gt = path_gt
        self.path_gt_depth = path_gt_depth
        self.path_img_hist = path_img_hist
        self.mode = mode
        self.logger = logging.getLogger(__name__)

        # 验证所有路径是否存在
        for path, name in [(path_img, "输入图片"), (path_gt, "GT图片"), 
                          (path_gt_depth, "深度图"), (path_img_hist, "直方图")]:
            if not os.path.exists(path):
                self.logger.error(f"{name}路径不存在: {path}")
        
        # 获取所有文件名列表
        self.data_img = sorted(os.listdir(self.path_img))
        self.data_gt = sorted(os.listdir(self.path_gt))
        self.data_gt_depth = sorted(os.listdir(self.path_gt_depth))
        self.data_img_hist = sorted(os.listdir(self.path_img_hist))
        
        # 记录每个列表的大小
        self.logger.info(f"输入图片数量: {len(self.data_img)}")
        self.logger.info(f"GT图片数量: {len(self.data_gt)}")
        self.logger.info(f"深度图数量: {len(self.data_gt_depth)}")
        self.logger.info(f"直方图数量: {len(self.data_img_hist)}")
        
        # 检查文件名是否一致（取交集）
        all_files = set(self.data_img) & set(self.data_gt) & set(self.data_gt_depth) & set(self.data_img_hist)
        self.data_files = sorted(list(all_files))
        
        if len(self.data_files) == 0:
            self.logger.error("四个文件夹中没有共同的图片文件！")
            self.logger.error(f"输入图片示例: {self.data_img[:5]}")
            self.logger.error(f"GT图片示例: {self.data_gt[:5]}")
            self.logger.error(f"深度图示例: {self.data_gt_depth[:5]}")
            self.logger.error(f"直方图示例: {self.data_img_hist[:5]}")
            raise ValueError("四个文件夹中没有共同的图片文件！")
        
        self.logger.info(f"共同文件数量: {len(self.data_files)}")
        if len(self.data_files) < min(len(self.data_img), len(self.data_gt), len(self.data_gt_depth), len(self.data_img_hist)):
            self.logger.warning(f"部分文件不在所有文件夹中，只使用 {len(self.data_files)} 个共同文件")
            self.logger.info(f"前10个共同文件: {self.data_files[:10]}")
        
        self.loadsize = loadSize  # e.g. (336, 336) or 336
        self.crop_pad_size = (
            loadSize[0] if isinstance(loadSize, (tuple, list)) else loadSize
        )
        
        if mode == 1:
            self.ImgTrans = ImageTransform(loadSize)["train"]
        else:
            self.ImgTrans = ImageTransform(loadSize)["test"]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        try:
            # 使用共同的文件名
            filename = self.data_files[idx]
            
            # 加载图片
            img_path = os.path.join(self.path_img, filename)
            gt_path = os.path.join(self.path_gt, filename)
            depth_path = os.path.join(self.path_gt_depth, filename)
            hist_path = os.path.join(self.path_img_hist, filename)
            
            # 检查文件是否存在
            if not os.path.exists(img_path):
                self.logger.error(f"输入图片不存在: {img_path}")
                raise FileNotFoundError(f"输入图片不存在: {img_path}")
            if not os.path.exists(gt_path):
                self.logger.error(f"GT图片不存在: {gt_path}")
                raise FileNotFoundError(f"GT图片不存在: {gt_path}")
            if not os.path.exists(depth_path):
                self.logger.error(f"深度图不存在: {depth_path}")
                raise FileNotFoundError(f"深度图不存在: {depth_path}")
            if not os.path.exists(hist_path):
                self.logger.error(f"直方图不存在: {hist_path}")
                raise FileNotFoundError(f"直方图不存在: {hist_path}")
            
            img = Image.open(img_path).convert("RGB")
            gt = Image.open(gt_path).convert("RGB")
            label_depth = Image.open(depth_path).convert("RGB")
            img_hist = Image.open(hist_path).convert("RGB")
            
            name = filename
            h, w = img.size
            
            if self.mode == 1:
                # 训练模式：使用相同的随机种子确保变换一致
                seed = torch.random.seed()
                torch.random.manual_seed(seed)
                img = self.ImgTrans(img)
                torch.random.manual_seed(seed)
                gt = self.ImgTrans(gt)
                torch.random.manual_seed(seed)
                label_depth = self.ImgTrans(label_depth)
                torch.random.manual_seed(seed)
                img_hist = self.ImgTrans(img_hist)
            else:
                # 测试模式
                img = self.ImgTrans(img)
                gt = self.ImgTrans(gt)
                label_depth = self.ImgTrans(label_depth)
                img_hist = self.ImgTrans(img_hist)
            
            return img, gt, label_depth, img_hist, name, (h, w)
            
        except Exception as e:
            self.logger.error(f"加载数据失败，索引 {idx}, 文件 {filename if 'filename' in locals() else 'unknown'}: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回一个默认的黑色图像，尺寸为256x256
            default_img = torch.zeros((3, self.loadsize[0], self.loadsize[1]))
            default_name = "error_image"
            return default_img, default_img, default_img, default_img, default_name, (self.loadsize[0], self.loadsize[1])