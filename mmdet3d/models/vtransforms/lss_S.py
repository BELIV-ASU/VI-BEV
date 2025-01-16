from typing import Tuple

from mmcv.runner import force_fp32
from torch import nn

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseTransform
from DepthAnything.depth_anything_v2.dpt import DepthAnythingV2
import torchvision
from torchvision.models.segmentation import fcn_resnet50
import time

__all__ = ["LSSTransform"]


@VTRANSFORMS.register_module()
class LSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        #self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )

        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # depth estimation network
        self.pyramid_depth_net = fcn_resnet50(pretrained=False)
        self.conv = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1)

        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, x_raw):
        B, N, C, fH, fW = x.shape

        # Generate the ground truth depth information
        # start_time = time.time()
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        encoder = 'vits' # or 'vits', 'vitb', 'vitg'
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'DepthAnything/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        model = model.to(DEVICE).eval()
        d_list = []
        for i in range(len(x_raw)):
            per_d = model.infer_image(x_raw[i].permute(1,2,0).cpu().numpy())  # Depth Ground truth, input size is [256,704,3]  the output size is [256,704]
            per_d = torch.from_numpy(per_d).cuda().unsqueeze(0).unsqueeze(0)
            d_list.append(per_d)
        d = torch.cat([t for t in d_list], dim=0).cuda()   # The size is [6,1,256,704]

        # Generate depth estimation from simple network
        d_est = self.pyramid_depth_net(x_raw)['out']  # Do depth estimation on original image, input size is [6,3,256,704] [B * N, C, H, W]
        d_est = self.conv(d_est)    # The size is [6,1,256,704]

        # Calculate the loss between estimation and ground truth
        mse_loss = nn.MSELoss()
        loss = mse_loss(d, d_est)

        d_est = d_est.unsqueeze(0)
        d_est = d_est.view(B * N, *d_est.shape[2:])
        d_est = self.dtransform(d_est)
        # end_time = time.time()
        # print('The depth estimation time', end_time - start_time)
        print(self.training)

        x = x.view(B * N, C, fH, fW)
        x = torch.cat([d_est, x], dim=1)

        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)

        # visualize the depth image
        # depth = depth.view(B, N, self.D, fH, fW)
        # final_depth_indices = torch.argmax(depth, dim=2)
        # batch_index = 0
        # view_index = 2
        # final_depth_map = final_depth_indices[batch_index, view_index].cpu().numpy()
        # final_depth_map = final_depth_map.astype(np.float32)
        # final_depth_map = (final_depth_map - np.min(final_depth_map)) / (np.max(final_depth_map) - np.min(final_depth_map))

        # # Step 8: Apply Gaussian smoothing
        # smoothed_depth_map = gaussian_filter(final_depth_map, sigma=1)

        # # Step 9: Upsample the depth map
        # upsampled_depth_map = cv2.resize(final_depth_map, (fW * 4, fH * 4), interpolation=cv2.INTER_LINEAR)

        # # Step 10: Apply colormap
        # colored_depth_map = cv2.applyColorMap((upsampled_depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # plt.imshow(colored_depth_map, cmap='gray')  # Use 'viridis' colormap for better visualization
        # plt.colorbar(label='Depth')
        # plt.title('Final Depth Map')
        # plt.xlabel('Width')
        # plt.ylabel('Height')
        # plt.show()

        return x, loss

    def forward(self, *args, **kwargs):
        x, loss = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x, loss
