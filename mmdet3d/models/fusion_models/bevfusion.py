from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

import math
import pickle
import numpy as np

import matplotlib.pyplot as plt
import time
from Transformer_Encoder import TransformerEncoder
__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        # initialize the infrastructure part
        if encoders.get("infra") is not None:
            self.encoders["infra"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["infra"]["backbone"]),
                    "neck": build_neck(encoders["infra"]["neck"]),
                    "vtransform": build_vtransform(encoders["infra"]["vtransform"]),
                }
            )

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.input_cross_transformer = TransformerEncoder(num_layers=4, input_dim=768, dim_feedforward=768*3, num_heads=2, cross_label=True, bev_cross_label=False)
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']
        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        pos_y = np.arange(H)
        pos_x = np.arange(W)
        pos_y_grid, pos_x_grid = np.meshgrid(pos_y, pos_x, indexing='ij')
        pos_enc = np.zeros((B*N, H, W, 256))
        for i in range(256//2):
            pos_enc[:,:,:,2*i] = np.sin(pos_y_grid/10000**(2*i/256))
            pos_enc[:,:,:,2*i+1] = np.cos(pos_x_grid/10000**(2*i/256))
        pos_enc = torch.tensor(pos_enc, dtype=torch.float16).cuda()
        img_pe = pos_enc.view(B*N, -1, H, W)
        mask_map = torch.ones(H,W)
        transformer_features = []
        final_features = []
        for i in range(len(x)):
            output = self.input_cross_transformer(x,feat_tag=i,pe=img_pe,mask_map=mask_map)
            transformer_features.append(output)
        final_features = torch.cat(transformer_features,dim=0)
        x = final_features
        


        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )
        return x
    def extract_infra_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x = self.encoders["infra"]["backbone"](x)
        x = self.encoders["infra"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        x = self.encoders["infra"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )
        return x
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    # def extract_lidar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    # def extract_radar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.radar_voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["radar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    # @torch.no_grad()
    # @force_fp32()
    # def radar_voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders["radar"]["voxelize"](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode="constant", value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
    #                 -1, 1
    #             )
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        auxiliary_losses = {}
        if 'infra' in list(self.encoders.keys()):
            vehicle_lidar_path = metas[0]['lidar_path']
            infra_lidar_path = vehicle_lidar_path.replace('LIDAR_TOP_id_1', 'LIDAR_TOP_id_0')
            with open('lidar_transform_dict.pkl', 'rb') as file:
                transform_dict = pickle.load(file)
            transformation_matrix = transform_dict[infra_lidar_path]
                    # load infrastructure lidar
            infra_point_clouds = np.fromfile(infra_lidar_path, dtype=np.float32)
            infra_point_clouds = infra_point_clouds.reshape((-1, 5))

            np.random.shuffle(infra_point_clouds)
            
            infra_point_clouds_raw = torch.from_numpy(infra_point_clouds).cuda().clone()
                    # transform infra lidar to vehicle lidar
            infra_point_cloud_geo = infra_point_clouds[:,:3]
            points_homogeneous = np.hstack((infra_point_cloud_geo, np.ones((infra_point_clouds.shape[0], 1))))
            infra_points_transformed = (transformation_matrix @ points_homogeneous.T).T
            infra_point_clouds[:, :3] = infra_points_transformed[:, :3]

            ones = np.ones((infra_point_clouds.shape[0],1))
            homo_points = np.hstack([infra_point_clouds[:,:3],ones])
            transformed_points = lidar_aug_matrix[0].cpu()@homo_points.T
            infra_point_clouds[:,:3] = transformed_points[:3,:].T
            filter_infra_points = infra_point_clouds[(infra_point_clouds[:,0]>-54.0)|(infra_point_clouds[:,0]<54.0)|(infra_point_clouds[:,1]>-54.0)|(infra_point_clouds[:,1]<54.0)]
            # filter_infra_points = infra_point_clouds[(infra_point_clouds[:,0]>-108.0)|(infra_point_clouds[:,0]<108.0)|(infra_point_clouds[:,1]>-108.0)|(infra_point_clouds[:,1]<108.0)]
            
            point_cloud_concat = torch.concat((points[0], torch.from_numpy(filter_infra_points).cuda()),dim=0)
            unique_point_cloud = torch.unique(point_cloud_concat, dim=0)
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                # Extract on-board vehicle img
                img_veh = img[:,:6]
                camera2ego_veh = camera2ego[:,:6]
                lidar2camera_veh = lidar2camera[:,:6]
                lidar2image_veh = lidar2image[:,:6]
                camera_intrinsics_veh = camera_intrinsics[:,:6]
                camera2lidar_veh = camera2lidar[:,:6]
                img_aug_matrix_veh = img_aug_matrix[:,:6]
                feature = self.extract_camera_features(
                    img_veh,
                    points,
                    radar,
                    camera2ego_veh,
                    lidar2ego,
                    lidar2camera_veh,
                    lidar2image_veh,
                    camera_intrinsics_veh,
                    camera2lidar_veh,
                    img_aug_matrix_veh,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                torch.cuda.empty_cache()
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            
            elif sensor == "lidar":
                if 'infra' in list(self.encoders.keys()):
                    feature = self.extract_features([unique_point_cloud], sensor)
                    #feature = self.extract_features(points, sensor)
                else:
                    feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            elif sensor == "infra":
                img_infra = img[:,6:]
                camera2ego_infra = camera2ego[:,6:]
                lidar2camera_infra = lidar2camera[:,6:]
                lidar2image_infra = lidar2image[:,6:]
                camera_intrinsics_infra = camera_intrinsics[:,6:]
                camera2lidar_infra = camera2lidar[:,6:]
                img_aug_matrix_infra = img_aug_matrix[:,6:]
                extra_trans_rot = torch.eye(4).unsqueeze(0).cuda()
                feature = self.extract_infra_features(
                    img_infra,
                    [torch.from_numpy(infra_point_clouds).cuda()],
                    radar,
                    camera2ego_infra,
                    lidar2ego,
                    lidar2camera_infra,
                    lidar2image_infra,
                    camera_intrinsics_infra,
                    camera2lidar_infra,
                    img_aug_matrix_infra,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            
            features.append(feature)
        #for i in range(3):
            #image = features[i][0,1,:,:]
            #plt.imshow(image.detach().cpu())
            #plt.axis('on')
            #plt.colorbar()
            #plt.show()
        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            None_check = any(x is None for x in features)
            if None_check == True:
                #fused_feature = torch.cat((features[0],features[1]),dim=1)
                conv_layer = nn.Conv2d(in_channels=80, out_channels=256, kernel_size=1).half().cuda()
                
                feat_exp = conv_layer(features[0])
                x = features[1]+feat_exp # got 0.693 by expand camera feat from 80 -> 256
                
            else:
                x = self.fuser(features)

        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]
        
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    #x[0] = x[0].float()
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                                 
                elif type == "map":
                    #x[0] = x[0].float()
                    losses = head(x, gt_masks_bev)
                    
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    #x[0] = x[0].float()
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    #x[0] = x[0].float()
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

