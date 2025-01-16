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

from Transformer_Encoder import TransformerEncoder
import Spatial_info
import matplotlib.pyplot as plt
import open3d as o3d

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
            # initialize the transformer here
            # self.input_self_transformer = TransformerEncoder(num_layers=1,input_dim=128,dim_feedforward=128,num_heads=2,cross_label=False,bev_cross_label=False)
            # self.input_cross_transformer = TransformerEncoder(num_layers=3, input_dim=256,dim_feedforward=128, num_heads=4,cross_label=False,bev_cross_label=False)

            # self.output_self_transformer = TransformerEncoder(num_layers=2,input_dim=320,dim_feedforward=128,num_heads=2,cross_label=False,bev_cross_label=False)
            # self.output_cross_transformer = TransformerEncoder(num_layers=3,input_dim=320,dim_feedforward=128,num_heads=2,cross_label=True,bev_cross_label=True)
            self.fuser = build_fuser(fuser)
        else:
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

        # Initial neural network for BEV & img position embedding
        # self.img_position_embed = nn.Linear(3, 3)   # input_dim = coordinate_dim = 3, output_dim = embed_dim

        self.count = 0

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
        gt_bboxes_3d,
        gt_labels_3d,
        gt_depths=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x_raw = x.clone()
        # x_original = x.clone()
        # Do img transformation here

        # x_grid = torch.arange(W).view(1, W).expand(H, W)
        # y_grid = torch.arange(H).view(H, 1).expand(H, W)
        # xy_grid = torch.stack((x_grid, y_grid), dim=-1).float()  # create a grid [H,W]
        # xy = xy_grid.view(-1, 2).unsqueeze(0).repeat(B * N, 1, 1)  # flatten the grid
        # xy_h = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
        # xy_h = torch.cat([xy_h, torch.ones_like(xy_h[..., :1])], dim=-1)
        # K_inv = torch.inverse(camera_intrinsics)
        # K_inv = K_inv.view(B * N, 4, 4)
        # cam_coords = torch.bmm(xy_h.cuda(device=0), K_inv.transpose(1, 2))
        # camera2lidar_trans = camera2lidar.view(B * N, 4, 4)
        # lidar_coords_h = torch.bmm(cam_coords, camera2lidar_trans.transpose(1, 2))
        # img_lidar_coords = lidar_coords_h[..., :3].view(B, N, H, W, 3)
        # # Generate the position embedding
        # img_pe = self.img_position_embed(img_lidar_coords).permute(0, 1, 4, 2, 3)
        # img_pe = img_pe.view(B * N, -1, H, W)
        # # img_pe = None
        # # # Generate the mask map here
        # # # mask_map = self.mask_map(x)   #[B*N, 1, H, W]
        # mask_map = None

        # # # Do transformer fusion here
        # # # Self-attention here
        # transformer_features = []
        # # # Cross-attention here
        # for i in range(len(x)):
        #     output = self.input_cross_transformer(x,feat_tag=i, pe=img_pe,mask_map=mask_map)
        #     transformer_features.append(output)
        
        # final_features = torch.cat(transformer_features,dim=0)
        # x = final_features + x
        # self.count += 1
        # if self.count % 2000 ==0:
        #     image = np.transpose(x[0].detach().cpu(), (1, 2, 0))
        #     image = np.clip(image, 0, 1)
        #     plt.imshow(image)
        #     plt.axis('off')  # Turn off axis
        #     plt.show()

        
        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x, depth_loss = self.encoders["camera"]["vtransform"](
            x,
            x_raw,
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
        # if self.count % 50 ==0:
        #     image = x[0,1,:,:]
        #     plt.imshow(image.detach().cpu())
        #     plt.axis('off')  # Turn off axis
        #     plt.colorbar()
        #     plt.show()
        return x, depth_loss
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
            
            infra_point_cloud_geo = infra_point_clouds[:,:3]
            points_homogeneous = np.hstack((infra_point_cloud_geo, np.ones((infra_point_clouds.shape[0], 1))))
            infra_points_transformed = (transformation_matrix @ points_homogeneous.T).T
            infra_point_clouds[:, :3] = infra_points_transformed[:, :3]
            ones = np.ones((infra_point_clouds.shape[0], 1))
            homogeneous_points = np.hstack([infra_point_clouds[:,:3], ones])
            transformed_points = lidar_aug_matrix[0].cpu() @ homogeneous_points.T
            infra_point_clouds[:,:3] = transformed_points[:3,:].T

            filter_infra_points = infra_point_clouds[(infra_point_clouds[:,0]>-54.0)|(infra_point_clouds[:,0]<54.0)|(infra_point_clouds[:,1]>-54.0)|(infra_point_clouds[:,1]<54.0)]

            point_cloud_concat = torch.concat((points[0], torch.from_numpy(filter_infra_points).cuda().to(dtype=torch.float16)),dim=0)
            uniqe_point_cloud = torch.unique(point_cloud_concat,dim=0)
        # print(len(points[0]),len(uniqe_point_cloud))
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(uniqe_point_cloud[:,:3].detach().cpu())
        # o3d.visualization.draw_geometries([pcd])
        features = []
        auxiliary_losses = {}
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                # Extract on-board vehicle img
                img_veh = img[:,:2]
                camera2ego_veh = camera2ego[:,:2]
                lidar2camera_veh = lidar2camera[:,:2]
                lidar2image_veh = lidar2image[:,:2]
                camera_intrinsics_veh = camera_intrinsics[:,:2]
                camera2lidar_veh = camera2lidar[:,:2]
                img_aug_matrix_veh = img_aug_matrix[:,:2]
                feature, depth_loss = self.extract_camera_features(
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
                    gt_bboxes_3d,
                    gt_labels_3d,
                    gt_depths=depths,
                )
                torch.cuda.empty_cache()
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                if 'infra' in list(self.encoders.keys()):
                    feature = self.extract_features([uniqe_point_cloud], sensor)
                    # print(len(uniqe_point_cloud))
                else:
                    feature = self.extract_features(points, sensor)
                    
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            elif sensor == "infra":
                img_infra = img[:,6:]
                camera2ego_infra = camera2ego[:,2:]
                lidar2camera_infra = lidar2camera[:,2:]
                lidar2image_infra = lidar2image[:,2:]
                camera_intrinsics_infra = camera_intrinsics[:,2:]
                camera2lidar_infra = camera2lidar[:,2:]
                img_aug_matrix_infra = img_aug_matrix[:,2:]
                # extra_trans_rot = torch.eye(4).unsqueeze(0).cuda()
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
        # for i in range(3):
        #     image = features[i][0,1,:,:]
        #     plt.imshow(image.detach().cpu())
        #     plt.axis('on')  # Turn off axis
        #     plt.colorbar()
        #     plt.show()
        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            ## This part is for attention method
            # Do self-attention here (comment out the below part till x=self.fuser(features) for base)
            # transformer_features = []
            # final_features = []
            # bev_pe_list = []
            # # # Add on a positional embedding here
            # height, width = features[0].shape[-2], features[0].shape[-1]
            # normalized_coords = Spatial_info.generate_normalized_coordinates(height, width)
            # for i in range(len(features)):
            #     coord_transform = Spatial_info.CoordTransform(input_dim=2, output_dim=features[i].shape[1])
            #     bev_pe = coord_transform(normalized_coords).cuda(device=0)
            # #     output = self.output_self_transformer(features[i],feat_tag=0, pe=bev_pe)
            # #     transformer_features.append(output)
            #     bev_pe_list.append(bev_pe)
            # # Do cross-attention here
            # for i in range(len(features)):
            #     output = self.output_cross_transformer(features,feat_tag=i, pe=bev_pe_list)
            #     final_features.append(output)
            # x = self.fuser(final_features)
            # x = self.fuser(transformer_features)
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
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                    for key in losses:
                        losses[key] = losses[key] + depth_loss
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
                
            ## This part is for gradient debug use
            # for name, param in self.named_parameters():
            #     if name == 'input_self_transformer.layers.0.self_attn.o_proj.weight':
            #         print(param.grad)
            #     if param.grad is None:
            #         print(f"Parameter {name} did not receive a gradient")

            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
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

