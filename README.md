# VI-BEV: Vehicle-Infrastructure Collaborative Perception for 3D Object Detection on Bird’s-Eye View

**Jingxiong Meng and Junfeng Zhao**

## Abstract

Accurate 3D object detection is critical for the performance of autonomous driving systems. Automated vehicles are equipped with various sensors, such as cameras and LiDAR, to facilitate 3D object detection. However, these sensors often encounter challenges in environments with occlusions or when detecting distant objects. 

As infrastructure sensor technology continues to advance, leveraging these assets to enhance the perception capabilities of automated vehicles has become increasingly valuable for achieving more accurate and extensive 3D object detection. This paper presents a simple yet scalable framework that integrates infrastructure sensors with vehicle onboard sensors to perform 3D object detection using Bird’s Eye View (BEV) images. Furthermore, a cross-attention-based module is introduced to effectively fuse sensor information by utilizing the interactions between the sensors.

The proposed model is validated on the V2X-Sim dataset under two scenarios: short-range and long-range detection. Experimental results demonstrate that the model achieves superior accuracy, with improvements of **11.3%** and **15.1%** over the baseline in the short-range and long-range cases, respectively. Additionally, the proposed framework enables broader and more robust detections compared to the baseline model.
