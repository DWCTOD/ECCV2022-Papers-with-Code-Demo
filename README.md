# ECCV2022-Papers-with-Code-Demo
收集 ECCV 最新的成果，包括论文、代码和demo视频等，欢迎大家推荐！

欢迎关注公众号：AI算法与图像处理

 :star_and_crescent:**福利 注册即可领取 200 块计算资源 : https://www.bkunyun.com/wap/console?source=aistudy**
 [使用说明](https://mp.weixin.qq.com/s?__biz=MzU4NTY4Mzg1Mw==&amp;mid=2247521550&amp;idx=1&amp;sn=db4c7f609bd61ae7734b9e012a763f98&amp;chksm=fd8413eccaf39afa686f69f2df2463f4a6a8233ba3b3edf698513bbee556c9f6c21e835b8eb8&token=705359263&lang=zh_CN#rd)



:star2: [ECCV 2022](https://eccv2022.ecva.net/) 持续更新最新论文/paper和相应的开源代码/code！

:car: ECCV 2022 收录列表ID：https://ailb-web.ing.unimore.it/releases/eccv2022/accepted_papers.txt

:car: 官网链接：https://eccv2022.ecva.net

B站demo：https://space.bilibili.com/288489574

> :hand: ​注：欢迎各位大佬提交issue，分享ECCV 2022论文/paper和开源项目！共同完善这个项目
>
> 往年顶会论文汇总：

> [CVPR2022](https://github.com/DWCTOD/CVPR2022-Papers-with-Code-Demo)
> 
> [CVPR2021](https://github.com/DWCTOD/CVPR2022-Papers-with-Code-Demo/blob/main/CVPR2021.md)
>
> [ICCV2021](https://github.com/DWCTOD/ICCV2021-Papers-with-Code-Demo)

### **:fireworks: 欢迎进群** | Welcome

ECCV 2022 论文/paper交流群已成立！已经收录的同学，可以添加微信：**nvshenj125**，请备注：**ECCV+姓名+学校/公司名称**！一定要根据格式申请，可以拉你进群。

<a name="Contents"></a>

### :hammer: **目录 |Table of Contents（点击直接跳转）**

<details open>
<summary> 目录（右侧点击可折叠）</summary>

- [数据集/Dataset](#Dataset)
- [Image Classification](#ImageClassification)
- [GAN](#GAN)
- [NeRF](#NeRF)
- [Visual Transformer](#VisualTransformer)
- [多模态/Multimodal ](#Multimodal)
- [Vision-Language](#Vision-Language)
- [对比学习/Contrastive Learning](#ContrastiveLearning)
- [Domain Adaptation](#DomainAdaptation)
- [目标检测/Object Detection](#ObjectDetection)
- [目标跟踪/Object Tracking](#ObjectTracking)
- [语义分割/Segmentation](#Segmentation)
- [Video Segmentation](#VS)
- [医学图像分割/Medical Image Segmentation](#MIS)
- [Knowledge Distillation](#KnowledgeDistillation)
- [Action Detection](#ActionDetection)
- [Action Recognition](#ActionRecognition)
- [Anomaly Detection](#AnomalyDetection)
- [人脸识别/Face Recognition](#FaceRecognition)
- [人脸检测/Face Detection](#FaceDetection)
- [人脸活体检测/Face Anti-Spoofing](#FaceAnti-Spoofing)
- [人脸年龄估计/Age Estimation](#AgeEstimation)
- [人脸表情识别/Facial Expression Recognition](#FacialExpressionRecognition)
- [人脸属性识别/Facial Attribute Recognition](#FacialAttributeRecognition)
- [人脸编辑/Facial Editing](#FacialEditing)
- [人脸相关 / Face](#Face)
- [人体姿态估计/Human Pose Estimation](#HumanPoseEstimation)
- [3D reconstruction](#3DReconstruction)
- [Human Reconstruction](#HumanReconstruction)
- [Relighting](#Relighting)
- [DeepFake](#DeepFake)
- [OCR](#OCR)
- [Text Recognition](#TextRecognition)
- [点云/Point Cloud](#PointCloud)
- [光流估计/Flow Estimation](#FlowEstimation)
- [深度估计/Depth Estimation](#DepthEstimation)
- [车道线检测/Lane Detection](#LaneDetection)
- [轨迹预测/Trajectory Prediction](#TrajectoryPrediction)
- [超分/Super-Resolution](#Super-Resolution)
- [图像去噪/Image Denoising](#ImageDenoising)
- [图像去模糊/Image Deblurring](#ImageDeblurring)
- [图像复原/Image Restoration](#ImageRestoration)
- [图像增强/Image Enhancement](#ImageEnhancement)
- [图像修复/Image Inpainting](#ImageInpainting)
- [视频插帧/Video Interpolation](#VideoInterpolation)
- [Temporal Action Segmentation](#TemporalActionSegmentation)
- [检索/Image Retrieval](#ImageRetrieval)
- [Diffusion](#diffusion)
- [其他/Other](#Other)

</details>

<a name="Dataset"></a> 

## 数据集/Dataset

**COO: Comic Onomatopoeia Dataset for Recognizing Arbitrary or Truncated Texts**

- 论文/Paper: http://arxiv.org/pdf/2207.04675
- 代码/Code: https://github.com/ku21fan/COO-Comic-Onomatopoeia

**Exploring Fine-Grained Audiovisual Categorization with the SSW60 Dataset**

- 论文/Paper: http://arxiv.org/pdf/2207.10664
- 代码/Code: https://github.com/visipedia/ssw60

**BRACE: The Breakdancing Competition Dataset for Dance Motion Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2207.10120
- 代码/Code: https://github.com/dmoltisanti/brace

**CelebV-HQ: A Large-Scale Video Facial Attributes Dataset**

- 论文/Paper: http://arxiv.org/pdf/2207.12393
- 代码/Code: https://github.com/CelebV-HQ/CelebV-HQ

**Ithaca365: Dataset and Driving Perception under Repeated and Challenging Weather Conditions**

- 论文/Paper: http://arxiv.org/pdf/2208.01166
- 代码/Code: None

**Fine-Grained Egocentric Hand-Object Segmentation: Dataset, Model, and Applications**

- 论文/Paper: http://arxiv.org/pdf/2208.03826
- 代码/Code: https://github.com/owenzlz/EgoHOS

**TRoVE: Transforming Road Scene Datasets into Photorealistic Virtual Environments**

- 论文/Paper: http://arxiv.org/pdf/2208.07943
- 代码/Code: https://github.com/shubham1810/trove_toolkit

[返回目录/back](#Contents)

<a name="ImageClassification"></a> 

## Image Classification

**Tree Structure-Aware Few-Shot Image Classification via Hierarchical Aggregation**

- 论文/Paper: http://arxiv.org/pdf/2207.06989
- 代码/Code: https://github.com/remiMZ/HTS-ECCV22

**Bagging Regional Classification Activation Maps for Weakly Supervised Object Localization**

- 论文/Paper: http://arxiv.org/pdf/2207.07818
- 代码/Code: https://github.com/zh460045050/BagCAMs

**Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification**

- 论文/Paper: http://arxiv.org/pdf/2207.09519
- 代码/Code: https://github.com/gaopengcuhk/tip-adapter

**Invariant Feature Learning for Generalized Long-Tailed Classification**

- 论文/Paper: http://arxiv.org/pdf/2207.09504
- 代码/Code: https://github.com/kaihuatang/generalized-long-tailed-benchmarks.pytorch

**RealFlow: EM-based Realistic Optical Flow Dataset Generation from Videos**

- 论文/Paper: http://arxiv.org/pdf/2207.11075
- 代码/Code: https://github.com/megvii-research/RealFlow

**PLMCL: Partial-Label Momentum Curriculum Learning for Multi-Label Image Classification**

- 论文/Paper: http://arxiv.org/pdf/2208.09999
- 代码/Code: None

[返回目录/back](#Contents)

<a name="GAN"></a> 

## GAN

**Ultra-high-resolution unpaired stain transformation via Kernelized Instance Normalization**

- 论文/Paper: https://arxiv.org/pdf/2208.10730v1.pdf
- 代码/Code: https://github.com/Kaminyou/URUST

**Accelerating Score-based Generative Models with Preconditioned Diffusion Sampling**

- 论文/Paper: http://arxiv.org/abs/2207.02196
- 代码/Code: https://github.com/fudan-zvg/pds

**CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer**

- 论文/Paper: http://arxiv.org/pdf/2207.04808
- 代码/Code: https://github.com/JarrentWu1031/CCPL

**Fast-Vid2Vid: Spatial-Temporal Compression for Video-to-Video Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2207.05049
- 代码/Code: https://github.com/fast-vid2vid/fast-vid2vid

**RepMix: Representation Mixing for Robust Attribution of Synthesized Images**

- 论文/Paper: http://arxiv.org/abs/2207.02063
- 代码/Code: https://github.com/tubui/image_attribution

**VecGAN: Image-to-Image Translation with Interpretable Latent Directions**

- 论文/Paper: http://arxiv.org/pdf/2207.03411
- 代码/Code: None

**Context-Consistent Semantic Image Editing with Style-Preserved Modulation**

- 论文/Paper: http://arxiv.org/pdf/2207.06252
- 代码/Code: https://github.com/wuyangluo/spmpgan

**DynaST: Dynamic Sparse Transformer for Exemplar-Guided Image Generation**

- 论文/Paper: http://arxiv.org/pdf/2207.06124
- 代码/Code: https://github.com/huage001/dynast

**Supervised Attribute Information Removal and Reconstruction for Image Manipulation**

- 论文/Paper: http://arxiv.org/pdf/2207.06555
- 代码/Code: https://github.com/nannanli999/airr

**Name: Adaptive Feature Interpolation for Low-Shot Image Generation**

- 论文/Paper: https://arxiv.org/abs/2112.02450
- 代码/Code: https://github.com/dzld00/Adaptive-Feature-Interpolation-for-Low-Shot-Image-Generation

**WaveGAN: Frequency-aware GAN for High-Fidelity Few-shot Image Generation**

- 论文/Paper: http://arxiv.org/pdf/2207.07288
- 代码/Code: Link:https://github.com/kobeshegu/ECCV2022_WaveGAN

**FakeCLR: Exploring Contrastive Learning for Solving Latent Discontinuity in Data-Efficient GANs**

- 论文/Paper: http://arxiv.org/pdf/2207.08630
- 代码/Code: https://github.com/iceli1007/FakeCLR

**Outpainting by Queries**
- 论文/Paper: https://arxiv.org/abs/2207.05312
- 代码/Code: https://github.com/Kaiseem/QueryOTR

**Single Stage Virtual Try-on via Deformable Attention Flows**

- 论文/Paper: http://arxiv.org/pdf/2207.09161
- 代码/Code: https://github.com/OFA-Sys/DAFlow

**Structure-aware Editable Morphable Model for 3D Facial Detail Animation and Manipulation**

- 论文/Paper: http://arxiv.org/pdf/2207.09019
- 代码/Code: https://github.com/gerwang/facial-detail-manipulation

**Monocular 3D Object Reconstruction with GAN Inversion**

- 论文/Paper: http://arxiv.org/pdf/2207.10061
- 代码/Code: https://github.com/junzhezhang/mesh-inversion

**Generative Multiplane Images: Making a 2D GAN 3D-Aware**

- 论文/Paper: http://arxiv.org/pdf/2207.10642
- 代码/Code: https://github.com/apple/ml-gmpi

**DeltaGAN: Towards Diverse Few-shot Image Generation with Sample-Specific Delta**

- 论文/Paper: http://arxiv.org/pdf/2207.10271
- 代码/Code: https://github.com/bcmi/deltagan-few-shot-image-generation

**Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for Editable Portrait Image Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2207.10257
- 代码/Code: https://github.com/jgkwak95/surf-gan

**SGBANet: Semantic GAN and Balanced Attention Network for Arbitrarily Oriented Scene Text Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.10256
- 代码/Code: None

**2D GANs Meet Unsupervised Single-view 3D Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2207.10183
- 代码/Code: None

**InfiniteNature-Zero: Learning Perpetual View Generation of Natural Scenes from Single Images**

- 论文/Paper: http://arxiv.org/pdf/2207.11148
- 代码/Code: None

**Auto-regressive Image Synthesis with Integrated Quantization**

- 论文/Paper: http://arxiv.org/pdf/2207.10776
- 代码/Code: None

**Compositional Human-Scene Interaction Synthesis with Semantic Control**

- 论文/Paper: http://arxiv.org/pdf/2207.12824
- 代码/Code: https://github.com/zkf1997/coins

**Generator Knows What Discriminator Should Learn in Unconditional GANs**

- 论文/Paper: http://arxiv.org/pdf/2207.13320
- 代码/Code: https://github.com/naver-ai/GGDR

**StyleLight: HDR Panorama Generation for Lighting Estimation and Editing**

- 论文/Paper: http://arxiv.org/pdf/2207.14811
- 代码/Code: https://github.com/Wanggcong/StyleLight

**Cross Attention Based Style Distribution for Controllable Person Image Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2208.00712
- 代码/Code: https://github.com/xyzhouo/casd

**SKDCGN: Source-free Knowledge Distillation of Counterfactual Generative Networks using cGANs**

- 论文/Paper: http://arxiv.org/pdf/2208.04226
- 代码/Code: https://github.com/ambekarsameer96/SKDCGN

**Hierarchical Semantic Regularization of Latent Spaces in StyleGANs**

- 论文/Paper: http://arxiv.org/pdf/2208.03764
- 代码/Code: None

**Style Your Hair: Latent Optimization for Pose-Invariant Hairstyle Transfer via Local-Style-Aware Hair Alignment**

- 论文/Paper: http://arxiv.org/pdf/2208.07765
- 代码/Code: https://github.com/taeu/style-your-hair

**Paint2Pix: Interactive Painting based Progressive Image Synthesis and Editing**

- 论文/Paper: http://arxiv.org/pdf/2208.08092
- 代码/Code: https://github.com/1jsingh/paint2pix

**Mind the Gap in Distilling StyleGANs**

- 论文/Paper: http://arxiv.org/pdf/2208.08840
- 代码/Code: https://github.com/xuguodong03/stylekd

**ModSelect: Automatic Modality Selection for Synthetic-to-Real Domain Generalization**

- 论文/Paper: http://arxiv.org/pdf/2208.09414
- 代码/Code: None

**FurryGAN: High Quality Foreground-aware Image Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2208.10422
- 代码/Code: None

**Improving GANs for Long-Tailed Data through Group Spectral Regularization**

- 论文/Paper: http://arxiv.org/pdf/2208.09932
- 代码/Code: None

**Unrestricted Black-box Adversarial Attack Using GAN with Limited Queries**

- 论文/Paper: http://arxiv.org/pdf/2208.11613
- 代码/Code: None

**3D-FM GAN: Towards 3D-Controllable Face Manipulation**

- 论文/Paper: http://arxiv.org/pdf/2208.11257
- 代码/Code: None

**High-Fidelity Image Inpainting with GAN Inversion**

- 论文/Paper: http://arxiv.org/pdf/2208.11850
- 代码/Code: None

**Bokeh-Loss GAN: Multi-Stage Adversarial Training for Realistic Edge-Aware Bokeh**

- 论文/Paper: http://arxiv.org/pdf/2208.12343
- 代码/Code: None

**Exploring Gradient-based Multi-directional Controls in GANs**

- 论文/Paper: http://arxiv.org/pdf/2209.00698
- 代码/Code: None

**Studying Bias in GANs through the Lens of Race**

- 论文/Paper: http://arxiv.org/pdf/2209.02836
- 代码/Code: None

**Improved Masked Image Generation with Token-Critic**

- 论文/Paper: http://arxiv.org/pdf/2209.04439
- 代码/Code: None

**Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation**

- 论文/Paper: http://arxiv.org/pdf/2209.05968
- 代码/Code: None

[返回目录/back](#Contents)

<a name="NeRF"></a> 

## NeRF

**Streamable Neural Fields**

- 论文/Paper: http://arxiv.org/pdf/2207.09663
- 代码/Code: https://github.com/jwcho5576/streamable_nf

**Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for Editable Portrait Image Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2207.10257
- 代码/Code: https://github.com/jgkwak95/surf-gan

**AdaNeRF: Adaptive Sampling for Real-time Rendering of Neural Radiance Fields**

- 论文/Paper: http://arxiv.org/pdf/2207.10312
- 代码/Code: https://github.com/thomasneff/AdaNeRF

**PS-NeRF: Neural Inverse Rendering for Multi-view Photometric Stereo**

- 论文/Paper: http://arxiv.org/pdf/2207.11406
- 代码/Code: None

**Neural-Sim: Learning to Generate Training Data with NeRF**

- 论文/Paper: http://arxiv.org/pdf/2207.11368
- 代码/Code: https://github.com/gyhandy/neural-sim-nerf

**Neural Density-Distance Fields**

- 论文/Paper: http://arxiv.org/pdf/2207.14455
- 代码/Code: https://github.com/ueda0319/neddf

**HDR-Plenoxels: Self-Calibrating High Dynamic Range Radiance Fields**

- 论文/Paper: http://arxiv.org/pdf/2208.06787
- 代码/Code: None

[返回目录/back](#Contents)

<a name="VisualTransformer"></a> 

## Visual Transformer

**k-means Mask Transformer**

- 论文/Paper: http://arxiv.org/pdf/2207.04044
- 代码/Code: https://github.com/google-research/deeplab2

**Weakly Supervised Grounding for VQA in Vision-Language Transformers**

- 论文/Paper: http://arxiv.org/pdf/2207.02334
- 代码/Code: https://github.com/aurooj/wsg-vqa-vltransformers

**Wave-ViT: Unifying Wavelet and Transformers for Visual Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.04978
- 代码/Code: https://github.com/YehLi/ImageNetModel

**CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.04410
- 代码/Code: https://github.com/Green-Wood/CoMER

**Towards Hard-Positive Query Mining for DETR-based Human-Object Interaction Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.05293
- 代码/Code: https://github.com/MuchHair/HQM

**Hunting Group Clues with Transformers for Social Group Activity Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.05254
- 代码/Code: None

**Entry-Flipped Transformer for Inference and Prediction of Participant Behavior**

- 论文/Paper: http://arxiv.org/pdf/2207.06235
- 代码/Code: None

**DynaST: Dynamic Sparse Transformer for Exemplar-Guided Image Generation**

- 论文/Paper: http://arxiv.org/pdf/2207.06124
- 代码/Code: https://github.com/huage001/dynast

**Global-local Motion Transformer for Unsupervised Skeleton-based Action Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.06101
- 代码/Code: https://github.com/boeun-kim/gl-transformer

**TokenMix: Rethinking Image Mixing for Data Augmentation in Vision Transformers**

- 论文/Paper: http://arxiv.org/pdf/2207.08409
- 代码/Code: https://github.com/Sense-X/TokenMix

**TS2-Net: Token Shift and Selection Transformer for Text-Video Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2207.07852
- 代码/Code: None

**Action Quality Assessment with Temporal Parsing Transformer**

- 论文/Paper: http://arxiv.org/pdf/2207.09270
- 代码/Code: None

**GRIT: Faster and Better Image captioning Transformer Using Dual Visual Features**

- 论文/Paper: http://arxiv.org/pdf/2207.09666
- 代码/Code: https://github.com/davidnvq/grit

**Hierarchically Self-Supervised Transformer for Human Skeleton Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.09644
- 代码/Code: None

**AiATrack: Attention in Attention for Transformer Visual Tracking**

- 论文/Paper: http://arxiv.org/pdf/2207.09603
- 代码/Code: https://github.com/Little-Podi/AiATrack

**Single Frame Atmospheric Turbulence Mitigation: A Benchmark Study and A New Physics-Inspired Transformer Model**

- 论文/Paper: http://arxiv.org/pdf/2207.10040
- 代码/Code: None

**TinyViT: Fast Pretraining Distillation for Small Vision Transformers**

- 论文/Paper: http://arxiv.org/pdf/2207.10666
- 代码/Code: https://github.com/microsoft/cream

**An Efficient Spatio-Temporal Pyramid Transformer for Action Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.10448
- 代码/Code: None

**Weakly Supervised Object Localization via Transformer with Implicit Spatial Calibration**

- 论文/Paper: http://arxiv.org/pdf/2207.10447
- 代码/Code: https://github.com/164140757/scm

**SeedFormer: Patch Seeds based Point Cloud Completion with Upsample Transformer**

- 论文/Paper: http://arxiv.org/pdf/2207.10315
- 代码/Code: https://github.com/hrzhou2/seedformer

**Cost Aggregation with 4D Convolutional Swin Transformer for Few-Shot Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.10866
- 代码/Code: None

**IGFormer: Interaction Graph Transformer for Skeleton-based Human Interaction Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.12100
- 代码/Code: None

**3D Siamese Transformer Network for Single Object Tracking on Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2207.11995
- 代码/Code: None

**Reference-based Image Super-Resolution with Deformable Attention Transformer**

- 论文/Paper: http://arxiv.org/pdf/2207.11938
- 代码/Code: None

**SiRi: A Simple Selective Retraining Mechanism for Transformer-based Visual Grounding**

- 论文/Paper: http://arxiv.org/pdf/2207.13325
- 代码/Code: None

**Online Continual Learning with Contrastive Vision Transformer**

- 论文/Paper: http://arxiv.org/pdf/2207.13516
- 代码/Code: None

**Cross-Attention of Disentangled Modalities for 3D Human Mesh Recovery with Transformers**

- 论文/Paper: http://arxiv.org/pdf/2207.13820
- 代码/Code: https://github.com/postech-ami/FastMETRO

**Toward Understanding WordArt: Corner-Guided Transformer for Scene Text Recognition**

- 论文/Paper: http://arxiv.org/pdf/2208.00438
- 代码/Code: https://github.com/xdxie/WordArt

**TransMatting: Enhancing Transparent Objects Matting with Transformers**

- 论文/Paper: http://arxiv.org/pdf/2208.03007
- 代码/Code: https://github.com/AceCHQ/TransMatting

**Ghost-free High Dynamic Range Imaging with Context-aware Transformer**

- 论文/Paper: http://arxiv.org/pdf/2208.05114
- 代码/Code: https://github.com/megvii-research/hdr-transformer



[返回目录/back](#Contents)

<a name="Multimodal"></a> 

## 多模态 / Multimodal

**Audio-Visual Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.05042
- 代码/Code: https://github.com/OpenNLPLab/AVSBench

**Cross-modal Prototype Driven Network for Radiology Report Generation**

- 论文/Paper: http://arxiv.org/pdf/2207.04818
- 代码/Code: None

**Hierarchical Latent Structure for Multi-Modal Vehicle Trajectory Forecasting**

- 论文/Paper: http://arxiv.org/pdf/2207.04624
- 代码/Code: https://github.com/d1024choi/HLSTrajForecast

**UniNet: Unified Architecture Search with Convolution, Transformer, and MLP**

- 论文/Paper: http://arxiv.org/pdf/2207.05420
- 代码/Code: https://github.com/Sense-X/UniNet

**Video Graph Transformer for Video Question Answering**

- 论文/Paper: http://arxiv.org/pdf/2207.05342
- 代码/Code: https://github.com/sail-sg/VGT

**Bootstrapped Masked Autoencoders for Vision BERT Pretraining**

- 论文/Paper: http://arxiv.org/pdf/2207.07116
- 代码/Code: https://github.com/lightdxy/bootmae

**Learning Mutual Modulation for Self-Supervised Cross-Modal Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2207.09156
- 代码/Code: None

**Exploiting Unlabeled Data with Vision and Language Models for Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.08954
- 代码/Code: https://github.com/xiaofeng94/VL-PLM

**LocVTP: Video-Text Pre-training for Temporal Localization**

- 论文/Paper: http://arxiv.org/pdf/2207.10362
- 代码/Code: https://github.com/mengcaopku/locvtp

**Inductive and Transductive Few-Shot Video Classification via Appearance and Temporal Alignments**

- 论文/Paper: http://arxiv.org/pdf/2207.10785
- 代码/Code: https://github.com/VinAIResearch/fsvc-ata

**Cross-Modal 3D Shape Generation and Manipulation**

- 论文/Paper: http://arxiv.org/pdf/2207.11795
- 代码/Code: None

**Learning Visual Representation from Modality-Shared Contrastive Language-Image Pre-training**

- 论文/Paper: http://arxiv.org/pdf/2207.12661
- 代码/Code: https://github.com/hxyou/msclip

**Frozen CLIP Models are Efficient Video Learners**

- 论文/Paper: http://arxiv.org/pdf/2208.03550
- 代码/Code: https://github.com/OpenGVLab/efficient-video-recognition

**Consistency-based Self-supervised Learning for Temporal Anomaly Localization**

- 论文/Paper: http://arxiv.org/pdf/2208.05251
- 代码/Code: None

**Motion Sensitive Contrastive Learning for Self-supervised Video Representation**

- 论文/Paper: http://arxiv.org/pdf/2208.06105
- 代码/Code: None

**TL;DW? Summarizing Instructional Videos with Task Relevance & Cross-Modal Saliency**

- 论文/Paper: http://arxiv.org/pdf/2208.06773
- 代码/Code: None

**See Finer, See More: Implicit Modality Alignment for Text-based Person Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2208.08608
- 代码/Code: https://github.com/TencentYoutuResearch/PersonRetrieval-IVT.

**Learning an Efficient Multimodal Depth Completion Model**

- 论文/Paper: http://arxiv.org/pdf/2208.10771
- 代码/Code: https://github.com/dwhou/emdc-pytorch

**Learning from Unlabeled 3D Environments for Vision-and-Language Navigation**

- 论文/Paper: http://arxiv.org/pdf/2208.11781
- 代码/Code: None

**CMD: Self-supervised 3D Action Representation Learning with Cross-modal Mutual Distillation**

- 论文/Paper: http://arxiv.org/pdf/2208.12448
- 代码/Code: https://github.com/maoyunyao/cmd

**StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation**

- 论文/Paper: http://arxiv.org/pdf/2209.06192
- 代码/Code: https://github.com/adymaharana/storydalle

**MUST-VQA: MUltilingual Scene-text VQA**

- 论文/Paper: http://arxiv.org/pdf/2209.06730
- 代码/Code: None

[返回目录/back](#Contents)

<a name="Vision-Language"></a> 

## Vision-Language

**Vision-Language Adaptive Mutual Decoder for OOV-STR**

- 论文/Paper: http://arxiv.org/pdf/2209.00859
- 代码/Code: None



[返回目录/back](#Contents)

<a name="DomainAdaptation"></a> 

## Domain Adaptation

**Concurrent Subsidiary Supervision for Unsupervised Source-Free Domain Adaptation**

- 论文/Paper: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900177.pdf
- 代码/Code: https://github.com/val-iisc/StickerDA

[返回目录/back](#Contents)

<a name="ContrastiveLearning"></a> 

## 对比学习/Contrastive Learning

**Network Binarization via Contrastive Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.02970
- 代码/Code: None

**Contrastive Deep Supervision**

- 论文/Paper: http://arxiv.org/pdf/2207.05306
- 代码/Code: None

**ConCL: Concept Contrastive Learning for Dense Prediction Pre-training in Pathology Images**

- 论文/Paper: http://arxiv.org/pdf/2207.06733
- 代码/Code: https://github.com/tencentailabhealthcare/concl

**Action-based Contrastive Learning for Trajectory Prediction**

- 论文/Paper: http://arxiv.org/pdf/2207.08664
- 代码/Code: None

**FakeCLR: Exploring Contrastive Learning for Solving Latent Discontinuity in Data-Efficient GANs**

- 论文/Paper: http://arxiv.org/pdf/2207.08630
- 代码/Code: https://github.com/iceli1007/FakeCLR.

**Adversarial Contrastive Learning via Asymmetric InfoNCE**

- 论文/Paper: http://arxiv.org/pdf/2207.08374
- 代码/Code: https://github.com/yqy2001/A-InfoNCE

**Fast-MoCo: Boost Momentum-based Contrastive Learning with Combinatorial Patches**

- 论文/Paper: http://arxiv.org/pdf/2207.08220
- 代码/Code: None

**Decoupled Adversarial Contrastive Learning for Self-supervised Adversarial Robustness**

- 论文/Paper: http://arxiv.org/pdf/2207.10899
- 代码/Code: https://github.com/pantheon5100/DeACL.

**Bi-directional Contrastive Learning for Domain Adaptive Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.10892
- 代码/Code: None

**Patient-level Microsatellite Stability Assessment from Whole Slide Images By Combining Momentum Contrast Learning and Group Patch Embeddings**

- 论文/Paper: http://arxiv.org/pdf/2208.10429
- 代码/Code: https://github.com/technioncomputationalmrilab/colorectal_cancer_ai

**FairDisCo: Fairer AI in Dermatology via Disentanglement Contrastive Learning**

- 论文/Paper: http://arxiv.org/pdf/2208.10013
- 代码/Code: https://github.com/siyi-wind/FairDisCo

**CODER: Coupled Diversity-Sensitive Momentum Contrastive Learning for Image-Text Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2208.09843
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ObjectDetection"></a> 

## 目标检测/Object Detection

**Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.02541
- 代码/Code: None

**Should All Proposals be Treated Equally in Object Detection?**

- 论文/Paper: http://arxiv.org/pdf/2207.03520
- 代码/Code: None

**HEAD: HEtero-Assists Distillation for Heterogeneous Object Detectors**

- 论文/Paper: http://arxiv.org/pdf/2207.05345
- 代码/Code: https://github.com/LutingWang/HEAD

**Adversarially-Aware Robust Object Detector**

- 论文/Paper: http://arxiv.org/pdf/2207.06202
- 代码/Code: https://github.com/7eu7d7/robustdet

**ObjectBox: From Centers to Boxes for Anchor-Free Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.06985
- 代码/Code: https://github.com/mohsenzand/objectbox

**Point-to-Box Network for Accurate Object Detection via Single Point Supervision**

- 论文/Paper: http://arxiv.org/pdf/2207.06827
- 代码/Code: None

**DID-M3D: Decoupling Instance Depth for Monocular 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.08531
- 代码/Code: https://github.com/SPengLiang/DID-M3D.

**SPSN: Superpixel Prototype Sampling Network for RGB-D Salient Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.07898
- 代码/Code: https://github.com/Hydragon516/SPSN

**Rethinking IoU-based Optimization for Single-stage 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.09332
- 代码/Code: https://github.com/hlsheng1/RDIoU

**Densely Constrained Depth Estimator for Monocular 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.10047
- 代码/Code: https://github.com/bravegroup/dcd

**Robust Object Detection With Inaccurate Bounding Boxes**

- 论文/Paper: http://arxiv.org/pdf/2207.09697
- 代码/Code: https://github.com/cxliu0/OA-MIL

**Unsupervised Domain Adaptation for One-stage Object Detector using Offsets to Bounding Box**

- 论文/Paper: http://arxiv.org/pdf/2207.09656
- 代码/Code: None

**AutoAlignV2: Deformable Feature Aggregation for Dynamic Multi-Modal 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.10316
- 代码/Code: https://github.com/zehuichen123/autoalignv2

**Rethinking Few-Shot Object Detection on a Multi-Domain Benchmark**

- 论文/Paper: http://arxiv.org/pdf/2207.11169
- 代码/Code: https://github.com/amazon-research/few-shot-object-detection-benchmark.

**DEVIANT: Depth EquiVarIAnt NeTwork for Monocular 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.10758
- 代码/Code: https://github.com/abhi1kumar/DEVIANT

**Active Learning Strategies for Weakly-supervised Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.12112
- 代码/Code: https://github.com/huyvvo/BiB.

**W2N:Switching From Weak Supervision to Noisy Supervision for Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.12104
- 代码/Code: https://github.com/1170300714/w2n_wsod.

**Salient Object Detection for Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2207.11889
- 代码/Code: None

**UC-OWOD: Unknown-Classified Open World Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.11455
- 代码/Code: https://github.com/JohnWuzh/UC-OWOD

**Monocular 3D Object Detection with Depth from Motion**

- 论文/Paper: http://arxiv.org/pdf/2207.12988
- 代码/Code: https://github.com/tai-wang/depth-from-motion

**Exploring Resolution and Degradation Clues as Self-supervised Signal for Low Quality Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2208.03062
- 代码/Code: https://github.com/cuiziteng/ECCV_AERIS

**Graph R-CNN: Towards Accurate 3D Object Detection with Semantic-Decorated Local Graph**

- 论文/Paper: http://arxiv.org/pdf/2208.03624
- 代码/Code: https://github.com/Nightmare-n/GraphRCNN

**Object Discovery via Contrastive Learning for Weakly Supervised Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2208.07576
- 代码/Code: https://github.com/jinhseo/od-wscl

**RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2208.08738
- 代码/Code: https://github.com/chasel-tsui/mmdet-rfla

**Object Detection in Aerial Images with Uncertainty-Aware Graph Network**

- 论文/Paper: http://arxiv.org/pdf/2208.10781
- 代码/Code: None

**Adversarial Vulnerability of Temporal Feature Networks for Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2208.10773
- 代码/Code: None

**Identifying Out-of-Distribution Samples in Real-Time for Safety-Critical 2D Object Detection with Margin Entropy Loss**

- 论文/Paper: http://arxiv.org/pdf/2209.00364
- 代码/Code: None

**CenterFormer: Center-based Transformer for 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2209.05588
- 代码/Code: https://github.com/tusimple/centerformer

[返回目录/back](#Contents)

<a name="ObjectTracking"></a> 

## 目标跟踪/Object Tracking

**Tracking Objects as Pixel-wise Distributions**

- 论文/Paper: http://arxiv.org/pdf/2207.05518
- 代码/Code: None

**Towards Grand Unification of Object Tracking**

- 论文/Paper: http://arxiv.org/pdf/2207.07078
- 代码/Code: https://github.com/masterbin-iiau/unicorn

**The Caltech Fish Counting Dataset: A Benchmark for Multiple-Object Tracking and Counting**

- 论文/Paper: http://arxiv.org/pdf/2207.09295
- 代码/Code: None

**MOTCOM: The Multi-Object Tracking Dataset Complexity Metric**

- 论文/Paper: http://arxiv.org/pdf/2207.10031
- 代码/Code: None

**Robust Landmark-based Stent Tracking in X-ray Fluoroscopy**

- 论文/Paper: http://arxiv.org/pdf/2207.09933
- 代码/Code: None

**AiATrack: Attention in Attention for Transformer Visual Tracking**

- 论文/Paper: http://arxiv.org/pdf/2207.09603
- 代码/Code: https://github.com/Little-Podi/AiATrack

**3D Siamese Transformer Network for Single Object Tracking on Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2207.11995
- 代码/Code: None

**Tracking Every Thing in the Wild**

- 论文/Paper: http://arxiv.org/pdf/2207.12978
- 代码/Code: None

**AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing**

- 论文/Paper: http://arxiv.org/pdf/2207.13784
- 代码/Code: https://github.com/eth-siplab/AvatarPoser

**Robust Multi-Object Tracking by Marginal Inference**

- 论文/Paper: http://arxiv.org/pdf/2208.03727
- 代码/Code: None

**Towards Sequence-Level Training for Visual Tracking**

- 论文/Paper: http://arxiv.org/pdf/2208.05810
- 代码/Code: https://github.com/byminji/SLTtrack

[返回目录/back](#Contents)

<a name="Segmentation"></a> 

## 语义分割/Segmentation

**Domain Adaptive Video Segmentation via Temporal Pseudo Supervision**

- 论文/Paper: http://arxiv.org/pdf/2207.02372
- 代码/Code: https://github.com/xing0047/tps

**OSFormer: One-Stage Camouflaged Instance Segmentation with Transformers**

- 论文/Paper: http://arxiv.org/pdf/2207.02255
- 代码/Code: https://github.com/pjlallen/osformer

**PseudoClick: Interactive Image Segmentation with Click Imitation**

- 论文/Paper: http://arxiv.org/pdf/2207.05282
- 代码/Code: None

**XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model**

- 论文/Paper: http://arxiv.org/pdf/2207.07115
- 代码/Code: https://github.com/hkchengrex/XMem

**Tackling Background Distraction in Video Object Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.06953
- 代码/Code: https://github.com/suhwan-cho/tbd

**Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.08549
- 代码/Code: None

**Hierarchical Feature Alignment Network for Unsupervised Video Object Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.08485
- 代码/Code: https://github.com/NUST-Machine-Intelligence-Laboratory/HFAN

**Open-world Semantic Segmentation via Contrasting and Clustering Vision-Language Embedding**

- 论文/Paper: http://arxiv.org/pdf/2207.08455
- 代码/Code: None

**Learning Quality-aware Dynamic Memory for Video Object Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.07922
- 代码/Code: https://github.com/workforai/QDMN

**Box-supervised Instance Segmentation with Level Set Evolution**

- 论文/Paper: http://arxiv.org/pdf/2207.09055
- 代码/Code: https://github.com/LiWentomng/boxlevelset

**ML-BPM: Multi-teacher Learning with Bidirectional Photometric Mixing for Open Compound Domain Adaptation in Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.09045
- 代码/Code: None

**Self-Supervised Interactive Object Segmentation Through a Singulation-and-Grasping Approach**

- 论文/Paper: http://arxiv.org/pdf/2207.09314
- 代码/Code: None

**DecoupleNet: Decoupled Network for Domain Adaptive Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.09988
- 代码/Code: https://github.com/dvlab-research/decouplenet

**CoSMix: Compositional Semantic Mix for Domain Adaptation in 3D LiDAR Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.09778
- 代码/Code: https://github.com/saltoricristiano/cosmix-uda

**GIPSO: Geometrically Informed Propagation for Online Adaptation in 3D LiDAR Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.09763
- 代码/Code: https://github.com/saltoricristiano/gipso-sfouda

**Online Domain Adaptation for Semantic Segmentation in Ever-Changing Conditions**

- 论文/Paper: http://arxiv.org/pdf/2207.10667
- 代码/Code: https://github.com/theo2021/onda

**In Defense of Online Models for Video Instance Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.10661
- 代码/Code: https://github.com/wjf5203/vnext

**Mining Relations among Cross-Frame Affinities for Video Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.10436
- 代码/Code: https://github.com/guoleisun/vss-mrcfa

**Long-tailed Instance Segmentation using Gumbel Optimized Loss**

- 论文/Paper: http://arxiv.org/pdf/2207.10936
- 代码/Code: https://github.com/kostas1515/GOL

**Bi-directional Contrastive Learning for Domain Adaptive Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.10892
- 代码/Code: None

**Cost Aggregation with 4D Convolutional Swin Transformer for Few-Shot Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.10866
- 代码/Code: None

**Self-Support Few-Shot Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.11549
- 代码/Code: https://github.com/fanq15/SSP

**Active Pointly-Supervised Instance Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.11493
- 代码/Code: None

**Video Mask Transfiner for High-Quality Video Instance Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.14012
- 代码/Code: None

**Doubly Deformable Aggregation of Covariance Matrices for Few-shot Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2208.00306
- 代码/Code: None

**Per-Clip Video Object Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2208.01924
- 代码/Code: https://github.com/pkyong95/PCVOS

**Cluster-to-adapt: Few Shot Domain Adaptation for Semantic Segmentation across Disjoint Labels**

- 论文/Paper: http://arxiv.org/pdf/2208.02804
- 代码/Code: None

**Generalizable Medical Image Segmentation via Random Amplitude Mixup and Domain-Specific Image Restoration**

- 论文/Paper: http://arxiv.org/pdf/2208.03901
- 代码/Code: None

**Fine-Grained Egocentric Hand-Object Segmentation: Dataset, Model, and Applications**

- 论文/Paper: http://arxiv.org/pdf/2208.03826
- 代码/Code: https://github.com/owenzlz/EgoHOS

**Multi-Granularity Distillation Scheme Towards Lightweight Semi-Supervised Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2208.10169
- 代码/Code: https://github.com/jayqine/mgd-ssss

**Occlusion-Aware Instance Segmentation via BiLayer Network Architectures**

- 论文/Paper: http://arxiv.org/pdf/2208.04438
- 代码/Code: https://github.com/lkeab/BCNet

[返回目录/back](#Contents)

<a name="VS"></a> 

## Video Segmentation

**Video Mask Transfiner for High-Quality Video Instance Segmentation**

- 论文/Paper: https://arxiv.org/abs/2207.14012
- 代码/Code: https://github.com/SysCV/vmt

[返回目录/back](#Contents)

<a name="MIS"></a> 

## 医学图像分割/Medical Image Segmentation

**Personalizing Federated Medical Image Segmentation via Local Calibration**

- 论文/Paper: http://arxiv.org/pdf/2207.04655
- 代码/Code: https://github.com/jcwang123/FedLC

**Learning Topological Interactions for Multi-Class Medical Image Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.09654
- 代码/Code: https://github.com/topoxlab/topointeraction

**qDWI-Morph: Motion-compensated quantitative Diffusion-Weighted MRI analysis for fetal lung maturity assessment**

- 论文/Paper: http://arxiv.org/pdf/2208.09836
- 代码/Code: https://github.com/TechnionComputationalMRILab/qDWI-Morph.

**Self-Supervised Pretraining for 2D Medical Image Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2209.00314
- 代码/Code: None

[返回目录/back](#Contents)

<a name="KnowledgeDistillation"></a> 

## Knowledge Distillation

**Knowledge Condensation Distillation**

- 论文/Paper: http://arxiv.org/pdf/2207.05409
- 代码/Code: https://github.com/dzy3/KCD

**FedX: Unsupervised Federated Learning with Cross Knowledge Distillation**

- 论文/Paper: http://arxiv.org/pdf/2207.09158
- 代码/Code: https://github.com/sungwon-han/fedx

[返回目录/back](#Contents)

<a name="ActionDetection"></a> 

## Action Detection

**ReAct: Temporal Action Detection with Relational Queries**

- 论文/Paper: http://arxiv.org/pdf/2207.07097
- 代码/Code: https://github.com/sssste/react

**Semi-Supervised Temporal Action Detection with Proposal-Free Masking**

- 论文/Paper: http://arxiv.org/pdf/2207.07059
- 代码/Code: https://github.com/sauradip/SPOT

**Temporal Action Detection with Global Segmentation Mask Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.06580
- 代码/Code: https://github.com/sauradip/TAGS

**Weakly-Supervised Temporal Action Detection for Fine-Grained Videos with Hierarchical Atomic Actions**

- 论文/Paper: http://arxiv.org/pdf/2207.11805
- 代码/Code: None

**HaloAE: An HaloNet based Local Transformer Auto-Encoder for Anomaly Detection and Localization**

- 论文/Paper: http://arxiv.org/pdf/2208.03486
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ActionRecognition"></a> 

## Action Recognition

**Compound Prototype Matching for Few-shot Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.05515
- 代码/Code: None

**Collaborating Domain-shared and Target-specific Feature Clustering for Cross-domain 3D Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.09767
- 代码/Code: https://github.com/canbaoburen/CoDT

**Combined CNN Transformer Encoder for Enhanced Fine-grained Human Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2208.01897
- 代码/Code: None

**PSUMNet: Unified Modality Part Streams are All You Need for Efficient Pose-based Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2208.05775
- 代码/Code: https://github.com/skelemoa/psumnet

**Lane Change Classification and Prediction with Action Recognition Networks**

- 论文/Paper: http://arxiv.org/pdf/2208.11650
- 代码/Code: None

**Dynamic Spatio-Temporal Specialization Learning for Fine-Grained Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2209.01425
- 代码/Code: None

[返回目录/back](#Contents)

<a name="AnomalyDetection"></a> 

## Anomaly Detection

**Registration based Few-Shot Anomaly Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.07361
- 代码/Code: https://github.com/MediaBrain-SJTU/RegAD

**Look at Adjacent Frames: Video Anomaly Detection without Offline Training**

- 论文/Paper: http://arxiv.org/pdf/2207.13798
- 代码/Code: None

**Towards Open Set Video Anomaly Detection**

- 论文/Paper: http://arxiv.org/pdf/2208.11113
- 代码/Code: None

[返回目录/back](#Contents)

<a name="FaceRecognition"></a> 

## 人脸识别/Face Recognition

**Controllable and Guided Face Synthesis for Unconstrained Face Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.10180
- 代码/Code: None

**Towards Robust Face Recognition with Comprehensive Search**

- 论文/Paper: http://arxiv.org/pdf/2208.13600
- 代码/Code: None

[返回目录/back](#Contents)

<a name="HumanPoseEstimation"></a> 

## 人体姿态估计/Human Pose Estimation

**Self-Constrained Inference Optimization on Structural Groups for Human Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2207.02425
- 代码/Code: None

**Category-Level 6D Object Pose and Size Estimation using Self-Supervised Deep Prior Deformation Networks**

- 论文/Paper: http://arxiv.org/pdf/2207.05444
- 代码/Code: https://github.com/JiehongLin/Self-DPDN

**Global-local Motion Transformer for Unsupervised Skeleton-based Action Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.06101
- 代码/Code: https://github.com/boeun-kim/gl-transformer

**TransGrasp: Grasp Pose Estimation of a Category of Objects by Transferring Grasps from Only One Labeled Instance**

- 论文/Paper: http://arxiv.org/pdf/2207.07861
- 代码/Code: https://github.com/yanjh97/TransGrasp

**Pose for Everything: Towards Category-Agnostic Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2207.10387
- 代码/Code: https://github.com/luminxu/Pose-for-Everything

**C3P: Cross-domain Pose Prior Propagation for Weakly Supervised 3D Human Pose Estimation**

- 论文/Paper: None
- 代码/Code: https://github.com/wucunlin/C3P

**3D Interacting Hand Pose Estimation by Hand De-occlusion and Removal**

- 论文/Paper: http://arxiv.org/pdf/2207.11061
- 代码/Code: https://github.com/MengHao666/HDR.

**Faster VoxelPose: Real-time 3D Human Pose Estimation by Orthographic Projection**

- 论文/Paper: http://arxiv.org/pdf/2207.10955
- 代码/Code: None

**ShAPO: Implicit Representations for Multi-Object Shape, Appearance, and Pose Optimization**

- 论文/Paper: http://arxiv.org/pdf/2207.13691
- 代码/Code: None

**RBP-Pose: Residual Bounding Box Projection for Category-Level Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2208.00237
- 代码/Code: None

**Neural Correspondence Field for Object Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2208.00113
- 代码/Code: None

**Explicit Occlusion Reasoning for Multi-person 3D Human Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2208.00090
- 代码/Code: None

**CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation**

- 论文/Paper: http://arxiv.org/pdf/2208.00571
- 代码/Code: https://github.com/huawei-noah/noah-research/tree/master/CLIFF

**PoseTrans: A Simple Yet Effective Pose Transformation Augmentation for Human Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2208.07755
- 代码/Code: None

**Towards Unbiased Label Distribution Learning for Facial Pose Estimation Using Anisotropic Spherical Gaussian**

- 论文/Paper: http://arxiv.org/pdf/2208.09122
- 代码/Code: None

**Learning Visibility for Robust Dense Human Body Estimation**

- 论文/Paper: http://arxiv.org/pdf/2208.10652
- 代码/Code: https://github.com/chhankyao/visdb

[返回目录/back](#Contents)

<a name="FaceAnti-Spoofing"></a> 

## 人脸活体检测/Face Anti-Spoofing

**Generative Domain Adaptation for Face Anti-Spoofing**

- 论文/Paper: http://arxiv.org/pdf/2207.10015
- 代码/Code: None

**Multi-domain Learning for Updating Face Anti-spoofing Models**

- 论文/Paper: http://arxiv.org/pdf/2208.11148
- 代码/Code: None



[返回目录/back](#Contents)

<a name="FacialAttributeRecognition"></a> 

## 人脸属性识别/Facial Attribute Recognition

**FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification**

- 论文/Paper: http://arxiv.org/pdf/2207.10888
- 代码/Code: https://github.com/Bernardo1998/FairGRAPE

[返回目录/back](#Contents)

<a name="Face"></a> 

## 人脸相关 / Face

**On Mitigating Hard Clusters for Face Clustering**

- 论文/Paper: http://arxiv.org/pdf/2207.11895
- 代码/Code: https://github.com/echoanran/On-Mitigating-Hard-Clusters.

**Learning Dynamic Facial Radiance Fields for Few-Shot Talking Head Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2207.11770
- 代码/Code: None

**Perspective Reconstruction of Human Faces by Joint Mesh and Landmark Regression**

- 论文/Paper: http://arxiv.org/pdf/2208.07142
- 代码/Code: None

[返回目录/back](#Contents)

<a name="3DReconstruction"></a> 

## 3D reconstruction

**Latent Partition Implicit with Surface Codes for 3D Representation**

- 论文/Paper: https://arxiv.org/abs/2207.08631
- 代码/Code: https://github.com/chenchao15/LPI

**LWA-HAND: Lightweight Attention Hand for Interacting Hand Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2208.09815
- 代码/Code: None

**SimpleRecon: 3D Reconstruction Without 3D Convolutions**

- 论文/Paper: http://arxiv.org/pdf/2208.14743
- 代码/Code: None



[返回目录/back](#Contents)

<a name="HumanReconstruction"></a> 

## Human Reconstruction

**3D Clothed Human Reconstruction in the Wild**

- 论文/Paper: http://arxiv.org/pdf/2207.10053
- 代码/Code: https://github.com/hygenie1228/clothwild_release

**UNIF: United Neural Implicit Functions for Clothed Human Reconstruction and Animation**

- 论文/Paper: http://arxiv.org/pdf/2207.09835
- 代码/Code: https://github.com/ShenhanQian/UNIF

**The One Where They Reconstructed 3D Humans and Environments in TV Shows**

- 论文/Paper: http://arxiv.org/pdf/2207.14279
- 代码/Code: None

**BCom-Net: Coarse-to-Fine 3D Textured Body Shape Completion Network**

- 论文/Paper: http://arxiv.org/pdf/2208.08768
- 代码/Code: None

**Neural Capture of Animatable 3D Human from Monocular Video**

- 论文/Paper: http://arxiv.org/pdf/2208.08728
- 代码/Code: None

[返回目录/back](#Contents)

<a name="Relighting"></a> 

## Relighting

**Geometry-aware Single-image Full-body Human Relighting**

- 论文/Paper: http://arxiv.org/pdf/2207.04750
- 代码/Code: None

**Relighting4D: Neural Relightable Human from Videos**

- 论文/Paper: http://arxiv.org/pdf/2207.07104
- 代码/Code: https://github.com/FrozenBurning/Relighting4D

[返回目录/back](#Contents)

<a name="DeepFake"></a> 

## DeepFake

**Detecting and Recovering Sequential DeepFake Manipulation**

- 论文/Paper: http://arxiv.org/abs/2207.02204
- 代码/Code: https://github.com/rshaojimmy/seqdeepfake

**An Efficient Method for Face Quality Assessment on the Edge**

- 论文/Paper: http://arxiv.org/pdf/2207.09505
- 代码/Code: None

[返回目录/back](#Contents)

<a name="OCR"></a>

## OCR

**Character decomposition to resolve class imbalance problem in Hangul OCR**

- 论文/Paper: http://arxiv.org/pdf/2208.06079
- 代码/Code: None

**Shift Variance in Scene Text Detection**

- 论文/Paper: http://arxiv.org/pdf/2208.09231
- 代码/Code: None

**1st Place Solution to ECCV 2022 Challenge on Out of Vocabulary Scene Text Understanding: End-to-End Recognition of Out of Vocabulary Words**

- 论文/Paper: http://arxiv.org/pdf/2209.00224
- 代码/Code: None

**Levenshtein OCR**

- 论文/Paper: http://arxiv.org/pdf/2209.03594
- 代码/Code: None

[返回目录/back](#Contents)

<a name="TextRecognition"></a>

## Text Recognition

**Scene Text Recognition with Permuted Autoregressive Sequence Models**

- 论文/Paper: http://arxiv.org/pdf/2207.06966
- 代码/Code: https://github.com/baudm/parseq

**Dynamic Low-Resolution Distillation for Cost-Efficient End-to-End Text Spotting**

- 论文/Paper: http://arxiv.org/pdf/2207.06694
- 代码/Code: https://github.com/hikopensource/davar-lab-ocr

**Contextual Text Block Detection towards Scene Text Understanding**

- 论文/Paper: http://arxiv.org/pdf/2207.12955
- 代码/Code: None

**GLASS: Global to Local Attention for Scene-Text Spotting**

- 论文/Paper: http://arxiv.org/pdf/2208.03364
- 代码/Code: None

**Multi-Granularity Prediction for Scene Text Recognition**

- 论文/Paper: http://arxiv.org/pdf/2209.03592
- 代码/Code: None

[返回目录/back](#Contents)

<a name="PointCloud"></a>

## 点云/Point Cloud

**Open-world Semantic Segmentation for LIDAR Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2207.01452
- 代码/Code: https://github.com/jun-cen/open_world_3d_semantic_segmentation

**2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2207.04397
- 代码/Code: None

**CPO: Change Robust Panorama to Point Cloud Localization**

- 论文/Paper: http://arxiv.org/pdf/2207.05317
- 代码/Code: None

**diffConv: Analyzing Irregular Point Clouds with an Irregular View**

- 论文/Paper: https://arxiv.org/abs/2111.14658
- 代码/Code: https://github.com/mmmmimic/diffConvNet

**CATRE: Iterative Point Clouds Alignment for Category-level Object Pose Refinement**

- 论文/Paper: http://arxiv.org/pdf/2207.08082
- 代码/Code: None

**Dual Adaptive Transformations for Weakly Supervised Point Cloud Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2207.09084
- 代码/Code: None

**SeedFormer: Patch Seeds based Point Cloud Completion with Upsample Transformer**

- 论文/Paper: http://arxiv.org/pdf/2207.10315
- 代码/Code: https://github.com/hrzhou2/seedformer

**Dynamic 3D Scene Analysis by Point Cloud Accumulation**

- 论文/Paper: http://arxiv.org/pdf/2207.12394
- 代码/Code: None

**3D Siamese Transformer Network for Single Object Tracking on Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2207.11995
- 代码/Code: None

**Salient Object Detection for Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2207.11889
- 代码/Code: None

**MonteBoxFinder: Detecting and Filtering Primitives to Fit a Noisy Point Cloud**

- 论文/Paper: http://arxiv.org/pdf/2207.14268
- 代码/Code: https://github.com/MichaelRamamonjisoa/MonteBoxFinder

**Improving RGB-D Point Cloud Registration by Learning Multi-scale Local Linear Transformation**

- 论文/Paper: http://arxiv.org/pdf/2208.14893
- 代码/Code: https://github.com/514dna/llt

**Learning to Generate Realistic LiDAR Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2209.03954
- 代码/Code: None

[返回目录/back](#Contents)



<a name="FlowEstimation"></a>

## 光流估计/Flow Estimation

**Bi-PointFlowNet: Bidirectional Learning for Point Cloud Based Scene Flow Estimation**

- 论文/Paper: http://arxiv.org/pdf/2207.07522
- 代码/Code: https://github.com/cwc1260/BiFlow

**What Matters for 3D Scene Flow Network**

- 论文/Paper: http://arxiv.org/pdf/2207.09143
- 代码/Code: https://github.com/IRMVLab/3DFlow

**Deep 360$^\circ$ Optical Flow Estimation Based on Multi-Projection Fusion**

- 论文/Paper: http://arxiv.org/pdf/2208.00776
- 代码/Code: None

[返回目录/back](#Contents)

<a name="DepthEstimation"></a>

## 深度估计/Depth Estimation

**Physical Attack on Monocular Depth Estimation with Optimal Adversarial Patches**

- 论文/Paper: http://arxiv.org/pdf/2207.04718
- 代码/Code: None

**Towards Scale-Aware, Robust, and Generalizable Unsupervised Monocular Depth Estimation by Integrating IMU Motion Dynamics**

- 论文/Paper: http://arxiv.org/pdf/2207.04680
- 代码/Code: https://github.com/SenZHANG-GitHub/ekf-imu-depth

**RA-Depth: Resolution Adaptive Self-Supervised Monocular Depth Estimation**

- 论文/Paper: http://arxiv.org/pdf/2207.11984
- 代码/Code: None

**Self-distilled Feature Aggregation for Self-supervised Monocular Depth Estimation**

- 论文/Paper: http://arxiv.org/pdf/2209.07088
- 代码/Code: https://github.com/ZM-Zhou/SDFA-Net_pytorch

[返回目录/back](#Contents)

<a name="LaneDetection"></a>

## 车道线检测/Lane Detection

**RCLane: Relay Chain Prediction for Lane Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.09399
- 代码/Code: None

[返回目录/back](#Contents)

<a name="TrajectoryPrediction"></a>

## 轨迹预测/Trajectory Prediction

**Action-based Contrastive Learning for Trajectory Prediction**

- 论文/Paper: http://arxiv.org/pdf/2207.08664
- 代码/Code: None

**Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction**

- 论文/Paper: http://arxiv.org/pdf/2207.09953
- 代码/Code: https://github.com/inhwanbae/gpgraph

**Aware of the History: Trajectory Forecasting with the Local Behavior Data**

- 论文/Paper: http://arxiv.org/pdf/2207.09646
- 代码/Code: None

**Human Trajectory Prediction via Neural Social Physics**

- 论文/Paper: http://arxiv.org/pdf/2207.10435
- 代码/Code: https://github.com/realcrane/human-trajectory-prediction-via-neural-social-physics

**D2-TPred: Discontinuous Dependency for Trajectory Prediction under Traffic Lights**

- 论文/Paper: http://arxiv.org/pdf/2207.10398
- 代码/Code: https://github.com/vtp-tl/d2-tpred

[返回目录/back](#Contents)

<a name="Super-Resolution"></a>

## 超分/Super-Resolution

**Image Super-Resolution with Deep Dictionary**

- 论文/Paper: http://arxiv.org/pdf/2207.09228
- 代码/Code: None

**Learning Mutual Modulation for Self-Supervised Cross-Modal Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2207.09156
- 代码/Code: None

**CADyQ: Content-Aware Dynamic Quantization for Image Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2207.10345
- 代码/Code: https://github.com/cheeun/cadyq

**Towards Interpretable Video Super-Resolution via Alternating Optimization**

- 论文/Paper: http://arxiv.org/pdf/2207.10765
- 代码/Code: None

**Reference-based Image Super-Resolution with Deformable Attention Transformer**

- 论文/Paper: http://arxiv.org/pdf/2207.11938
- 代码/Code: None

**Learning Spatiotemporal Frequency-Transformer for Compressed Video Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2208.03012
- 代码/Code: https://github.com/researchmm/FTVSR

**HST: Hierarchical Swin Transformer for Compressed Image Super-resolution**

- 论文/Paper: http://arxiv.org/pdf/2208.09885
- 代码/Code: None

**DSR: Towards Drone Image Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2208.12327
- 代码/Code: https://github.com/ivrl/dsr

[返回目录/back](#Contents)

<a name="ImageDenoising"></a>

## 图像去噪/Image Denoising

**Optimizing Image Compression via Joint Learning with Denoising**

- 论文/Paper: http://arxiv.org/pdf/2207.10869
- 代码/Code: https://github.com/felixcheng97/DenoiseCompression

[返回目录/back](#Contents)

<a name="ImageDeblurring"></a>

## 图像去模糊/Image Deblurring

**Spatio-Temporal Deformable Attention Network for Video Deblurring**

- 论文/Paper: http://arxiv.org/pdf/2207.10852
- 代码/Code: None

**Efficient Video Deblurring Guided by Motion Magnitude**

- 论文/Paper: http://arxiv.org/pdf/2207.13374
- 代码/Code: None

**Learning Degradation Representations for Image Deblurring**

- 论文/Paper: http://arxiv.org/pdf/2208.05244
- 代码/Code: https://github.com/dasongli1/learning_degradation

**Towards Real-World Video Deblurring by Exploring Blur Formation Process**

- 论文/Paper: http://arxiv.org/pdf/2208.13184
- 代码/Code: None



[返回目录/back](#Contents)

<a name="ImageRestoration"></a>

## 图像复原/Image Restoration

**D2HNet: Joint Denoising and Deblurring with Hierarchical Network for Robust Night Image Restoration**

- 论文/Paper: http://arxiv.org/pdf/2207.03294
- 代码/Code: https://github.com/zhaoyuzhi/D2HNet

[返回目录/back](#Contents)

<a name="ImageInpainting"></a> 

## 图像修复/Image Inpainting

**Flow-Guided Transformer for Video Inpainting**

- 论文/Paper: http://arxiv.org/pdf/2208.06768
- 代码/Code: https://github.com/hitachinsk/fgt

**Unbiased Multi-Modality Guidance for Image Inpainting**

- 论文/Paper: http://arxiv.org/pdf/2208.11844
- 代码/Code: https://github.com/yeates/MMT

[返回目录/back](#Contents)

<a name="ImageEnhancement"></a> 

## 图像增强/Image Enhancement

**Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression**

- 论文/Paper: http://arxiv.org/pdf/2207.10564
- 代码/Code: https://github.com/jinyeying/night-enhancement

[返回目录/back](#Contents)

<a name="VideoInterpolation"></a> 

## Video Interpolation

**Video Interpolation by Event-driven Anisotropic Adjustment of Optical Flow**

- 论文/Paper: http://arxiv.org/pdf/2208.09127
- 代码/Code: None

[返回目录/back](#Contents)

<a name="TemporalActionSegmentation"></a> 

## Temporal Action Segmentation

**Unified Fully and Timestamp Supervised Temporal Action Segmentation via Sequence to Sequence Translation**

- 论文/Paper: http://arxiv.org/pdf/2209.00638
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ImageRetrieval"></a> 

## 检索/Image Retrieval 

**Feature Representation Learning for Unsupervised Cross-domain Image Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2207.09721
- 代码/Code: https://github.com/conghuihu/ucdir

**A Sketch Is Worth a Thousand Words: Image Retrieval with Text and Sketch**

- 论文/Paper: http://arxiv.org/pdf/2208.03354
- 代码/Code: None

**CODER: Coupled Diversity-Sensitive Momentum Contrastive Learning for Image-Text Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2208.09843
- 代码/Code: None

[返回目录/back](#Contents)

<a name="diffusion"></a> 

**Lossy Image Compression with Conditional Diffusion Models**

- 论文/Paper: http://arxiv.org/pdf/2209.06950
- 代码/Code: None

[返回目录/back](#Contents)

<a name="Other"></a> 

## 其他/Other

**Embedding contrastive unsupervised features to cluster in- and out-of-distribution noise in corrupted image datasets**

- 论文/Paper: http://arxiv.org/pdf/2207.01573
- 代码/Code: None

**GraphVid: It Only Takes a Few Nodes to Understand a Video**

- 论文/Paper: http://arxiv.org/pdf/2207.01375
- 代码/Code: None

**Target-absent Human Attention**

- 论文/Paper: http://arxiv.org/pdf/2207.01166
- 代码/Code: None

**Lottery Ticket Hypothesis for Spiking Neural Networks**

- 论文/Paper: http://arxiv.org/pdf/2207.01382
- 代码/Code: None

**Improving Covariance Conditioning of the SVD Meta-layer by Orthogonality**

- 论文/Paper: http://arxiv.org/abs/2207.02119
- 代码/Code: https://github.com/kingjamessong/orthoimprovecond

**AvatarCap: Animatable Avatar Conditioned Monocular Human Volumetric Capture**

- 论文/Paper: http://arxiv.org/abs/2207.02031
- 代码/Code: https://github.com/lizhe00/AvatarCap.

**DeepPS2: Revisiting Photometric Stereo Using Two Differently Illuminated Images**

- 论文/Paper: http://arxiv.org/abs/2207.02025
- 代码/Code: None

**Learning Local Implicit Fourier Representation for Image Warping**

- 论文/Paper: http://arxiv.org/abs/2207.01831
- 代码/Code: https://github.com/jaewon-lee-b/ltew

**SESS: Saliency Enhancing with Scaling and Sliding**

- 论文/Paper: http://arxiv.org/abs/2207.01769
- 代码/Code: https://github.com/neouyghur/sess

**TM2T: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts**

- 论文/Paper: http://arxiv.org/abs/2207.01696
- 代码/Code: None

**DenseHybrid: Hybrid Anomaly Detection for Dense Open-set Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.02606
- 代码/Code: None

**FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling**

- 论文/Paper: http://arxiv.org/pdf/2207.02595
- 代码/Code: https://github.com/timothyhtimothy/fast-vqa

**Towards Realistic Semi-Supervised Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.02269
- 代码/Code: None

**OpenLDN: Learning to Discover Novel Classes for Open-World Semi-Supervised Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.02261
- 代码/Code: None

**Predicting is not Understanding: Recognizing and Addressing Underspecification in Machine Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.02598
- 代码/Code: None

**Factorizing Knowledge in Neural Networks**

- 论文/Paper: http://arxiv.org/pdf/2207.03337
- 代码/Code: None

**SuperTickets: Drawing Task-Agnostic Lottery Tickets from Supernets via Jointly Architecture Searching and Parameter Pruning**

- 论文/Paper: http://arxiv.org/pdf/2207.03677
- 代码/Code: https://github.com/RICE-EIC/SuperTickets.

**Video Dialog as Conversation about Objects Living in Space-Time**

- 论文/Paper: http://arxiv.org/pdf/2207.03656
- 代码/Code: https://github.com/hoanganhpham1006/COST

**Demystifying Unsupervised Semantic Correspondence Estimation**

- 论文/Paper: http://arxiv.org/pdf/2207.05054
- 代码/Code: None

**A Closer Look at Invariances in Self-supervised Pre-training for 3D Vision**

- 论文/Paper: http://arxiv.org/pdf/2207.04997
- 代码/Code: None

**DCCF: Deep Comprehensible Color Filter Learning Framework for High-Resolution Image Harmonization**

- 论文/Paper: http://arxiv.org/pdf/2207.04788
- 代码/Code: None

**Batch-efficient EigenDecomposition for Small and Medium Matrices**

- 论文/Paper: http://arxiv.org/pdf/2207.04228
- 代码/Code: None

**Few 'Zero Level Set'-Shot Learning of Shape Signed Distance Functions in Feature Space**

- 论文/Paper: http://arxiv.org/pdf/2207.04161
- 代码/Code: None

**Camera Pose Auto-Encoders for Improving Pose Regression**

- 论文/Paper: http://arxiv.org/pdf/2207.05530
- 代码/Code: https://github.com/yolish/camera-pose-auto-encoders

**Synergistic Self-supervised and Quantization Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.05432
- 代码/Code: https://github.com/megvii-research/SSQL-ECCV2022

**Frequency Domain Model Augmentation for Adversarial Attack**

- 论文/Paper: http://arxiv.org/pdf/2207.05382
- 代码/Code: https://github.com/yuyang-long/ssa

**Organic Priors in Non-Rigid Structure from Motion**

- 论文/Paper: http://arxiv.org/pdf/2207.06262
- 代码/Code: None

**Unsupervised Visual Representation Learning by Synchronous Momentum Grouping**

- 论文/Paper: http://arxiv.org/pdf/2207.06167
- 代码/Code: None

**Learning Implicit Templates for Point-Based Clothed Human Modeling**

- 论文/Paper: http://arxiv.org/pdf/2207.06955
- 代码/Code: https://github.com/jsnln/fite

**BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks**

- 论文/Paper: http://arxiv.org/pdf/2207.06873
- 代码/Code: https://github.com/explainableml/bayescap

**Lipschitz Continuity Retained Binary Neural Network**

- 论文/Paper: http://arxiv.org/pdf/2207.06540
- 代码/Code: https://github.com/42shawn/lcr_bnn

**3D Instances as 1D Kernels**

- 论文/Paper: http://arxiv.org/pdf/2207.07372
- 代码/Code: https://github.com/W1zheng/DKNet

**ScaleNet: Searching for the Model to Scale**

- 论文/Paper: http://arxiv.org/pdf/2207.07267
- 代码/Code: https://github.com/luminolx/ScaleNet

**Rethinking Data Augmentation for Robust Visual Question Answering**

- 论文/Paper: http://arxiv.org/pdf/2207.08739
- 代码/Code: https://github.com/ItemZheng/KDDAug

**Semantic Novelty Detection via Relational Reasoning**

- 论文/Paper: http://arxiv.org/pdf/2207.08699
- 代码/Code: None

**Label2Label: A Language Modeling Framework for Multi-Attribute Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.08677
- 代码/Code: https://github.com/Li-Wanhua/Label2Label.

**Towards High-Fidelity Single-view Holistic Reconstruction of Indoor Scenes**

- 论文/Paper: http://arxiv.org/pdf/2207.08656
- 代码/Code: https://github.com/UncleMEDM/InstPIFu

**Class-incremental Novel Class Discovery**

- 论文/Paper: http://arxiv.org/pdf/2207.08605
- 代码/Code: https://github.com/OatmealLiu/class-iNCD

**MPIB: An MPI-Based Bokeh Rendering Framework for Realistic Partial Occlusion Effects**

- 论文/Paper: http://arxiv.org/pdf/2207.08403
- 代码/Code: None

**SepLUT: Separable Image-adaptive Lookup Tables for Real-time Image Enhancement**

- 论文/Paper: http://arxiv.org/pdf/2207.08351
- 代码/Code: None

**Learning with Recoverable Forgetting**

- 论文/Paper: http://arxiv.org/pdf/2207.08224
- 代码/Code: None

**Zero-Shot Temporal Action Detection via Vision-Language Prompting**

- 论文/Paper: http://arxiv.org/pdf/2207.08184
- 代码/Code: https://github.com/sauradip/STALE

**Watermark Vaccine: Adversarial Attacks to Prevent Watermark Removal**

- 论文/Paper: http://arxiv.org/pdf/2207.08178
- 代码/Code: None

**FashionViL: Fashion-Focused Vision-and-Language Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.08150
- 代码/Code: https://github.com/BrandonHanx/mmf.

**E-NeRV: Expedite Neural Video Representation with Disentangled Spatial-Temporal Context**

- 论文/Paper: http://arxiv.org/pdf/2207.08132
- 代码/Code: https://github.com/kyleleey/E-NeRV.

**Neural Color Operators for Sequential Image Retouching**

- 论文/Paper: http://arxiv.org/pdf/2207.08080
- 代码/Code: https://github.com/amberwangyili/neurop

**Semi-Supervised Keypoint Detector and Descriptor for Retinal Image Matching**

- 论文/Paper: http://arxiv.org/pdf/2207.07932
- 代码/Code: None

**JPerceiver: Joint Perception Network for Depth, Pose and Layout Estimation in Driving Scenes**

- 论文/Paper: http://arxiv.org/pdf/2207.07895
- 代码/Code: at~\href{https://github.com/sunnyHelen/JPerceiver}{https://github.com/sunnyHelen/JPerceiver}.

**You Should Look at All Objects**

- 论文/Paper: http://arxiv.org/pdf/2207.07889
- 代码/Code: None

**NeFSAC: Neurally Filtered Minimal Samples**

- 论文/Paper: http://arxiv.org/pdf/2207.07872
- 代码/Code: https://github.com/cavalli1234/NeFSAC.

**CLOSE: Curriculum Learning On the Sharing Extent Towards Better One-shot NAS**

- 论文/Paper: http://arxiv.org/pdf/2207.07868
- 代码/Code: https://github.com/walkerning/aw_nas.

**Cross-Domain Cross-Set Few-Shot Learning via Learning Compact and Aligned Representations**

- 论文/Paper: http://arxiv.org/pdf/2207.07826
- 代码/Code: https://github.com/WentaoChen0813/CDCS-FSL

**Self-calibrating Photometric Stereo by Neural Inverse Rendering**

- 论文/Paper: http://arxiv.org/pdf/2207.07815
- 代码/Code: https://github.com/junxuan-li/SCPS-NIR

**Learning Long-Term Spatial-Temporal Graphs for Active Speaker Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.07783
- 代码/Code: https://github.com/SRA2/SPELL

**Towards Understanding The Semidefinite Relaxations of Truncated Least-Squares in Robust Rotation Search**

- 论文/Paper: http://arxiv.org/pdf/2207.08350
- 代码/Code: None

**PoserNet: Refining Relative Camera Poses Exploiting Object Detections**

- 论文/Paper: http://arxiv.org/pdf/2207.09445
- 代码/Code: https://github.com/IIT-PAVIS/PoserNet

**Geometric Features Informed Multi-person Human-object Interaction Recognition in Videos**

- 论文/Paper: http://arxiv.org/pdf/2207.09425
- 代码/Code: None

**Deep Semantic Statistics Matching (D2SM) Denoising Network**

- 论文/Paper: http://arxiv.org/pdf/2207.09302
- 代码/Code: None

**3D Room Layout Estimation from a Cubemap of Panorama Image via Deep Manhattan Hough Transform**

- 论文/Paper: http://arxiv.org/pdf/2207.09291
- 代码/Code: https://github.com/Starrah/DMH-Net

**NDF: Neural Deformable Fields for Dynamic Human Modelling**

- 论文/Paper: http://arxiv.org/pdf/2207.09193
- 代码/Code: None

**Self-Supervision Can Be a Good Few-Shot Learner**

- 论文/Paper: http://arxiv.org/pdf/2207.09176
- 代码/Code: https://github.com/bbbdylan/unisiam

**ParticleSfM: Exploiting Dense Point Trajectories for Localizing Moving Cameras in the Wild**

- 论文/Paper: http://arxiv.org/pdf/2207.09137
- 代码/Code: https://github.com/bytedance/particle-sfm.

**MHR-Net: Multiple-Hypothesis Reconstruction of Non-Rigid Shapes from 2D Views**

- 论文/Paper: http://arxiv.org/pdf/2207.09086
- 代码/Code: None

**SelectionConv: Convolutional Neural Networks for Non-rectilinear Image Data**

- 论文/Paper: http://arxiv.org/pdf/2207.08979
- 代码/Code: None

**Prior-Guided Adversarial Initialization for Fast Adversarial Training**

- 论文/Paper: http://arxiv.org/pdf/2207.08859
- 代码/Code: https://github.com/jiaxiaojunQAQ/FGSM-PGI.

**Prior Knowledge Guided Unsupervised Domain Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2207.08877
- 代码/Code: https://github.com/tsun/KUDA

**Discover and Mitigate Unknown Biases with Debiasing Alternate Networks**

- 论文/Paper: http://arxiv.org/pdf/2207.10077
- 代码/Code: https://github.com/zhihengli-UR/DebiAN

**Difficulty-Aware Simulator for Open Set Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.10024
- 代码/Code: https://github.com/wjun0830/difficulty-aware-simulator

**Tailoring Self-Supervision for Supervised Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.10023
- 代码/Code: https://github.com/wjun0830/localizable-rotation

**Overcoming Shortcut Learning in a Target Domain by Generalizing Basic Visual Factors from a Source Domain**

- 论文/Paper: http://arxiv.org/pdf/2207.10002
- 代码/Code: https://github.com/boschresearch/sourcegen

**Temporal and cross-modal attention for audio-visual zero-shot learning**

- 论文/Paper: http://arxiv.org/pdf/2207.09966
- 代码/Code: https://github.com/explainableml/tcaf-gzsl

**Telepresence Video Quality Assessment**

- 论文/Paper: http://arxiv.org/pdf/2207.09956
- 代码/Code: None

**Towards Efficient and Scale-Robust Ultra-High-Definition Image Demoireing**

- 论文/Paper: http://arxiv.org/pdf/2207.09935
- 代码/Code: None

**Negative Samples are at Large: Leveraging Hard-distance Elastic Loss for Re-identification**

- 论文/Paper: http://arxiv.org/pdf/2207.09884
- 代码/Code: None

**Discrete-Constrained Regression for Local Counting Models**

- 论文/Paper: http://arxiv.org/pdf/2207.09865
- 代码/Code: None

**Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction**

- 论文/Paper: http://arxiv.org/pdf/2207.09705
- 代码/Code: None

**Efficient Meta-Tuning for Content-aware Neural Video Delivery**

- 论文/Paper: http://arxiv.org/pdf/2207.09691
- 代码/Code: https://github.com/neural-video-delivery/emt-pytorch-eccv2022

**Object-Compositional Neural Implicit Surfaces**

- 论文/Paper: http://arxiv.org/pdf/2207.09686
- 代码/Code: https://github.com/qianyiwu/objsdf

**Explaining Deepfake Detection by Analysing Image Matching**

- 论文/Paper: http://arxiv.org/pdf/2207.09679
- 代码/Code: https://github.com/megvii-research/fst-matching

**ERA: Expert Retrieval and Assembly for Early Action Prediction**

- 论文/Paper: http://arxiv.org/pdf/2207.09675
- 代码/Code: None

**Perspective Phase Angle Model for Polarimetric 3D Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2207.09629
- 代码/Code: https://github.com/gcchen97/ppa4p3d

**Explicit Image Caption Editing**

- 论文/Paper: http://arxiv.org/pdf/2207.09625
- 代码/Code: https://github.com/baaaad/ece

**Unsupervised Deep Multi-Shape Matching**

- 论文/Paper: http://arxiv.org/pdf/2207.09610
- 代码/Code: None

**Contributions of Shape, Texture, and Color in Visual Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.09510
- 代码/Code: https://github.com/gyhandy/humanoid-vision-engine

**Novel Class Discovery without Forgetting**

- 论文/Paper: http://arxiv.org/pdf/2207.10659
- 代码/Code: None

**Approximate Differentiable Rendering with Algebraic Surfaces**

- 论文/Paper: http://arxiv.org/pdf/2207.10606
- 代码/Code: None

**FADE: Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling**

- 论文/Paper: http://arxiv.org/pdf/2207.10392
- 代码/Code: None

**Error Compensation Framework for Flow-Guided Video Inpainting**

- 论文/Paper: http://arxiv.org/pdf/2207.10391
- 代码/Code: None

**NSNet: Non-saliency Suppression Sampler for Efficient Video Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.10388
- 代码/Code: None

**Temporal Saliency Query Network for Efficient Video Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.10379
- 代码/Code: None

**UFO: Unified Feature Optimization**

- 论文/Paper: http://arxiv.org/pdf/2207.10341
- 代码/Code: None

**OIMNet++: Prototypical Normalization and Localization-aware Learning for Person Search**

- 论文/Paper: http://arxiv.org/pdf/2207.10320
- 代码/Code: None

**Towards Accurate Open-Set Recognition via Background-Class Regularization**

- 论文/Paper: http://arxiv.org/pdf/2207.10287
- 代码/Code: None

**Grounding Visual Representations with Texts for Domain Generalization**

- 论文/Paper: http://arxiv.org/pdf/2207.10285
- 代码/Code: https://github.com/mswzeus/gvrt

**SPIN: An Empirical Evaluation on Sharing Parameters of Isotropic Networks**

- 论文/Paper: http://arxiv.org/pdf/2207.10237
- 代码/Code: https://github.com/apple/ml-spin

**MeshMAE: Masked Autoencoders for 3D Mesh Data Analysis**

- 论文/Paper: http://arxiv.org/pdf/2207.10228
- 代码/Code: None

**On Label Granularity and Object Localization**

- 论文/Paper: http://arxiv.org/pdf/2207.10225
- 代码/Code: https://github.com/visipedia/inat_loc

**Spotting Temporally Precise, Fine-Grained Events in Video**

- 论文/Paper: http://arxiv.org/pdf/2207.10213
- 代码/Code: None

**Video Anomaly Detection by Solving Decoupled Spatio-Temporal Jigsaw Puzzles**

- 论文/Paper: http://arxiv.org/pdf/2207.10172
- 代码/Code: None

**GOCA: Guided Online Cluster Assignment for Self-Supervised Video Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.10158
- 代码/Code: https://github.com/seleucia/goca

**Visual Knowledge Tracing**

- 论文/Paper: http://arxiv.org/pdf/2207.10157
- 代码/Code: https://github.com/nkondapa/visualknowledgetracing

**Tackling Long-Tailed Category Distribution Under Domain Shifts**

- 论文/Paper: http://arxiv.org/pdf/2207.10150
- 代码/Code: https://github.com/guxiao0822/lt-ds

**Latent Discriminant deterministic Uncertainty**

- 论文/Paper: http://arxiv.org/pdf/2207.10130
- 代码/Code: https://github.com/ensta-u2is/ldu

**Animation from Blur: Multi-modal Blur Decomposition with Motion Guidance**

- 论文/Paper: http://arxiv.org/pdf/2207.10123
- 代码/Code: https://github.com/zzh-tech/Animation-from-Blur.

**Bitwidth-Adaptive Quantization-Aware Neural Network Training: A Meta-Learning Approach**

- 论文/Paper: http://arxiv.org/pdf/2207.10188
- 代码/Code: None

**Structural Causal 3D Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2207.10156
- 代码/Code: None

**AudioScopeV2: Audio-Visual Attention Architectures for Calibrated Open-Domain On-Screen Sound Separation**

- 论文/Paper: http://arxiv.org/pdf/2207.10141
- 代码/Code: None

**Continual Variational Autoencoder Learning via Online Cooperative Memorization**

- 论文/Paper: http://arxiv.org/pdf/2207.10131
- 代码/Code: https://github.com/dtuzi123/ovae

**Panoptic Scene Graph Generation**

- 论文/Paper: http://arxiv.org/pdf/2207.11247
- 代码/Code: https://github.com/Jingkang50/OpenPSG

**Few-Shot Class-Incremental Learning via Entropy-Regularized Data-Free Replay**

- 论文/Paper: http://arxiv.org/pdf/2207.11213
- 代码/Code: None

**POP: Mining POtential Performance of new fashion products via webly cross-modal query expansion**

- 论文/Paper: http://arxiv.org/pdf/2207.11001
- 代码/Code: https://github.com/HumaticsLAB/POP-Mining-POtential-Performance

**Few-shot Object Counting and Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.10988
- 代码/Code: https://github.com/VinAIResearch/Counting-DETR

**Dynamic Local Aggregation Network with Adaptive Clusterer for Anomaly Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.10948
- 代码/Code: https://github.com/Beyond-Zw/DLAN-AC.

**My View is the Best View: Procedure Learning from Egocentric Videos**

- 论文/Paper: http://arxiv.org/pdf/2207.10883
- 代码/Code: https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning

**Prototype-Guided Continual Adaptation for Class-Incremental Unsupervised Domain Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2207.10856
- 代码/Code: https://github.com/Hongbin98/ProCA.git

**MeshLoc: Mesh-Based Visual Localization**

- 论文/Paper: http://arxiv.org/pdf/2207.10762
- 代码/Code: None

**MemSAC: Memory Augmented Sample Consistency for Large Scale Domain Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2207.12389
- 代码/Code: None

**Deforming Radiance Fields with Cages**

- 论文/Paper: http://arxiv.org/pdf/2207.12298
- 代码/Code: None

**Equivariance and Invariance Inductive Bias for Learning from Insufficient Data**

- 论文/Paper: http://arxiv.org/pdf/2207.12258
- 代码/Code: https://github.com/Wangt-CN/EqInv

**Black-box Few-shot Knowledge Distillation**

- 论文/Paper: http://arxiv.org/pdf/2207.12106
- 代码/Code: https://github.com/nphdang/FS-BBT

**Balancing Stability and Plasticity through Advanced Null Space in Continual Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.12061
- 代码/Code: None

**Optimal Boxes: Boosting End-to-End Scene Text Recognition by Adjusting Annotated Bounding Boxes via Reinforcement Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.11934
- 代码/Code: None

**NeuMesh: Learning Disentangled Neural Mesh-based Implicit Field for Geometry and Texture Editing**

- 论文/Paper: http://arxiv.org/pdf/2207.11911
- 代码/Code: None

**Domain Adaptive Person Search**

- 论文/Paper: http://arxiv.org/pdf/2207.11898
- 代码/Code: https://github.com/caposerenity/DAPS.

**VizWiz-FewShot: Locating Objects in Images Taken by People With Visual Impairments**

- 论文/Paper: http://arxiv.org/pdf/2207.11810
- 代码/Code: None

**Label-Guided Auxiliary Training Improves 3D Object Detector**

- 论文/Paper: http://arxiv.org/pdf/2207.11753
- 代码/Code: None

**Combining Internal and External Constraints for Unrolling Shutter in Videos**

- 论文/Paper: http://arxiv.org/pdf/2207.11725
- 代码/Code: None

**TIPS: Text-Induced Pose Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2207.11718
- 代码/Code: None

**Improving Test-Time Adaptation via Shift-agnostic Weight Regularization and Nearest Source Prototypes**

- 论文/Paper: http://arxiv.org/pdf/2207.11707
- 代码/Code: None

**Learning Graph Neural Networks for Image Style Transfer**

- 论文/Paper: http://arxiv.org/pdf/2207.11681
- 代码/Code: None

**Contrastive Monotonic Pixel-Level Modulation**

- 论文/Paper: http://arxiv.org/pdf/2207.11517
- 代码/Code: https://github.com/lukun199/MonoPix.

**CompNVS: Novel View Synthesis with Scene Completion**

- 论文/Paper: http://arxiv.org/pdf/2207.11467
- 代码/Code: None

**When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.11463
- 代码/Code: https://github.com/LBH1024/CAN.

**Meta Spatio-Temporal Debiasing for Video Scene Graph Generation**

- 论文/Paper: http://arxiv.org/pdf/2207.11441
- 代码/Code: None

**3D Shape Sequence of Human Comparison and Classification using Current and Varifolds**

- 论文/Paper: http://arxiv.org/pdf/2207.12485
- 代码/Code: https://github.com/cristal-3dsam/humancomparisonvarifolds

**NewsStories: Illustrating articles with visual summaries**

- 论文/Paper: http://arxiv.org/pdf/2207.13061
- 代码/Code: https://github.com/newsstoriesdata/newsstories.github.io

**Efficient One Pass Self-distillation with Zipf's Label Smoothing**

- 论文/Paper: http://arxiv.org/pdf/2207.12980
- 代码/Code: https://github.com/megvii-research/zipfls

**AlignSDF: Pose-Aligned Signed Distance Fields for Hand-Object Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2207.12909
- 代码/Code: None

**Static and Dynamic Concepts for Self-supervised Video Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.12795
- 代码/Code: None

**Learning Hierarchy Aware Features for Reducing Mistake Severity**

- 论文/Paper: http://arxiv.org/pdf/2207.12646
- 代码/Code: https://github.com/07agarg/haf

**Translating a Visual LEGO Manual to a Machine-Executable Plan**

- 论文/Paper: http://arxiv.org/pdf/2207.12572
- 代码/Code: None

**Semi-Leak: Membership Inference Attacks Against Semi-supervised Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.12535
- 代码/Code: https://github.com/xinleihe/semi-leak

**Trainability Preserving Neural Structured Pruning**

- 论文/Paper: http://arxiv.org/pdf/2207.12534
- 代码/Code: https://github.com/mingsun-tse/tpp

**Shift-tolerant Perceptual Similarity Metric**

- 论文/Paper: http://arxiv.org/pdf/2207.13686
- 代码/Code: http://github.com/abhijay9/ShiftTolerant-LPIPS/

**Abstracting Sketches through Simple Primitives**

- 论文/Paper: http://arxiv.org/pdf/2207.13543
- 代码/Code: https://github.com/ExplainableML/sketch-primitives.

**AutoTransition: Learning to Recommend Video Transition Effects**

- 论文/Paper: http://arxiv.org/pdf/2207.13479
- 代码/Code: https://github.com/acherstyx/AutoTransition

**Hardly Perceptible Trojan Attack against Neural Networks with Bit Flips**

- 论文/Paper: http://arxiv.org/pdf/2207.13417
- 代码/Code: https://github.com/jiawangbai/HPT

**Identifying Hard Noise in Long-Tailed Sample Distribution**

- 论文/Paper: http://arxiv.org/pdf/2207.13378
- 代码/Code: https://github.com/yxymessi/H2E-Framework

**One-Trimap Video Matting**

- 论文/Paper: http://arxiv.org/pdf/2207.13353
- 代码/Code: https://github.com/Hongje/OTVM

**PointFix: Learning to Fix Domain Bias for Robust Online Stereo Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2207.13340
- 代码/Code: None

**End-to-end Graph-constrained Vectorized Floorplan Generation with Panoptic Refinement**

- 论文/Paper: http://arxiv.org/pdf/2207.13268
- 代码/Code: None

**Spatiotemporal Self-attention Modeling with Temporal Patch Shift for Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2207.13259
- 代码/Code: https://github.com/MartinXM/TPS

**Concurrent Subsidiary Supervision for Unsupervised Source-Free Domain Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2207.13247
- 代码/Code: None

**LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity**

- 论文/Paper: http://arxiv.org/pdf/2207.13129
- 代码/Code: None

**Initialization and Alignment for Adversarial Texture Optimization**

- 论文/Paper: http://arxiv.org/pdf/2207.14289
- 代码/Code: None

**Depth Field Networks for Generalizable Multi-view Scene Representation**

- 论文/Paper: http://arxiv.org/pdf/2207.14287
- 代码/Code: None

**Mining Cross-Person Cues for Body-Part Interactiveness Learning in HOI Detection**

- 论文/Paper: http://arxiv.org/pdf/2207.14192
- 代码/Code: https://github.com/enlighten0707/Body-Part-Map-for-Interactiveness.

**Neural Strands: Learning Hair Geometry and Appearance from Multi-View Images**

- 论文/Paper: http://arxiv.org/pdf/2207.14067
- 代码/Code: None

**Break and Make: Interactive Structural Understanding Using LEGO Bricks**

- 论文/Paper: http://arxiv.org/pdf/2207.13738
- 代码/Code: https://github.com/aaronwalsman/ltron.

**A Repulsive Force Unit for Garment Collision Handling in Neural Networks**

- 论文/Paper: http://arxiv.org/pdf/2207.13871
- 代码/Code: None

**Minimal Neural Atlas: Parameterizing Complex Surfaces with Minimal Charts and Distortion**

- 论文/Paper: http://arxiv.org/pdf/2207.14782
- 代码/Code: https://github.com/low5545/minimal-neural-atlas

**Can Shuffling Video Benefit Temporal Bias Problem: A Novel Training Framework for Temporal Grounding**

- 论文/Paper: http://arxiv.org/pdf/2207.14698
- 代码/Code: https://github.com/haojc/ShufflingVideosForTSG.

**AlphaVC: High-Performance and Efficient Learned Video Compression**

- 论文/Paper: http://arxiv.org/pdf/2207.14678
- 代码/Code: None

**WISE: Whitebox Image Stylization by Example-based Learning**

- 论文/Paper: http://arxiv.org/pdf/2207.14606
- 代码/Code: None

**Centrality and Consistency: Two-Stage Clean Samples Identification for Learning with Instance-Dependent Noisy Labels**

- 论文/Paper: http://arxiv.org/pdf/2207.14476
- 代码/Code: None

**Video Question Answering with Iterative Video-Text Co-Tokenization**

- 论文/Paper: http://arxiv.org/pdf/2208.00934
- 代码/Code: None

**S$^2$Contact: Graph-based Network for 3D Hand-Object Contact Estimation with Semi-Supervised Learning**

- 论文/Paper: http://arxiv.org/pdf/2208.00874
- 代码/Code: None

**Skeleton-free Pose Transfer for Stylized 3D Characters**

- 论文/Paper: http://arxiv.org/pdf/2208.00790
- 代码/Code: None

**Improving Fine-Grained Visual Recognition in Low Data Regimes via Self-Boosting Attention Mechanism**

- 论文/Paper: http://arxiv.org/pdf/2208.00617
- 代码/Code: https://github.com/GANPerf/SAM

**SdAE: Self-distillated Masked Autoencoder**

- 论文/Paper: http://arxiv.org/pdf/2208.00449
- 代码/Code: https://github.com/AbrahamYabo/SdAE.

**Out-of-Distribution Detection with Semantic Mismatch under Masking**

- 论文/Paper: http://arxiv.org/pdf/2208.00446
- 代码/Code: https://github.com/cure-lab/MOODCat

**Skeleton-Parted Graph Scattering Networks for 3D Human Motion Prediction**

- 论文/Paper: http://arxiv.org/pdf/2208.00368
- 代码/Code: None

**Revisiting the Critical Factors of Augmentation-Invariant Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2208.00275
- 代码/Code: None

**Few-shot Single-view 3D Reconstruction with Memory Prior Contrastive Network**

- 论文/Paper: http://arxiv.org/pdf/2208.00183
- 代码/Code: None

**Few-Shot Class-Incremental Learning from an Open-Set Perspective**

- 论文/Paper: http://arxiv.org/pdf/2208.00147
- 代码/Code: None

**DAS: Densely-Anchored Sampling for Deep Metric Learning**

- 论文/Paper: http://arxiv.org/pdf/2208.00119
- 代码/Code: https://github.com/lizhaoliu-Lec/DAS

**Fast Two-step Blind Optical Aberration Correction**

- 论文/Paper: http://arxiv.org/pdf/2208.00950
- 代码/Code: None

**Negative Frames Matter in Egocentric Visual Query 2D Localization**

- 论文/Paper: http://arxiv.org/pdf/2208.01949
- 代码/Code: https://github.com/facebookresearch/vq2d_cvpr

**Neighborhood Collective Estimation for Noisy Label Identification and Correction**

- 论文/Paper: http://arxiv.org/pdf/2208.03207
- 代码/Code: None

**PlaneFormers: From Sparse View Planes to 3D Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2208.04307
- 代码/Code: None

**SLiDE: Self-supervised LiDAR De-snowing through Reconstruction Difficulty**

- 论文/Paper: http://arxiv.org/pdf/2208.04043
- 代码/Code: None

**Domain Randomization-Enhanced Depth Simulation and Restoration for Perceiving and Grasping Specular and Transparent Objects**

- 论文/Paper: http://arxiv.org/pdf/2208.03792
- 代码/Code: https://github.com/PKU-EPIC/DREDS

**Class-Incremental Learning with Cross-Space Clustering and Controlled Transfer**

- 论文/Paper: http://arxiv.org/pdf/2208.03767
- 代码/Code: None

**Learning Omnidirectional Flow in 360-degree Video via Siamese Representation**

- 论文/Paper: http://arxiv.org/pdf/2208.03620
- 代码/Code: None

**Inpainting at Modern Camera Resolution by Guided PatchMatch with Auto-Curation**

- 论文/Paper: http://arxiv.org/pdf/2208.03552
- 代码/Code: None

**Contrastive Positive Mining for Unsupervised 3D Action Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2208.03497
- 代码/Code: None

**Speaker-adaptive Lip Reading with User-dependent Padding**

- 论文/Paper: http://arxiv.org/pdf/2208.04498
- 代码/Code: None

**Contrast-Phys: Unsupervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast**

- 论文/Paper: http://arxiv.org/pdf/2208.04378
- 代码/Code: https://github.com/zhaodongsun/contrast-phys

**Rethinking Robust Representation Learning Under Fine-grained Noisy Faces**

- 论文/Paper: http://arxiv.org/pdf/2208.04352
- 代码/Code: None

**RDA: Reciprocal Distribution Alignment for Robust SSL**

- 论文/Paper: http://arxiv.org/pdf/2208.04619
- 代码/Code: https://github.com/njuyued/rda4robustssl

**RelPose: Predicting Probabilistic Relative Rotation for Single Objects in the Wild**

- 论文/Paper: http://arxiv.org/pdf/2208.05963
- 代码/Code: None

**PointTree: Transformation-Robust Point Cloud Encoder with Relaxed K-D Trees**

- 论文/Paper: http://arxiv.org/pdf/2208.05962
- 代码/Code: https://github.com/immortalco/pointtree

**MixSKD: Self-Knowledge Distillation from Mixup for Image Recognition**

- 论文/Paper: http://arxiv.org/pdf/2208.05768
- 代码/Code: https://github.com/winycg/self-kd-lib

**PRIF: Primary Ray-based Implicit Function**

- 论文/Paper: http://arxiv.org/pdf/2208.06143
- 代码/Code: None

**Learning Semantic Correspondence with Sparse Annotations**

- 论文/Paper: http://arxiv.org/pdf/2208.06974
- 代码/Code: None

**CCRL: Contrastive Cell Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2208.06445
- 代码/Code: None

**Pose Forecasting in Industrial Human-Robot Collaboration**

- 论文/Paper: http://arxiv.org/pdf/2208.07308
- 代码/Code: None

**Combating Label Distribution Shift for Active Domain Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2208.06604
- 代码/Code: None

**Matching Multiple Perspectives for Efficient Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2208.07654
- 代码/Code: None

**Uncertainty-guided Source-free Domain Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2208.07591
- 代码/Code: https://github.com/roysubhankar/uncertainty-sfda

**Context-Aware Streaming Perception in Dynamic Environments**

- 论文/Paper: http://arxiv.org/pdf/2208.07479
- 代码/Code: https://github.com/eyalsel/contextual-streaming-perception

**Towards an Error-free Deep Occupancy Detector for Smart Camera Parking System**

- 论文/Paper: http://arxiv.org/pdf/2208.08220
- 代码/Code: None

**AdaBin: Improving Binary Neural Networks with Adaptive Binary Sets**

- 论文/Paper: http://arxiv.org/pdf/2208.08084
- 代码/Code: None

**DLCFT: Deep Linear Continual Fine-Tuning for General Incremental Learning**

- 论文/Paper: http://arxiv.org/pdf/2208.08112
- 代码/Code: None

**L3: Accelerator-Friendly Lossless Image Format for High-Resolution, High-Throughput DNN Training**

- 论文/Paper: http://arxiv.org/pdf/2208.08711
- 代码/Code: https://github.com/snu-arc/l3

**ConMatch: Semi-Supervised Learning with Confidence-Guided Consistency Regularization**

- 论文/Paper: http://arxiv.org/pdf/2208.08631
- 代码/Code: https://github.com/jiwoncocoder/conmatch

**Unifying Visual Perception by Dispersible Points Learning**

- 论文/Paper: http://arxiv.org/pdf/2208.08630
- 代码/Code: https://github.com/sense-x/unihead

**Visual Cross-View Metric Localization with Dense Uncertainty Estimates**

- 论文/Paper: http://arxiv.org/pdf/2208.08519
- 代码/Code: https://github.com/tudelft-iv/crossviewmetriclocalization

**GCISG: Guided Causal Invariant Learning for Improved Syn-to-real Generalization**

- 论文/Paper: http://arxiv.org/pdf/2208.10024
- 代码/Code: None

**SIM2E: Benchmarking the Group Equivariant Capability of Correspondence Matching Algorithms**

- 论文/Paper: http://arxiv.org/pdf/2208.09896
- 代码/Code: None

**Artifact-Based Domain Generalization of Skin Lesion Models**

- 论文/Paper: http://arxiv.org/pdf/2208.09756
- 代码/Code: None

**Fuse and Attend: Generalized Embedding Learning for Art and Sketches**

- 论文/Paper: http://arxiv.org/pdf/2208.09698
- 代码/Code: None

**Effectiveness of Function Matching in Driving Scene Recognition**

- 论文/Paper: http://arxiv.org/pdf/2208.09694
- 代码/Code: None

**Consistency Regularization for Domain Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2208.11084
- 代码/Code: https://github.com/kw01sg/crda

**IMPaSh: A Novel Domain-shift Resistant Representation for Colorectal Cancer Tissue Classification**

- 论文/Paper: http://arxiv.org/pdf/2208.11052
- 代码/Code: https://github.com/trinhvg/impash

**Deep Structural Causal Shape Models**

- 论文/Paper: http://arxiv.org/pdf/2208.10950
- 代码/Code: None

**Learning from Noisy Labels with Coarse-to-Fine Sample Credibility Modeling**

- 论文/Paper: http://arxiv.org/pdf/2208.10683
- 代码/Code: None

**Anatomy-Aware Contrastive Representation Learning for Fetal Ultrasound**

- 论文/Paper: http://arxiv.org/pdf/2208.10642
- 代码/Code: None

**The Value of Out-of-Distribution Data**

- 论文/Paper: http://arxiv.org/pdf/2208.10967
- 代码/Code: None

**Ultra-high-resolution unpaired stain transformation via Kernelized Instance Normalization**

- 论文/Paper: http://arxiv.org/pdf/2208.10730
- 代码/Code: https://github.com/kaminyou/urust

**RIBAC: Towards Robust and Imperceptible Backdoor Attack against Compact DNN**

- 论文/Paper: http://arxiv.org/pdf/2208.10608
- 代码/Code: https://github.com/huyvnphan/eccv2022-ribac

**Cross-Camera View-Overlap Recognition**

- 论文/Paper: http://arxiv.org/pdf/2208.11661
- 代码/Code: None

**On the Design of Privacy-Aware Cameras: a Study on Deep Neural Networks**

- 论文/Paper: http://arxiv.org/pdf/2208.11372
- 代码/Code: https://github.com/upciti/privacy-by-design-semseg

**Discovering Transferable Forensic Features for CNN-generated Images Detection**

- 论文/Paper: http://arxiv.org/pdf/2208.11342
- 代码/Code: None

**Doc2Graph: a Task Agnostic Document Understanding Framework based on Graph Neural Networks**

- 论文/Paper: http://arxiv.org/pdf/2208.11168
- 代码/Code: https://github.com/andreagemelli/doc2graph

**Learning Continuous Implicit Representation for Near-Periodic Patterns**

- 论文/Paper: http://arxiv.org/pdf/2208.12278
- 代码/Code: None

**NeuralSI: Structural Parameter Identification in Nonlinear Dynamical Systems**

- 论文/Paper: http://arxiv.org/pdf/2208.12771
- 代码/Code: https://github.com/human-analysis/neural-structural-identification

**Take One Gram of Neural Features, Get Enhanced Group Robustness**

- 论文/Paper: http://arxiv.org/pdf/2208.12625
- 代码/Code: None

**CIRCLe: Color Invariant Representation Learning for Unbiased Classification of Skin Lesions**

- 论文/Paper: http://arxiv.org/pdf/2208.13528
- 代码/Code: None

**ASpanFormer: Detector-Free Image Matching with Adaptive Span Transformer**

- 论文/Paper: http://arxiv.org/pdf/2208.14201
- 代码/Code: None

**Probing Contextual Diversity for Dense Out-of-Distribution Detection**

- 论文/Paper: http://arxiv.org/pdf/2208.14195
- 代码/Code: None

**CAIR: Fast and Lightweight Multi-Scale Color Attention Network for Instagram Filter Removal**

- 论文/Paper: http://arxiv.org/pdf/2208.14039
- 代码/Code: https://github.com/hnv-lab/cair

**FUSION: Fully Unsupervised Test-Time Stain Adaptation via Fused Normalization Statistics**

- 论文/Paper: http://arxiv.org/pdf/2208.14206
- 代码/Code: None

**Style-Agnostic Reinforcement Learning**

- 论文/Paper: http://arxiv.org/pdf/2208.14863
- 代码/Code: https://github.com/postech-cvlab/style-agnostic-rl

**LiteDepth: Digging into Fast and Accurate Depth Estimation on Mobile Devices**

- 论文/Paper: http://arxiv.org/pdf/2209.00961
- 代码/Code: https://github.com/zhyever/LiteDepth

**Unpaired Image Translation via Vector Symbolic Architectures**

- 论文/Paper: http://arxiv.org/pdf/2209.02686
- 代码/Code: None

**CNSNet: A Cleanness-Navigated-Shadow Network for Shadow Removal**

- 论文/Paper: http://arxiv.org/pdf/2209.02174
- 代码/Code: None

**Semi-Supervised Domain Adaptation by Similarity based Pseudo-label Injection**

- 论文/Paper: http://arxiv.org/pdf/2209.01881
- 代码/Code: None

**Recurrent Bilinear Optimization for Binary Neural Networks**

- 论文/Paper: http://arxiv.org/pdf/2209.01542
- 代码/Code: https://github.com/SteveTsui/RBONN

**Meta-Learning with Less Forgetting on Large-Scale Non-Stationary Task Distributions**

- 论文/Paper: http://arxiv.org/pdf/2209.01501
- 代码/Code: None

**Towards Accurate Binary Neural Networks via Modeling Contextual Dependencies**

- 论文/Paper: http://arxiv.org/pdf/2209.01404
- 代码/Code: https://github.com/Sense-GVT/BCDN

**Interpretations Steered Network Pruning via Amortized Inferred Saliency Maps**

- 论文/Paper: http://arxiv.org/pdf/2209.02869
- 代码/Code: https://github.com/Alii-Ganjj/InterpretationsSteeredPruning

**Exploring Anchor-based Detection for Ego4D Natural Language Query**

- 论文/Paper: http://arxiv.org/pdf/2208.05375
- 代码/Code: None

**Detecting Driver Drowsiness as an Anomaly Using LSTM Autoencoders**

- 论文/Paper: http://arxiv.org/pdf/2209.05269
- 代码/Code: None

**Switchable Online Knowledge Distillation**

- 论文/Paper: http://arxiv.org/pdf/2209.04996
- 代码/Code: https://github.com/hfutqian/SwitOKD

**Self-supervised Human Mesh Recovery with Cross-Representation Alignment**

- 论文/Paper: http://arxiv.org/pdf/2209.04596
- 代码/Code: None

**Check and Link: Pairwise Lesion Correspondence Guides Mammogram Mass Detection**

- 论文/Paper: http://arxiv.org/pdf/2209.05809
- 代码/Code: None

**PointScatter: Point Set Representation for Tubular Structure Extraction**

- 论文/Paper: http://arxiv.org/pdf/2209.05774
- 代码/Code: https://github.com/zhangzhao2022/pointscatter

**Adversarial Coreset Selection for Efficient Robust Training**

- 论文/Paper: http://arxiv.org/pdf/2209.05785
- 代码/Code: None

**Out-of-Vocabulary Challenge Report**

- 论文/Paper: http://arxiv.org/pdf/2209.06717
- 代码/Code: None

**DevNet: Self-supervised Monocular Depth Learning via Density Volume Construction**

- 论文/Paper: http://arxiv.org/pdf/2209.06351
- 代码/Code: https://github.com/gitkaichenzhou/DevNet.

**MIPI 2022 Challenge on RGB+ToF Depth Completion: Dataset and Report**

- 论文/Paper: http://arxiv.org/pdf/2209.07057
- 代码/Code: https://github.com/mipi-challenge/MIPI2022.

**MIPI 2022 Challenge on Quad-Bayer Re-mosaic: Dataset and Report**

- 论文/Paper: http://arxiv.org/pdf/2209.07060
- 代码/Code: https://github.com/mipi-challenge/MIPI2022.

**MIPI 2022 Challenge on Under-Display Camera Image Restoration: Methods and Results**

- 论文/Paper: http://arxiv.org/pdf/2209.07052
- 代码/Code: https://github.com/mipi-challenge/MIPI2022.

**Hydra Attention: Efficient Attention with Many Heads**

- 论文/Paper: http://arxiv.org/pdf/2209.07484
- 代码/Code: None

[返回目录/back](#Contents)

