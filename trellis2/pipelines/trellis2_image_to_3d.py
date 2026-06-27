from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel

if TYPE_CHECKING:
    # 仅供 IDE/Pylance 静态分析；不会在运行时额外导入模型实现。
    from ..models.sparse_structure_flow import SparseStructureFlowModel
    from ..models.sparse_structure_vae import SparseStructureDecoder
    from ..models.structured_latent_flow import ElasticSLatFlowModel
    from ..models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        预处理输入图像：缩放、去背景、裁剪并合成到黑色背景上。

        处理流程：
        1. 检测图像是否已有透明通道（alpha），若有则直接使用，否则调用去背景模型
        2. 将图像缩放到最大边不超过 1024 像素
        3. 对图像进行去背景处理（若无透明通道）
        4. 根据前景区域的边界框进行正方形裁剪
        5. 将前景预乘 alpha 合成到黑色背景上，输出 RGB 图像

        Args:
            input (Image.Image): 输入的 PIL 图像，可以是 RGB 或 RGBA 模式。

        Returns:
            Image.Image: 预处理后的 RGB 图像，前景已预乘 alpha 合成到黑色背景上。
        """
        # 检测图像是否已有有效的透明通道
        # 如果是 RGBA 模式且 alpha 通道不全为 255，则认为已有透明通道，无需去背景
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True

        # 缩放图像：确保最大边不超过 1024 像素，使用 LANCZOS 高质量插值
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)

        # 去背景处理：若无透明通道，则调用 rembg 模型去除背景
        # low_vram 模式下会临时将模型移到 GPU，处理完后移回 CPU 以节省显存
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()

        # 根据前景区域（alpha > 0.8*255）的边界框进行正方形裁剪
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        # 计算前景区域的边界框 (x_min, y_min, x_max, y_max)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        # 以前景中心为基准，取最大宽/高作为边长，构建正方形裁剪框
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)  # 乘以 1 保持不变，可调整此系数来控制裁剪留白
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore

        # 预乘 alpha 合成：将前景合成到黑色背景上，输出为 RGB 图像
        # rgb * alpha 使得透明区域变为黑色，半透明区域按比例混合
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        从输入图像中提取条件特征向量，用于指导 flow model 的生成。

        可选用的模型（由第 104 行初始化，通过 pipeline.json 配置）：
        - DinoV2FeatureExtractor: 基于 DINOv2 的特征提取器
          支持的 model_name: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14 等
        - DinoV3FeatureExtractor: 基于 DINOv3 的特征提取器（当前项目默认使用）
          支持的 model_name: "facebook/dinov3-vitl16-pretrain-lvd1689m" 等
          当前所有配置文件均使用 DinoV3FeatureExtractor + dinov3-vitl16

        输入 shape：
        - image 为 PIL Image 列表时: [Image.Image, ...]，内部会被 resize 为
          (resolution, resolution) 并转为 Tensor
        - image 为 Tensor 时: (B, 3, H, W)，值域 [0, 1]
        - resolution: 512 或 1024，决定图像被缩放的尺寸

        输出 shape：
        - 'cond': (B, N, D) 正向条件特征
          - B = batch_size (通常为 1)
          - N = (resolution / patch_size)² 个 patch token
            - DinoV2 (patch_size=14): N = (512/14)² ≈ 1369 (512分辨率)
            - DinoV3 (patch_size=16): N = (512/16)² = 1024 (512分辨率)
                                        N = (1024/16)² = 4096 (1024分辨率)
          - D = embedding_dim，取决于模型:
            - dinov2_vitl14: D = 1024
            - dinov3-vitl16: D = 1024
        - 'neg_cond': (B, N, D) 全零张量，与 cond 同形状，用于 classifier-free guidance

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): 输入图像，可为 PIL 图像列表或张量
            resolution (int): 条件特征分辨率，通常为 512 或 1024
            include_neg_cond (bool): 是否生成负向条件特征。默认 True

        Returns:
            dict: 条件信息字典，包含 'cond' 和可选的 'neg_cond'
        """
        # 设置模型的分辨率参数，影响图像缩放尺寸和 patch token 数量
        self.image_cond_model.image_size = resolution

        # low_vram 模式下，临时将模型移到 GPU
        if self.low_vram:
            self.image_cond_model.to(self.device)

        # 前向传播：提取图像条件特征
        # 输出 cond 的形状为 (B, N, D)，其中 N 取决于 resolution/patch_size
        cond = self.image_cond_model(image)

        # low_vram 模式下，处理完后将模型移回 CPU 以节省显存
        if self.low_vram:
            self.image_cond_model.cpu()

        # 仅返回正向条件
        if not include_neg_cond:
            return {'cond': cond}

        # 生成负向条件特征（全零张量，用于 classifier-free guidance）
        # 与 cond 形状和数据类型相同
        neg_cond = torch.zeros_like(cond)

        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        根据图像条件生成物体的稀疏空间结构（论文 Sec. 3.3 的第一阶段）。

        这一阶段只回答“3D 空间中的哪些位置可能包含物体表面”，并不生成
        顶点位置、拓扑细节或 PBR 材质。返回的活跃坐标会成为后续 shape
        SLat flow 的 token 布局，使后续模型只在非空位置上进行稀疏计算。

        数据流：
            高斯噪声（dense sparse-structure latent）
              -> 图像条件 Sparse Structure Flow
              -> 去噪后的结构 latent z_s
              -> Sparse Structure Decoder
              -> 二值 occupancy grid
              -> 活跃体素坐标 [batch, x, y, z]

        注意这里有两种不同的分辨率：
        - ``flow_model.resolution``：结构 latent 的内部网格分辨率，例如 16。
        - ``resolution``：本函数最终返回的 occupancy 坐标分辨率，例如
          512 pipeline 使用 32，1024 pipeline 使用 64。

        官方 TRELLIS.2-4B 配置对应的具体实现：
        - ``flow_model`` 是 ``SparseStructureFlowModel``，定义于
          ``trellis2/models/sparse_structure_flow.py``。它把 16³ 网格展平为
          4096 个 token，通过 30 层带图像 cross-attention 的 DiT 预测
          flow velocity，再恢复为 3D 网格。
        - ``self.sparse_structure_sampler`` 是
          ``FlowEulerGuidanceIntervalSampler``，定义于
          ``trellis2/pipelines/samplers/flow_euler.py``。它组合了 Euler
          flow 积分、classifier-free guidance 和分时段 guidance。
        - 具体类名和默认参数并非硬编码在此处，而是由模型仓库中的
          ``pipeline.json`` 在 ``from_pretrained`` 中动态构造。因此换用其他
          checkpoint 时，sampler 配置理论上也可以改变。

        TRELLIS.2-4B 的结构 flow 典型配置为：
        ``resolution=16, in_channels=8, model_channels=1536,
        cond_channels=1024, num_blocks=30, num_heads=12``。
        
        Args:
            cond (dict): ``get_cond`` 生成的图像条件，通常包含 ``cond`` 和
                用于 classifier-free guidance 的 ``neg_cond``。
            resolution (int): 后续 shape SLat 所需的稀疏坐标网格分辨率。
            num_samples (int): 并行生成的 3D 样本数量。
            sampler_params (dict): 覆盖 pipeline 默认采样参数，例如采样步数
                和 CFG 强度。

        Returns:
            torch.Tensor: 形状为 ``(L, 4)`` 的整数坐标，其中每行为
                ``[batch_index, x, y, z]``，L 是所有样本的活跃体素总数。
        """
        # 论文 Sec. 3.3 stage 1：加载稀疏结构生成模型。
        # 它虽然名为 Sparse Structure Flow，但输入是在低分辨率规则网格上的
        # dense latent；真正的“稀疏性”要等 decoder 预测 occupancy 后才出现。
        # cast 只补充静态类型，不会复制、转换或重新加载模型。
        flow_model: "SparseStructureFlowModel" = cast(
            "SparseStructureFlowModel",
            self.models['sparse_structure_flow_model'],
        )
        sparse_structure_sampler: samplers.FlowEulerGuidanceIntervalSampler = cast(
            samplers.FlowEulerGuidanceIntervalSampler,
            self.sparse_structure_sampler,
        )
        reso = flow_model.resolution
        in_channels = flow_model.in_channels

        # 从标准高斯噪声开始 flow-matching 采样。
        # 例如公开训练配置中，张量形状为 (B, 8, 16, 16, 16)：
        # 8 是结构 latent 通道数，16³ 是 flow model 的内部 latent 网格。
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)

        # 调用时传入的参数优先级更高，可覆盖 pipeline.json 中的默认值。
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)

        # Flow Euler sampler 从 t=1 的噪声逐步积分到 t=0 的数据分布。
        # flow_model 在每一步都通过 cross-attention 使用 DINO 图像特征；
        # 若 sampler 支持 CFG，还会结合 cond / neg_cond 引导结构贴合输入图像。
        # z_s 仍是连续 latent，而不是 0/1 occupancy。
        z_s = sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        # 用结构 VAE decoder 将连续 latent 解码为空间 occupancy logits。
        # decoder 输出通常为 (B, 1, D, H, W)；logit > 0 等价于概率 > 0.5，
        # 表示该位置是活跃体素，后续需要在这里生成几何 SLat token。
        decoder: "SparseStructureDecoder" = cast(
            "SparseStructureDecoder",
            self.models['sparse_structure_decoder'],
        )
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0
        if self.low_vram:
            decoder.cpu()

        # decoder 的原始 occupancy 分辨率可能高于当前 pipeline 所需分辨率。
        # 使用 max pooling 保留一个 block 中的任意活跃位置：
        # 只要高分辨率子体素中至少有一个为真，对应低分辨率体素就保持活跃。
        # 这比平均池化更不容易漏掉细小、稀疏的结构。
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5

        # argwhere 对 (B, C, X, Y, Z) 返回 [batch, channel, x, y, z]。
        # occupancy 只有一个 channel，因此丢弃 channel 列，得到 SparseTensor
        # 所需的 [batch, x, y, z] 坐标。这里不返回 dense occupancy，可避免
        # 后续 shape/material 阶段在大量空体素上浪费计算。
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model: "ElasticSLatFlowModel",
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        在给定稀疏坐标上生成形状 Structured Latent（论文 Sec. 3.3 第二阶段）。

        SLat（Structured Latent）可以理解为“带明确 3D 坐标的稀疏 latent
        token 集合”。它由两部分组成：

        - ``coords``：每个 token 位于 3D latent 网格的什么位置；
        - ``feats``：该位置上的连续 latent feature，官方 shape 模型为 32 维。

        上一阶段只预测“哪里存在物体”，本阶段进一步预测“这些位置附近的
        几何具体长什么样”。生成出的 feature 仍是 SC-VAE 学习到的隐变量，
        并不是可直接解释的顶点、法线或占用值。后续 ``decode_shape_slat``
        会使用 Shape SC-VAE decoder 将其展开为高分辨率 O-Voxel 几何，
        包括 dual vertex、edge intersection flags 和 quad splitting weight。

        采样过程中，稀疏坐标始终保持不变，flow matching 只更新 feature：

            SparseTensor(
                coords=(L, 4),       # 固定的 [batch, x, y, z]
                feats=(L, 32),       # 高斯噪声 -> shape latent
            )

        其中 L 是当前 batch 中所有活跃 token 的总数。由于只处理这些活跃
        位置，而不是完整的 32³/64³ 网格，这一阶段使用 Sparse DiT。

        官方 512/1024 shape flow 均使用 ``ElasticSLatFlowModel``：
        32 维输入经过 SparseLinear 投影到 1536 维，再通过 30 个 sparse
        Transformer block。模型利用 self-attention 交换 3D token 信息，
        通过 cross-attention 读取 DINO 图像条件，并通过 timestep embedding
        和 3D RoPE 感知 flow 时间与空间位置。
        
        Args:
            cond (dict): ``get_cond`` 得到的图像条件，通常包含 ``cond`` 和
                CFG 使用的 ``neg_cond``。
            flow_model (ElasticSLatFlowModel): 当前分辨率对应的 shape SLat
                flow model。512³ 输出使用 32³ latent 网格，1024³ 输出使用
                64³ latent 网格。
            coords (torch.Tensor): 形状为 ``(L, 4)`` 的活跃坐标，每行为
                ``[batch_index, x, y, z]``，来自 ``sample_sparse_structure``。
            sampler_params (dict): 覆盖默认 shape SLat 采样参数，例如 steps、
                guidance_strength、guidance_rescale 和 rescale_t。

        Returns:
            SparseTensor: 已反归一化的 shape SLat。坐标仍为 ``(L, 4)``，
                feature 通常为 ``(L, 32)``，可直接交给 Shape SC-VAE decoder。
        """
        # 为每个已确定的活跃坐标初始化一个独立的高斯噪声 feature。
        # 注意：这里不会再生成或删除坐标；flow 只让 noise.feats 从随机噪声
        # 演化为有意义的 shape latent feature。
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )

        # 调用时参数覆盖 pipeline.json 中 shape SLat sampler 的默认参数。
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)

        # 官方 checkpoint 通常使用 FlowEulerGuidanceIntervalSampler。
        # sampler 从 t=1 的高斯噪声沿预测的 flow velocity 积分到 t=0；
        # 每一步中 Sparse DiT 都同时读取当前 noisy features、3D coords、
        # timestep 和图像条件。返回值仍保留输入 SparseTensor 的坐标布局。
        shape_slat_sampler: samplers.FlowEulerGuidanceIntervalSampler = cast(
            samplers.FlowEulerGuidanceIntervalSampler,
            self.shape_slat_sampler,
        )
        slat = shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        # Shape flow 训练时使用标准化后的 SC-VAE latent：
        #     normalized = (latent - mean) / std
        # 因而 sampler 生成的也是标准化空间中的 feature。交给 Shape SC-VAE
        # decoder 前，需要逐通道执行反归一化恢复原始 latent 分布。
        # mean/std 的长度与 latent channel 数一致，官方模型中均为 32。
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        将 shape SLat 解码为高分辨率 O-Voxel 几何并提取 Mesh。

        这一步对应论文 Sec. 3.2 的 Shape SC-VAE decoder。输入 ``slat`` 是
        低分辨率、每个活跃位置带 32 维 feature 的 Structured Latent；
        decoder 通过多级稀疏卷积与 4 次 2x 上采样，将空间分辨率放大 16 倍：

            32³ shape SLat -> 512³ O-Voxel
            64³ shape SLat -> 1024³ O-Voxel

        每次上采样前，decoder 都会预测父 token 的 8 个子位置中哪些应继续
        保持活跃（论文 Fig. 4 的 early-pruning upsampler）。因此它不只是
        放大 feature，也会逐层细化和扩展稀疏空间结构。

        decoder 最终为每个高分辨率活跃 O-Voxel 预测 7 个几何量：

        - 3 维 dual vertex position；
        - 3 维 X/Y/Z edge intersection flags；
        - 1 维 quad splitting weight。

        ``FlexiDualGridVaeDecoder`` 随后调用 O-Voxel 的 Flexible Dual Grid
        转换，把这些局部几何量直接连接成 Mesh。

        除 Mesh 外，这里通过 ``return_subs=True`` 保留每一级上采样预测出的
        subdivision structure。后续材质 decoder 使用这些 ``subs`` 作为
        ``guide_subs``，让 PBR 属性严格沿用 shape decoder 已确定的多尺度
        活跃体素结构，从而保持几何与材质的空间对齐。

        Args:
            slat (SparseTensor): 已反归一化的 shape SLat。其 ``coords`` 为
               低分辨率活跃坐标，``feats`` 通常为 32 维 shape latent。
            resolution (int): 最终 O-Voxel 网格和 Mesh 提取使用的目标分辨率，
               例如 512、1024 或级联推理得到的 1536。

        Returns:
            Tuple[List[Mesh], List[SparseTensor]]:
                - ``meshes``：batch 中每个样本对应的 Flexible Dual Grid Mesh；
                - ``subs``：各上采样层的 subdivision masks/structures，供
                  ``decode_tex_slat`` 引导材质 decoder 使用。
        """
        # 从模型字典提取具名对象并补充精确类型，方便 IDE 追踪到
        # trellis2/models/sc_vaes/fdg_vae.py::FlexiDualGridVaeDecoder。
        # cast 仅影响静态类型分析，不会复制或重新实例化模型。
        shape_slat_decoder: "FlexiDualGridVaeDecoder" = cast(
            "FlexiDualGridVaeDecoder",
            self.models['shape_slat_decoder'],
        )

        # decoder 可跨分辨率复用；该值会传给 Flexible Dual Grid 的 mesh
        # 提取逻辑，用于解释 voxel 坐标及设置最终网格大小。
        shape_slat_decoder.set_resolution(resolution)

        if self.low_vram:
            shape_slat_decoder.to(self.device)
            # decoder 内部也采用分段迁移策略，降低多级稀疏上采样的显存峰值。
            shape_slat_decoder.low_vram = True

        # 推理模式下返回 (meshes, subs)：
        # 先从 SLat 恢复 7 通道 O-Voxel 几何，再由 Flexible Dual Grid
        # 立即提取 Mesh；subs 同时记录每一级上采样选中的活跃子体素。
        meshes, subs = shape_slat_decoder(slat, return_subs=True)

        if self.low_vram:
            shape_slat_decoder.cpu()
            shape_slat_decoder.low_vram = False

        return meshes, subs
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params
        )
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
