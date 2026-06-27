# TRELLIS.2 512³ 前向推理流程

本文以 `Trellis2ImageTo3DPipeline.run(..., pipeline_type="512")` 为例，说明一张输入图片如何经过 TRELLIS.2，最终变成带 PBR 材质的 3D 资产。

对应的主要代码入口：

- 推理总入口：[`trellis2/pipelines/trellis2_image_to_3d.py`](trellis2/pipelines/trellis2_image_to_3d.py)
- 稀疏结构 Flow：[`trellis2/models/sparse_structure_flow.py`](trellis2/models/sparse_structure_flow.py)
- Shape/Texture SLat Flow：[`trellis2/models/structured_latent_flow.py`](trellis2/models/structured_latent_flow.py)
- Shape SC-VAE decoder：[`trellis2/models/sc_vaes/fdg_vae.py`](trellis2/models/sc_vaes/fdg_vae.py)
- 通用稀疏 SC-VAE：[`trellis2/models/sc_vaes/sparse_unet_vae.py`](trellis2/models/sc_vaes/sparse_unet_vae.py)
- Flow Euler sampler：[`trellis2/pipelines/samplers/flow_euler.py`](trellis2/pipelines/samplers/flow_euler.py)
- 稀疏张量实现：[`trellis2/modules/sparse/basic.py`](trellis2/modules/sparse/basic.py)

## 1. 一句话理解整条流水线

TRELLIS.2 并不是从图片直接回归 Mesh，而是分三次生成：

1. 生成稀疏结构：确定物体大致占据 3D 空间中的哪些位置。
2. 生成 Shape SLat：为这些位置生成压缩后的局部几何描述。
3. 生成 Texture SLat：在相同空间结构上生成压缩后的 PBR 材质描述。

最后，Shape SC-VAE 和 Texture SC-VAE 分别解码几何与材质，并组合成 `MeshWithVoxel`。

```text
输入图片
  │
  ├─ preprocess_image
  │    去背景、裁剪、预乘 alpha
  │
  ├─ DINOv3 get_cond
  │    输出图像 patch tokens: cond_512
  │
  ├─ Sparse Structure Flow
  │    生成 16³ dense structure latent
  │
  ├─ Sparse Structure Decoder
  │    解码为 occupancy，再提取 32³ 活跃坐标 coords
  │
  ├─ Shape SLat Flow
  │    在 coords 上生成 32 维 shape latent
  │
  ├─ Texture SLat Flow
  │    条件于图片和 shape SLat，生成 32 维 material latent
  │
  ├─ Shape SC-VAE Decoder
  │    32³ SLat → 512³ O-Voxel geometry → Mesh
  │
  ├─ Texture SC-VAE Decoder
  │    32³ SLat → 512³ PBR voxel attributes
  │
  └─ MeshWithVoxel
       Mesh + Base Color/Metallic/Roughness/Alpha 属性体素
```

### 1.1 各模块的模型来源与是否需要重新训练

正常使用官方 `microsoft/TRELLIS.2-4B` 进行推理时，下面所有神经网络权重都会由
`Trellis2ImageTo3DPipeline.from_pretrained(...)` 自动加载，**不需要先在当前仓库
重新训练**。当前仓库提供训练代码的意义，是允许研究者从头训练、复现或在自定义
数据上微调其中一部分模型。

| 推理模块 | 具体模型/权重来源 | 由哪个项目训练 | 当前仓库是否提供训练代码 | 官方推理时如何使用 |
| --- | --- | --- | --- | --- |
| 去背景 | `ZhengPeng7/BiRefNet` | 外部 BiRefNet 项目 | 否 | 直接加载外部预训练权重；输入已有有效 alpha 时可以跳过 |
| 图像特征 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Meta DINOv3 项目 | 否 | 直接加载并冻结，仅提取图片条件 token |
| Sparse Structure Flow | `microsoft/TRELLIS.2-4B` 中的 structure flow checkpoint | TRELLIS.2 | 是：`configs/gen/ss_flow_img_dit_1_3B_64_bf16.json` | 直接加载官方 checkpoint；只有复现或适配新数据时才重新训练 |
| Sparse Structure Decoder | `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16` | 前代 TRELLIS / `TRELLIS-image-large` | 代码和 Trainer 存在，但本仓库没有配套训练配置与原始 occupancy Dataset | 作为固定的预训练 decoder 直接复用，不属于 TRELLIS.2 标准重训流程 |
| Shape SLat Flow | `microsoft/TRELLIS.2-4B` 中的 512 shape flow checkpoint | TRELLIS.2 | 是：`configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json` | 直接加载官方 checkpoint，或用当前仓库重新训练/微调 |
| Texture SLat Flow | `microsoft/TRELLIS.2-4B` 中的 512 texture flow checkpoint | TRELLIS.2 | 是：`configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json` | 直接加载官方 checkpoint，或用当前仓库重新训练/微调 |
| Shape SC-VAE Decoder | `microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16` | TRELLIS.2 | 是：`configs/scvae/shape_vae_next_dc_f16c32_fp16*.json` | 推理时直接加载 decoder；训练数据编码还会使用配套 encoder |
| Texture SC-VAE Decoder | `microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16` | TRELLIS.2 | 是：`configs/scvae/tex_vae_next_dc_f16c32_fp16*.json` | 推理时直接加载 decoder；训练数据编码还会使用配套 encoder |
| Flow Euler sampler | 当前仓库中的确定性采样算法，没有可学习参数 | TRELLIS.2 代码实现 | 不需要训练 | 根据 `pipeline.json` 创建并直接使用 |
| O-Voxel / Flexible Dual Grid | 当前仓库的 `o-voxel` 子项目和 CUDA/C++ 扩展，没有神经网络权重 | TRELLIS.2 / O-Voxel | 不需要训练 | 直接完成 O-Voxel 与 Mesh 的转换 |

这里存在一个容易混淆的历史依赖：

```text
TRELLIS-image-large
  └─ Sparse Structure Encoder / Decoder
       ├─ encoder：离线把 32³/64³ occupancy 编码成 16³ × 8 SS latent
       └─ decoder：推理时把 Structure Flow 生成的 latent 解码回 occupancy

TRELLIS.2
  ├─ Sparse Structure Flow
  ├─ Shape SC-VAE + Shape SLat Flow
  └─ Texture SC-VAE + Texture SLat Flow
```

也就是说，TRELLIS.2 重新训练了生成 Flow、Shape SC-VAE 和 Texture SC-VAE，
但第一阶段使用的 Structure VAE 沿用了前代 TRELLIS 的预训练表示空间。当前仓库的
`data_toolkit/encode_ss_latent.py` 默认调用的也正是前代 Structure Encoder：

```text
microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16
```

它将 Shape SLat 的二值坐标布局编码成 Sparse Structure Flow 的训练目标。

如果只是使用官方模型，推荐路径是：

```python
pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
    "microsoft/TRELLIS.2-4B"
)
```

如果希望从头复现或在自定义数据上训练，则依赖顺序是：

```text
1. 直接复用前代 Structure VAE encoder/decoder
2. 训练 TRELLIS.2 Shape SC-VAE
3. 训练 TRELLIS.2 Texture SC-VAE
4. 用三个 encoder 离线生成 SS / Shape / Texture latent 数据
5. 训练 Sparse Structure Flow
6. 训练 Shape SLat Flow
7. 训练 Texture SLat Flow
```

其中第 1 步在当前公开训练流程中不是重新训练，而是下载并固定使用
`TRELLIS-image-large` 的 Structure VAE 权重。

## 2. `run()` 的 512 分支

512 分辨率的核心代码可以简化为：

```python
image = self.preprocess_image(image)
torch.manual_seed(seed)

cond_512 = self.get_cond([image], 512)

coords = self.sample_sparse_structure(
    cond_512,
    resolution=32,
    num_samples=num_samples,
)

shape_slat = self.sample_shape_slat(
    cond_512,
    self.models["shape_slat_flow_model_512"],
    coords,
)

tex_slat = self.sample_tex_slat(
    cond_512,
    self.models["tex_slat_flow_model_512"],
    shape_slat,
)

out_mesh = self.decode_latent(
    shape_slat,
    tex_slat,
    resolution=512,
)
```

这里需要始终区分三种分辨率：

| 表示 | 分辨率 | 含义 |
| --- | ---: | --- |
| Sparse Structure Flow latent | `16³` | 第一阶段使用的低分辨率 dense latent |
| Shape/Texture SLat | `32³` | 稀疏结构化 latent 所在的网格 |
| 最终 O-Voxel | `512³` | 几何和材质属性所在的高分辨率网格 |

Shape/Texture SC-VAE 在空间上压缩了 16 倍，因此：

```text
512 / 16 = 32
```

## 3. 输入与随机状态

### 3.1 输入参数

典型调用：

```python
outputs = pipeline.run(
    image,
    pipeline_type="512",
    num_samples=1,
    seed=42,
)
```

主要参数：

| 参数 | 作用 |
| --- | --- |
| `image` | 输入的 PIL 图片 |
| `num_samples` | 从同一图片并行采样多少个 3D 结果 |
| `seed` | 控制三个生成阶段的初始噪声 |
| `preprocess_image` | 是否自动去背景和裁剪 |
| `return_latent` | 是否同时返回 shape/texture SLat |
| 三组 `sampler_params` | 分别覆盖结构、形状、材质 sampler 的默认参数 |

### 3.2 随机种子

```python
torch.manual_seed(seed)
```

后续三个阶段都调用 `torch.randn`：

- 稀疏结构 Flow 的 dense 高斯噪声；
- Shape SLat Flow 的稀疏高斯噪声；
- Texture SLat Flow 的稀疏高斯噪声。

因此相同输入、相同权重、相同 sampler 参数和相同 seed 通常会得到相同结果。

## 4. 图像预处理：`preprocess_image`

输入：

```text
PIL.Image
```

主要计算：

1. 如果输入没有有效 alpha，调用背景移除模型。
2. 将最大边限制在 1024 像素。
3. 根据 alpha 前景区域计算边界框。
4. 以前景中心进行正方形裁剪。
5. 执行预乘 alpha：

```text
output_rgb = rgb × alpha
```

透明区域因此变为黑色。

输出：

```text
image: PIL RGB Image
```

这一步只规范输入图片，不产生任何 3D 信息。

## 5. 图像条件提取：`get_cond`

调用：

```python
cond_512 = self.get_cond([image], resolution=512)
```

### 5.1 模块

官方模型使用 `DinoV3FeatureExtractor`，底层是 DINOv3 ViT-L。

### 5.2 计算过程

1. 将图片缩放到 `512 × 512`。
2. 转换为 `(B, 3, 512, 512)` Tensor。
3. 使用 ImageNet mean/std 标准化。
4. 经过 DINOv3 patch embedding 和 Transformer layers。
5. 对输出 token 进行 LayerNorm。

### 5.3 中间变量

```python
cond_512 = {
    "cond": cond,
    "neg_cond": torch.zeros_like(cond),
}
```

其语义为：

```text
cond:     (B, N_image, 1024)
neg_cond: (B, N_image, 1024)
```

- `N_image` 是 DINOv3 输出的图像 token 数量。
- `1024` 是 DINOv3-L 的 feature 维度。
- `cond` 在三个 Flow 模型中通过 cross-attention 提供图片条件。
- `neg_cond` 是全零条件，用于 classifier-free guidance（CFG）。

至此，图片已经被转换成所有生成阶段共享的视觉 token。

## 6. 第一阶段：生成稀疏结构

调用：

```python
coords = self.sample_sparse_structure(
    cond_512,
    resolution=32,
    num_samples=num_samples,
)
```

这一阶段只回答：

> 物体表面大致位于 32³ 网格中的哪些位置？

它不生成局部顶点、边连接、细节或材质。

### 6.1 初始噪声

官方 Sparse Structure Flow 配置：

```text
resolution     = 16
in_channels    = 8
out_channels   = 8
model_channels = 1536
num_blocks     = 30
num_heads      = 12
```

创建：

```python
noise = torch.randn(B, 8, 16, 16, 16)
```

这是一个低分辨率 dense 3D latent，而不是最终 occupancy。

### 6.2 `SparseStructureFlowModel`

模型首先将空间展平：

```text
(B, 8, 16, 16, 16)
    ↓ flatten + permute
(B, 4096, 8)
```

然后：

```text
Linear: 8 → 1536
30 × ModulatedTransformerCrossBlock
Linear: 1536 → 8
恢复 16³ 网格
```

每个 Transformer block 使用：

- self-attention：让不同 3D 位置交换信息；
- cross-attention：读取 `cond_512` 图像 token；
- timestep embedding + AdaLN：指示当前 flow 时间；
- 3D RoPE：编码 token 的三维位置。

### 6.3 Flow Euler 采样

sampler 通常为 `FlowEulerGuidanceIntervalSampler`。

它从 `t=1` 的噪声开始，多次调用 Flow 模型预测 velocity：

```text
v_t = flow_model(x_t, t, image_condition)
```

并执行 Euler 更新：

```text
x_t_prev = x_t - (t - t_prev) × v_t
```

CFG 会组合正条件和空条件预测：

```text
v_cfg = guidance_strength × v_cond
      + (1 - guidance_strength) × v_uncond
```

输出：

```python
z_s
```

典型形状：

```text
z_s: (B, 8, 16, 16, 16)
```

`z_s` 是连续的 sparse-structure latent，尚不是二值结构。

### 6.4 Sparse Structure Decoder

```python
decoded = sparse_structure_decoder(z_s) > 0
```

decoder 将 `16³ × 8 channels` latent 解码为更高分辨率 occupancy logits，并用零作为分类阈值：

```text
logit > 0  → active
logit ≤ 0  → empty
```

如果 decoder 输出高于当前需要的 `32³`，代码使用 max pooling 降采样：

```python
decoded = max_pool3d(decoded.float(), ratio) > 0.5
```

max pooling 的语义是：

> 一个低分辨率格子覆盖的高分辨率子格中，只要有一个活跃，就保留该格子。

最后提取所有活跃位置：

```python
coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
```

输出：

```text
coords: (L, 4)
```

每行是：

```text
[batch_index, x, y, z]
```

`L` 是当前 batch 中的活跃位置总数，随物体复杂度变化。

## 7. 第二阶段：生成 Shape SLat

调用：

```python
shape_slat = self.sample_shape_slat(
    cond_512,
    shape_slat_flow_model_512,
    coords,
)
```

这一阶段回答：

> 已知物体大致位于这些位置，每个位置附近的具体几何应该是什么？

### 7.1 SLat 是什么

SLat 是 Structured Latent，即带有明确 3D 坐标的稀疏 latent token：

```python
SparseTensor(
    coords=(L, 4),
    feats=(L, C),
)
```

它的“Structured”来自每个 feature 都绑定了 `(x, y, z)` 坐标；它的“Sparse”来自只存储活跃位置。

### 7.2 Shape SLat 初始噪声

官方 Shape Flow 的 latent channel 为 32：

```python
shape_noise = SparseTensor(
    coords=coords,
    feats=torch.randn(L, 32),
)
```

采样过程中：

- `coords` 始终不变；
- 只有 `feats` 从高斯噪声逐步变成 shape latent。

### 7.3 `ElasticSLatFlowModel`

官方 512 Shape Flow 配置：

```text
resolution     = 32
in_channels    = 32
out_channels   = 32
model_channels = 1536
num_blocks     = 30
num_heads      = 12
```

数据流：

```text
SparseTensor feats: (L, 32)
    ↓ SparseLinear
(L, 1536)
    ↓ 30 × ModulatedSparseTransformerCrossBlock
(L, 1536)
    ↓ LayerNorm + SparseLinear
(L, 32)
```

每一步模型同时读取：

- 当前 noisy shape features；
- 每个 token 的 3D 坐标；
- flow timestep；
- `cond_512` 图像特征。

Sparse Transformer 只处理 L 个活跃 token，不处理完整的 `32³=32768` 网格。

### 7.4 Shape latent 反归一化

Shape Flow 的训练目标是标准化后的 SC-VAE latent：

```text
normalized_shape = (raw_shape - mean_shape) / std_shape
```

因此采样结束后执行：

```python
shape_slat = shape_slat * std_shape + mean_shape
```

最终：

```text
shape_slat.coords: (L, 4)
shape_slat.feats:  (L, 32)
```

这 32 个通道是 SC-VAE 学到的压缩特征，不应逐通道解释成顶点或法线。只有经过 Shape SC-VAE decoder 后，它们才会变成具有明确语义的 O-Voxel 几何量。

## 8. 第三阶段：生成 Texture SLat

调用：

```python
tex_slat = self.sample_tex_slat(
    cond_512,
    tex_slat_flow_model_512,
    shape_slat,
)
```

这一阶段回答：

> 已知图片和已经生成的几何，物体表面各处应该具有什么 PBR 材质？

### 8.1 Shape SLat 重新标准化

上一阶段为了交给 decoder，已经把 shape SLat 恢复到了原始 SC-VAE latent 分布。

Texture Flow 训练时使用的是标准化 shape latent，因此先执行：

```python
normalized_shape_slat = (shape_slat - mean_shape) / std_shape
```

坐标保持不变。

### 8.2 材质噪声

官方 Texture Flow 配置：

```text
resolution     = 32
in_channels    = 64
out_channels   = 32
model_channels = 1536
num_blocks     = 30
num_heads      = 12
```

这里的 `in_channels=64` 容易误解。

模型真正生成的 Texture SLat 仍然只有 32 维。64 维输入由两部分拼接：

```text
32 维 texture noise
32 维 normalized shape SLat
------------------------------
64 维 Texture Flow 输入
```

代码先创建：

```python
texture_noise = shape_slat.replace(
    feats=torch.randn(L, 32)
)
```

`replace` 会保留 shape SLat 的：

- `coords`
- batch layout
- 稀疏结构缓存

只替换 feature。

### 8.3 条件拼接

调用 sampler 时：

```python
tex_slat_sampler.sample(
    flow_model,
    texture_noise,
    concat_cond=normalized_shape_slat,
    **cond_512,
)
```

在 `SLatFlowModel.forward()` 中：

```python
x = sparse_cat([x, concat_cond], dim=-1)
```

于是：

```text
x.feats:           (L, 32) texture noisy feature
concat_cond.feats: (L, 32) shape feature
拼接结果:           (L, 64)
```

Texture Flow 同时受到两类条件约束：

1. 图像条件：通过 cross-attention 读取 `cond_512`；
2. 几何条件：在每个 3D token 上直接拼接 shape SLat。

这使材质既能匹配输入图片，也能与生成出的几何结构对齐。

### 8.4 Texture SLat 输出与反归一化

Flow 输出：

```text
normalized_tex_slat:
    coords: (L, 4)
    feats:  (L, 32)
```

然后恢复 Texture SC-VAE 的原始 latent 分布：

```python
tex_slat = tex_slat * std_tex + mean_tex
```

最终：

```text
tex_slat.coords: (L, 4)
tex_slat.feats:  (L, 32)
```

Shape SLat 和 Texture SLat 使用同一组低分辨率坐标，因此几何与材质从 latent 阶段起就是空间对齐的。

## 9. 解码 Shape SLat

`run()` 最后调用：

```python
out_mesh = self.decode_latent(shape_slat, tex_slat, resolution=512)
```

其中首先执行：

```python
meshes, subs = self.decode_shape_slat(shape_slat, 512)
```

### 9.1 Shape SC-VAE decoder

具体类为：

```text
FlexiDualGridVaeDecoder
```

输入：

```text
shape_slat:
    coords: (L, 4), 位于 32³ 网格
    feats:  (L, 32)
```

decoder 通过 4 次 2 倍稀疏上采样：

```text
32³ → 64³ → 128³ → 256³ → 512³
```

总空间放大倍率：

```text
2⁴ = 16
```

### 9.2 Early-pruning subdivision

每次上采样时，一个父 voxel 最多对应 `2×2×2=8` 个子 voxel。

decoder 会预测哪些子 voxel 应继续活跃：

```text
父 token
  ├─ child 0: active
  ├─ child 1: inactive
  ├─ ...
  └─ child 7: active
```

未激活的子 voxel 会被提前剪枝，从而避免在大量空空间上继续计算。

每一级预测出的 subdivision structure 被保存到：

```python
subs: List[SparseTensor]
```

`subs` 不仅服务于几何解码，稍后也会指导 Texture SC-VAE decoder 使用完全一致的多尺度稀疏结构。

### 9.3 7 通道 O-Voxel 几何

Shape decoder 最终输出每个高分辨率活跃 voxel 的 7 个通道：

```text
0:3  dual vertex position
3:6  X/Y/Z edge intersection flags
6:7  quad splitting weight
```

代码进一步施加输出变换：

- vertex position：使用 sigmoid 限制位置范围；
- intersection flags：使用 `> 0` 二值化；
- splitting weight：使用 softplus 保证为正。

### 9.4 Flexible Dual Grid 转 Mesh

`FlexiDualGridVaeDecoder` 调用：

```python
flexible_dual_grid_to_mesh(...)
```

主要过程：

1. 根据 edge intersection flags 判断哪些相邻 dual vertices 需要连接。
2. 在相交边周围连接四个 dual vertices，形成 quad。
3. 根据 splitting weight 将 quad 拆为两个三角形。

输出：

```text
meshes: List[Mesh]
```

每个 Mesh 包含：

```text
vertices: (V, 3)
faces:    (F, 3)
```

## 10. 解码 Texture SLat

调用：

```python
tex_voxels = self.decode_tex_slat(tex_slat, subs)
```

具体 decoder 为：

```text
SparseUnetVaeDecoder
```

输入：

```text
tex_slat:
    coords: (L, 4), 位于 32³ 网格
    feats:  (L, 32)
```

### 10.1 使用 shape subdivision

```python
tex_slat_decoder(
    tex_slat,
    guide_subs=subs,
)
```

材质 decoder 自己不重新判断每一级哪些体素应活跃，而是直接沿用 shape decoder 生成的 `subs`：

```text
Shape decoder 的活跃子体素结构
              ↓
Texture decoder 在完全相同位置恢复材质
```

这样可以避免几何表面与材质属性位于不同 voxel 位置。

### 10.2 输出 PBR 属性

Texture decoder 最终输出 6 个通道：

```text
0:3  Base Color RGB
3:4  Metallic
4:5  Roughness
5:6  Alpha
```

decoder 原始输出以大约 `[-1, 1]` 为目标范围，因此代码执行：

```python
tex_voxels = decoder_output * 0.5 + 0.5
```

将其转换到约 `[0, 1]` 的 PBR 属性范围。

输出：

```text
tex_voxels: SparseTensor
    coords: (K, 4), 位于 512³ 网格
    feats:  (K, 6)
```

`K` 是高分辨率材质活跃 voxel 数量。

## 11. 组合最终 `MeshWithVoxel`

`decode_latent()` 对每个样本组合 Mesh 和 PBR voxel：

```python
for mesh, voxel in zip(meshes, tex_voxels):
    mesh.fill_holes()

    output = MeshWithVoxel(
        vertices=mesh.vertices,
        faces=mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / 512,
        coords=voxel.coords[:, 1:],
        attrs=voxel.feats,
        voxel_shape=...,
        layout={
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        },
    )
```

### 11.1 几何部分

```text
vertices: Mesh 顶点
faces:    Mesh 三角形
```

在封装前，`fill_holes()` 会尝试修复较小的边界孔洞。

### 11.2 材质部分

```text
coords: 512³ 网格中的活跃材质 voxel 坐标
attrs:  每个 voxel 的 6 维 PBR 属性
```

空间定义：

```text
origin     = (-0.5, -0.5, -0.5)
voxel_size = 1 / 512
```

因此整个资产位于单位立方体：

```text
[-0.5, 0.5]³
```

### 11.3 为什么不是直接输出 UV 贴图

`MeshWithVoxel` 暂时保留体素形式的 PBR 属性。

渲染或导出时，可以在任意 3D 表面位置通过三线性插值查询：

```python
attrs = mesh.query_attrs(xyz)
```

之后 `o_voxel.postprocess.to_glb()` 可以执行重网格化、UV 展开和纹理烘焙，最终导出常规 GLB。

## 12. 最终返回值

默认：

```python
out_mesh = pipeline.run(..., pipeline_type="512")
```

返回：

```text
List[MeshWithVoxel]
```

列表长度通常等于 `num_samples`。

如果：

```python
return_latent=True
```

则返回：

```python
(
    out_mesh,
    (
        shape_slat,
        tex_slat,
        512,
    ),
)
```

这对于调试非常有用，可以分别观察：

- `shape_slat.coords` 与 `shape_slat.feats`；
- `tex_slat.coords` 与 `tex_slat.feats`；
- 最终采用的输出分辨率。

## 13. 中间变量总表

下表以单样本为例。`L` 和 `K` 取决于物体的空间复杂度。

| 变量 | 类型/典型形状 | 所在网格 | 含义 |
| --- | --- | ---: | --- |
| `image` | PIL Image | 2D | 去背景并裁剪后的输入 |
| `cond_512["cond"]` | `(1, N_image, 1024)` | 2D token | DINOv3 图像条件 |
| `structure_noise` | `(1, 8, 16, 16, 16)` | `16³` | 结构 Flow 初始噪声 |
| `z_s` | `(1, 8, 16, 16, 16)` | `16³` | 生成出的连续结构 latent |
| `decoded` | `(1, 1, 32, 32, 32)` 左右 | `32³` | 二值 occupancy |
| `coords` | `(L, 4)` | `32³` | 活跃 SLat 坐标 |
| `shape_noise` | SparseTensor, feats `(L, 32)` | `32³` | Shape Flow 初始噪声 |
| `shape_slat` | SparseTensor, feats `(L, 32)` | `32³` | 反归一化 shape latent |
| `texture_noise` | SparseTensor, feats `(L, 32)` | `32³` | Texture Flow 初始噪声 |
| Texture Flow 拼接输入 | feats `(L, 64)` | `32³` | texture noise + normalized shape |
| `tex_slat` | SparseTensor, feats `(L, 32)` | `32³` | 反归一化 material latent |
| `subs` | List[SparseTensor] | `64³...512³` | Shape decoder 多尺度细分结构 |
| `meshes` | List[Mesh] | 连续 3D | Flexible Dual Grid 提取的网格 |
| `tex_voxels` | SparseTensor, feats `(K, 6)` | `512³` | Base Color/Metallic/Roughness/Alpha |
| `out_mesh` | List[MeshWithVoxel] | 连续 3D + `512³` | 最终几何与 PBR 属性 |

## 14. 模型之间的依赖关系

从依赖角度，512 推理不是几个互相独立的模型：

```text
DINO image features
  ├─────────────────────────────┐
  │                             │
  ▼                             │
Sparse Structure Flow           │
  │ coords                      │
  ▼                             │
Shape SLat Flow ◄───────────────┤
  │ shape_slat                  │
  ├───────────────┐             │
  │               │             │
  ▼               ▼             │
Shape Decoder   Texture Flow ◄──┘
  │ mesh, subs     │ tex_slat
  │                ▼
  └──────────► Texture Decoder
                   │ PBR voxels
                   ▼
              MeshWithVoxel
```

关键依赖：

1. Shape Flow 依赖 Sparse Structure Flow 生成的 `coords`。
2. Texture Flow 依赖 Shape Flow 生成的 `shape_slat`。
3. Texture Decoder 依赖 Shape Decoder 生成的 `subs`。
4. 三个 Flow 都依赖同一份 DINO 图像条件。

这体现了 TRELLIS.2 的生成顺序：

```text
先确定空间布局，再确定几何，最后在确定的几何上生成材质。
```

## 15. 推荐的调试观察点

阅读或实际运行时，可以在下列位置打印中间变量：

```python
print("cond", cond_512["cond"].shape)

print("coords", coords.shape)
print("coords range", coords[:, 1:].min(0).values, coords[:, 1:].max(0).values)

print("shape slat coords", shape_slat.coords.shape)
print("shape slat feats", shape_slat.feats.shape)
print("shape slat stats", shape_slat.feats.mean(), shape_slat.feats.std())

print("tex slat feats", tex_slat.feats.shape)
print("tex slat stats", tex_slat.feats.mean(), tex_slat.feats.std())

print("mesh", out_mesh[0].vertices.shape, out_mesh[0].faces.shape)
print("pbr voxels", out_mesh[0].coords.shape, out_mesh[0].attrs.shape)
```

最有价值的检查通常是：

- `coords.shape[0]`：低分辨率 token 数量；
- `shape_slat.feats.shape[-1] == 32`；
- `tex_slat.feats.shape[-1] == 32`；
- 最终 `attrs.shape[-1] == 6`；
- PBR 属性是否大致位于 `[0, 1]`；
- `subs` 各层的 token 数如何随上采样增长。

## 16. 最核心的心智模型

如果只记住一个版本，可以记成：

```text
图片
  → DINO token
  → 生成“哪里有东西”
  → 在那些位置生成“几何 latent”
  → 条件于几何生成“材质 latent”
  → 两套 SC-VAE 将 latent 展开 16 倍
  → O-Voxel 几何和 PBR 属性
  → MeshWithVoxel
```

其中：

- Sparse Structure 是位置集合；
- Shape SLat 是带坐标的压缩几何；
- Texture SLat 是带坐标的压缩材质；
- O-Voxel 是可直接转换为 Mesh、同时承载 PBR 属性的高分辨率原生 3D 表示。
