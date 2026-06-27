# TRELLIS.1 与 TRELLIS.2 的 Sparse Structure Generation 对比

本文专门比较 TRELLIS.1 和 TRELLIS.2 在第一阶段 **Sparse Structure Generation**
上的继承关系和主要差异。这里的 Sparse Structure Generation 指从图像条件生成
稀疏空间结构的阶段，它决定后续高分辨率几何/材质生成在哪些 3D 位置上展开。

核心结论：

- **Sparse Structure VAE 表示空间基本复用**：TRELLIS.2 仍使用前代
  `TRELLIS-image-large` 的 sparse structure encoder/decoder 家族。
- **条件生成模型重新训练并升级**：TRELLIS.2 的 Sparse Structure Flow 是新的
  `microsoft/TRELLIS.2-4B` checkpoint，不是直接沿用 TRELLIS.1 的 flow。
- **图像条件、模型规模、训练 recipe 和数据预处理流程都有变化**：最明显的是
  DINOv2 换成 DINOv3，flow backbone 也从 TRELLIS.1 的 large 级别扩展到
  TRELLIS.2 的 1.3B 级别配置。

## 1. 第一阶段在两代系统中的位置

两代系统都不是从图片直接回归最终 mesh，而是先生成一个较粗的空间结构：

```text
input image
  -> image encoder tokens
  -> Sparse Structure Flow
  -> 16^3 x 8 sparse-structure latent
  -> Sparse Structure Decoder
  -> 64^3 occupancy / active spatial structure
  -> downstream 3D latent generation
```

这个阶段的作用是回答“物体大概占据哪些空间单元”。后续模型再在这些位置上生成
更细的几何、外观或材质 latent。

TRELLIS.1 的后续表示是原始 Structured Latent，支持 Gaussian、Radiance Field、
Mesh 等 decoder。TRELLIS.2 后续改为面向 O-Voxel / Flexible Dual Grid 的
Shape SC-VAE 和 Texture SC-VAE，因此后半段系统已经明显不同。但第一阶段的
Sparse Structure VAE 表示仍然和 TRELLIS.1 保持兼容。

## 2. 哪些部分是复用的

### 2.1 Sparse Structure VAE 复用

TRELLIS.1 的 sparse structure VAE 配置是：

```text
configs/vae/ss_vae_conv3d_16l8_fp16.json
```

它的关键结构是：

| 项目 | TRELLIS.1 Sparse Structure VAE |
| --- | --- |
| encoder | `SparseStructureEncoder` |
| decoder | `SparseStructureDecoder` |
| input structure resolution | `64` |
| latent spatial resolution | `16^3` |
| latent channels | `8` |
| loss | `dice` + KL |
| pretrained weights | `microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16` 和 `ss_dec_conv3d_16l8_fp16` |

TRELLIS.2 的 Sparse Structure Flow 配置里仍然指定：

```json
"pretrained_ss_dec": "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
```

当前仓库的数据编码脚本也默认使用同一家族的 encoder：

```text
microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16
```

因此可以理解为：**TRELLIS.2 没有重新定义第一阶段的 sparse structure latent
空间，而是在这个旧 latent 空间上训练了新的条件生成模型。**

### 2.2 训练目标仍是 SS latent

TRELLIS.2 训练 Sparse Structure Flow 时，数据目录中需要已经离线编码好的：

```text
ss_latents/ss_enc_conv3d_16l8_fp16_64
```

训练时 flow 学的是从噪声到这些 `16^3 x 8` SS latent 的映射。decoder 只用于
可视化/采样阶段把 latent 解码回 occupancy，不是被 flow 训练更新的对象。

## 3. 哪些部分重新训练或改进了

### 3.1 Image-conditioned Sparse Structure Flow 是新模型

两代的 image-conditioned Sparse Structure Flow 都叫 `SparseStructureFlowModel`，
但配置已经不同：

| 项目 | TRELLIS.1 | TRELLIS.2 |
| --- | --- | --- |
| 配置文件 | `configs/generation/ss_flow_img_dit_L_16l8_fp16.json` | `configs/gen/ss_flow_img_dit_1_3B_64_bf16.json` |
| 权重来源 | `microsoft/TRELLIS-image-large` | `microsoft/TRELLIS.2-4B` |
| denoiser | `SparseStructureFlowModel` | `SparseStructureFlowModel` |
| latent resolution | `16` | `16` |
| in/out channels | `8 -> 8` | `8 -> 8` |
| model channels | `1024` | `1536` |
| blocks | `24` | `30` |
| heads | `16` | `12` |
| MLP ratio | `4` | `5.3334` |
| positional encoding | `ape` | `rope` |
| precision | `fp16_mode: inflat_all` / `use_fp16` | AMP `bfloat16` |
| optimizer weight decay | `0.0` | `0.01` |
| batch split | `1` | `4` |
| cross-attn qk RMS norm | basic `qk_rms_norm` | `qk_rms_norm` + `qk_rms_norm_cross` |

所以 TRELLIS.2 的第一阶段不是“只把 DINOv2 输出塞进旧模型”，而是重新训练了一个
更大的 flow denoiser，并调整了位置编码、归一化、优化器和混合精度设置。

### 3.2 图像条件从 DINOv2 换成 DINOv3

TRELLIS.1 的 image flow 使用：

```json
"image_cond_model": "dinov2_vitl14_reg"
```

TRELLIS.2 使用：

```json
"image_cond_model": {
    "name": "DinoV3FeatureExtractor",
    "args": {
        "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512
    }
}
```

这带来几个直接影响：

- 条件 token 的语义空间变了，旧 flow 对 DINOv2 token 学到的 cross-attention
  关系不能直接复用。
- 图像输入尺寸从 TRELLIS.1 配置中的 `518` 变为 TRELLIS.2 的 `512`。
- DINOv3 extractor 通过 Hugging Face `DINOv3ViTModel` 加载，并显式使用模型的
  RoPE position embeddings。

因此，即使 sparse structure latent 的目标空间没变，条件分布已经变了。为了让
flow 正确解释 DINOv3 的视觉 token，重新训练是合理且必要的。

### 3.3 TRELLIS.2 与后续 Shape/Texture 生成重新对齐

TRELLIS.1 的第一阶段服务于原始 SLat 生成；TRELLIS.2 的第一阶段服务于：

```text
Sparse Structure coords
  -> Shape SLat Flow
  -> Texture SLat Flow
  -> Shape / Texture SC-VAE Decoder
  -> O-Voxel / PBR mesh asset
```

TRELLIS.2 的下游生成目标变成了 512 甚至 1024/1536 级别的 O-Voxel 资产，并且要
显式生成 PBR 材质属性。第一阶段结构生成虽然仍然是粗结构，但它的错误会影响后续
Shape SC-VAE 和 Texture SC-VAE 的空间坐标。因此 TRELLIS.2 重新训练第一阶段 flow
也可以看作是让旧 SS latent 空间适配新下游生成栈。

## 4. 数据集与数据制备差异

这里要区分“官方项目描述”和“当前仓库公开配置能直接看到的内容”。

### 4.1 TRELLIS.1

TRELLIS.1 README 声明预训练使用 **TRELLIS-500K**，包含约 500K 个 3D 资产，来源
包括 Objaverse(XL)、ABO、3D-FUTURE、HSSD、Toys4k，并基于 aesthetic score 过滤。

其 Sparse Structure VAE 和 Sparse Structure Flow 配置都使用：

```json
"min_aesthetic_score": 4.5
```

Image-conditioned Sparse Structure Flow 的数据集参数还包含：

```json
"latent_model": "ss_enc_conv3d_16l8_fp16",
"image_size": 518,
"pretrained_ss_dec": "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
```

### 4.2 TRELLIS.2

TRELLIS.2 当前 README 给出的训练示例使用 `ObjaverseXL_sketchfab`：

```sh
python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_64_bf16.json \
  --output_dir results/ss_flow_img_dit_1_3B_64_bf16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \"ss_latent\": \"datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16_64\", \"render_cond\": \"datasets/ObjaverseXL_sketchfab/renders_cond\"}}"
```

公开配置同样使用：

```json
"min_aesthetic_score": 4.5
```

但 TRELLIS.2 的数据工具链变成：

```text
raw 3D assets
  -> mesh / PBR dumps
  -> O-Voxel / dual grid preprocessing
  -> Shape/PBR SC-VAE latent encoding
  -> SS latent encoding
  -> render_cond image rendering
  -> flow training
```

也就是说，TRELLIS.2 的公开训练入口仍围绕 Objaverse-XL 风格的数据组织展开，
但它的中间产物已经服务于新的 O-Voxel + SC-VAE pipeline。

## 5. 为什么 TRELLIS.2 需要重训 Sparse Structure Flow

可以把原因拆成三层：

1. **条件编码器不兼容**  
   DINOv2 和 DINOv3 的 token 分布、位置编码和视觉语义表示不同。旧 flow 的
   cross-attention 权重是围绕 DINOv2 条件学出来的，不能假定直接适配 DINOv3。

2. **flow backbone 已经升级**  
   TRELLIS.2 把模型宽度、深度、位置编码和归一化策略都改了。结构不同意味着
   checkpoint 也不能直接继承。

3. **下游生成系统换代**  
   TRELLIS.2 后续是 Shape/Texture SC-VAE 与 O-Voxel 资产生成。第一阶段虽然仍在
   旧 SS latent 空间中生成结构，但它需要和新的 shape/texture flow 训练分布对齐。

## 6. 一句话总结

TRELLIS.2 的 Sparse Structure Generation 是“**复用前代 Sparse Structure VAE
表示空间，重训并升级 image-conditioned Sparse Structure Flow**”。

换句话说：

- 没变的是：`16^3 x 8` 的 SS latent 空间，以及
  `TRELLIS-image-large` 的 sparse structure encoder/decoder。
- 变了的是：图像条件从 DINOv2 到 DINOv3，flow denoiser 规模和训练 recipe 升级，
  并且整个第一阶段被重新接入 TRELLIS.2 的 O-Voxel/SC-VAE 生成链路。

## 参考文件

本仓库：

- [`configs/gen/ss_flow_img_dit_1_3B_64_bf16.json`](configs/gen/ss_flow_img_dit_1_3B_64_bf16.json)
- [`trellis2/modules/image_feature_extractor.py`](trellis2/modules/image_feature_extractor.py)
- [`trellis2/datasets/sparse_structure_latent.py`](trellis2/datasets/sparse_structure_latent.py)
- [`data_toolkit/README.md`](data_toolkit/README.md)
- [`README_INFERENCE_512_ZH.md`](README_INFERENCE_512_ZH.md)

官方 TRELLIS.1：

- [`microsoft/TRELLIS/configs/vae/ss_vae_conv3d_16l8_fp16.json`](https://github.com/microsoft/TRELLIS/blob/main/configs/vae/ss_vae_conv3d_16l8_fp16.json)
- [`microsoft/TRELLIS/configs/generation/ss_flow_img_dit_L_16l8_fp16.json`](https://github.com/microsoft/TRELLIS/blob/main/configs/generation/ss_flow_img_dit_L_16l8_fp16.json)
- [`microsoft/TRELLIS` README](https://github.com/microsoft/TRELLIS) 的 Dataset 与 Training 配置表
