# GT-free K=96 Moving-Cylinder PINN Basis

本仓库是最终采用的移动圆柱连续参数基底代码。训练阶段严格
GT-free：优化器不读取 Navier–Stokes（NS）快照、数值特征函数或监督标签。
独立 Re=100 NS 数据只在模型冻结后用于投影、误差评估和视频绘制。

## 最终协议

- 参数域：`alpha ∈ [0, 1]`
- 圆柱中心：`x_c(alpha) = 1 + 8 alpha`，`y_c = 0`
- 通道：`[0, 10] × [-2, 2]`
- 圆柱半径：`0.45`
- 模态数：`K = 96`
- 96 个相互独立的正弦 MLP，每个模态一个 Adam 优化器
- 固定模态顺序；不排序、不交换模态
- 前序模态在后续模态优化时 `detach`
- 基于同一最小二乘系数，同时 peeling 速度场和速度梯度
- 局部化双零边界包络，流入端、上下壁面和圆柱满足齐次速度边界
- 混合 Sobol 采样：50% 全局、25% 圆柱邻域、25% 尾流区
- 所有 Fourier 频率从 epoch 1 起全部启用，无频率课程
- 无 projection penalty
- 学习率固定为 `8e-4`，无 scheduler
- 训练损失只有平均归一化 Rayleigh 能量：

  `loss = mean_k[(||grad u_k||² / ||u_k||²) / 80]`

`projection ratio` 仍作为 raw candidate overlap 诊断量保存，但不在 loss
中，也不进入梯度。Peeling 后的输出基底在离散速度内积下保持正交。

## 目录

```text
configs/k96_final.json            最终 5000-epoch 配置
src/k96_pinn/training.py          网络、采样、peeling、训练、续训、checkpoint
src/k96_pinn/postprocess.py       PINN 前向、导数、Ritz 提取、投影、MP4 编码
src/k96_pinn/ns_reference.py      独立 Re=100 NS 数据的规范化缓存
scripts/train.py                  训练入口
scripts/build_ns_cache.py         从原始 HDF5 生成 NS 缓存
scripts/predict_video.py          101 个 alpha 的 NS/PINN/误差视频
scripts/plot_training.py          loss、学习率及诊断曲线
tests/test_smoke.py               配置、前向、采样和广义特征值测试
```

旧实验、日志、checkpoint、原始 HDF5、缓存、帧和视频均不进入仓库。

## 环境

推荐：

- Windows 10/11 或 Linux
- Python 3.11–3.13
- PyTorch 2.5+
- CUDA GPU；正式 K=96 训练建议至少 16 GB 显存
- `ffmpeg` 在 `PATH` 中

Windows PowerShell：

```powershell
git clone <PRIVATE_REPOSITORY_URL>
cd gt-free-k96-moving-cylinder-pinn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -version
```

Linux：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## 1. 自检

快速测试：

```powershell
pytest -q
python scripts/train.py --self-test
```

`--self-test` 使用 3 个小模态验证：

- later-mode loss 不向 earlier mode 泄漏梯度；
- 每个独立优化器只更新自己的模态；
- peeling 正交误差有限；
- one-shot alpha 推理形状正确。

## 2. 正式训练

```powershell
$run = "results\k96_final\run_001"

python -u scripts/train.py `
  --run-dir $run `
  --modes 96 `
  --epochs 5000 `
  --hidden 64 `
  --hidden-layers 4 `
  --frequencies 6 `
  --points-per-step 512 `
  --validation-points 1024 `
  --validation-alphas 7 `
  --validate-every 50 `
  --checkpoint-every 25 `
  --probe-alpha 0.15 `
  --probe-every 100 `
  --learning-rate 8e-4 `
  --envelope-mode localized `
  --sampling-mode mixed `
  --global-fraction 0.50 `
  --cylinder-fraction 0.25 `
  --wake-fraction 0.25
```

推荐直接读取版本控制中的完整配置：

```powershell
python -u scripts/train.py `
  --config configs\k96_final.json `
  --run-dir results\k96_final\run_001
```

命令行参数可以覆盖配置文件中的同名参数。例如先进行 10 epoch
流程检查：

```powershell
python -u scripts/train.py `
  --config configs\k96_final.json `
  --epochs 10 `
  --run-dir results\k96_smoke
```

### 每个 epoch 的学习逻辑

1. 由确定性 Sobol 序列获得一个连续 `alpha`；每 20 epoch 插入一次
   `alpha=0/1` 端点。
2. 在全局、圆柱邻域和尾流区生成 512 个混合连续采样点。
3. 按 `k=0,…,95` 固定顺序：
   - 对第 `k` 个 MLP 求 streamfunction、速度和速度梯度；
   - 将 raw velocity 最小二乘投影到已 peeling 的 `0,…,k-1` 模态；
   - 用同一组系数从 streamfunction 和梯度中减去前序分量；
   - 归一化 residual velocity；
   - 最小化 `Rayleigh / 80`；
   - 前序模态已经 detach，只有第 `k` 个网络更新。
4. 保存 train loss、mean/max energy、raw projection ratio、正交误差和
   effective raw rank。
5. 每 50 epoch 在 7 个固定 alpha 上验证；每 100 epoch 在
   `alpha=0.15` 固定 Sobol 点上 probe。
6. validation loss 改善时更新 `checkpoints/best_model.pt`。

### 续训

```powershell
python -u scripts/train.py `
  --run-dir results\k96_final\run_001 `
  --resume results\k96_final\run_001\checkpoints\latest.pt
```

续训会恢复模型、96 个 Adam optimizer、epoch、历史和 best validation。

### 训练输出

```text
config.json
protocol.json
train.log
data/training_history.json
data/training_history.csv
data/alpha_0p15_milestones.*
checkpoints/latest.pt
checkpoints/best_model.pt
checkpoints/epoch_*.pt
checkpoints/sss_official_peeling_frozen.pt
```

正式训练完成的双重判据是：

1. 日志包含 `training complete`；
2. `checkpoints/sss_official_peeling_frozen.pt` 存在。

## 3. 训练曲线

```powershell
python scripts/plot_training.py `
  --run-dir results\k96_final\run_001
```

生成 PNG/SVG/PDF，包含：

- train、validation、`alpha=0.15` probe loss；
- 固定学习率；
- mean Rayleigh energy；
- raw projection ratio（仅诊断）；
- peeling 正交误差；
- effective rank。

## 4. 构造独立 NS 标准缓存

原始数据为 Re=100 cylinder NS HDF5：

- DOI：`10.5281/zenodo.18600528`
- 预期字段：
  - `coordinates/x`
  - `coordinates/y`
  - `time`
  - `fields/field_1`（u）
  - `fields/field_2`（v）

```powershell
python scripts/build_ns_cache.py `
  --source data\raw\cylinder_re100.h5 `
  --output-dir data\cache\re100 `
  --phase-frames 17
```

生成 `actual_canonical_cache.npz`。缓存包含圆柱相对坐标、17 个周期相位
的 `u/v/psi/omega`，以及时间平均 lifting。

该步骤完全独立于 basis 训练。训练代码不会打开该缓存。

## 5. 101-alpha 预测与视频

最终发布代码直接计算：

```python
alphas = np.linspace(0.0, 1.0, 101)
```

即步长 `0.01`。每个 alpha 都重新执行：

1. PINN one-shot forward；
2. 512 点对称 quadrature 上的一、二阶导数；
3. raw 96 维空间的 mass/stiffness 矩阵；
4. 广义对称特征值问题，得到该 alpha 的 K=96 Ritz basis；
5. 在独立 NS 相位快照上进行 mean-flow lifting；
6. 用速度加权最小二乘求 96 个系数；
7. 重构 `u/v/psi/omega` 并计算速度、涡量相对误差。

这里不再使用 21 个 alpha 锚点，也不存在 21→201 的场插值。

```powershell
python -u scripts/predict_video.py `
  --checkpoint results\k96_final\run_001\checkpoints\best_model.pt `
  --source-cache data\cache\re100\actual_canonical_cache.npz `
  --output-dir results\k96_final\prediction_101_alpha `
  --alpha-samples 101 `
  --phase-cycles 2 `
  --fps 10
```

视频保持最终确认的单列三行格式：

1. independent Re=100 NS reference；
2. PINN K=96 reconstruction；
3. absolute velocity error。

只删除 GT 列；不构造也不展示 GT basis。101 个 alpha 同时推进两个流动
相位周期。默认视频为 10.1 秒、10 fps。

预测支持断点续跑：每个 alpha 的 Ritz basis 和重构样本分别写入
`basis/` 与 `samples/`。再次执行同一命令会复用已完成的文件。

主要输出：

```text
basis/alpha_*.npz
samples/sample_*.npz
frames/frame_*.png
sample_metrics.csv
frame_metrics.csv
summary.json
ns_vs_pinn_k96_continuous_alpha_flow.mp4
```

## 可复现性说明

- 所有随机源由 `seed=20260715` 控制。
- 训练和 Ritz quadrature 使用确定性 scrambled Sobol 序列及固定 seed。
- 每个 checkpoint 保存完整配置、96 个 optimizer state 和历史。
- `best_model.pt` 按 validation loss 选择，不按 NS 误差选择。
- 训练固定学习率，无 scheduler、projection penalty、frequency
  curriculum。
- CUDA 不同版本可能造成末位浮点差异；rank、曲线趋势和重构指标应稳定。
- 正式 K=96 训练及 101-alpha 二阶导数预测耗时较长。先使用
  `--self-test` 和较小的 `--modes/--epochs/--points-per-step` 验证环境。

## 数据与权重

GitHub 仓库不包含第三方 NS HDF5、训练 checkpoint 或生成视频。请依据
DOI 获取数据，或使用自己的同结构数据。若需要发布权重，应使用 GitHub
Release、私有对象存储或 Git LFS，并同时保存 `config.json` 与
`training_history.json`。
