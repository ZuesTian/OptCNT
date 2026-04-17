# OptCNT

OptCNT 是一个基于 Python、Tkinter 和 OpenCV 的 CNT SEM 图像桌面分析工具，面向单图分析、ROI 分析和多图对比分析场景。

当前版本重点能力包括：

- 自动或手动比例尺标定
- 支持实时预览的预处理参数调节
- CNT 检测与长度 / 宽度 / 长宽比测量
- 分散 CNT / 团聚 CNT 分类统计
- 骨架、轮廓和检测结果可视化
- 空间分布、热点区域和均匀度分析
- 双图对比、组图对比与摘要报告
- Windows 轻量打包流程

## 快速开始

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

可用启动方式：

- `python main.py`：推荐，从仓库根目录启动
- `python src/main.py`：直接使用源码入口，适合本地调试

根目录下的 `main.py` 是轻量入口，会转发到 `src.main`。

## 主要功能

### 单图分析

- 结果表格默认一行对应一根 CNT
- 结果表格包含 `ID`、`Length (um)`、`Dispersed CNT`、`Agglomerated CNT`
- 结果面板会显示长度统计、空间分布、均匀度和评判框架摘要

### 对比分析

- 双图对比和组图对比都会沿用当前界面的预处理参数和识别参数
- 默认分析图像中部 `75%` 区域，尽量减小比例尺区域干扰
- 对比分析在后台执行，适合连续多次调参与复核
- 对比摘要会显示本次分析实际使用的参数快照

## 核心算法与数学模型

本系统不仅提供图像的可视化分析，更内置了一套严谨的数学评价体系，以科学量化碳纳米管（CNT）的分布特征。以下是核心算法及指标的公式说明：

### 1. 骨架提取与长度测量 (Skeleton & Length)

- **骨架化 (Skeletonization)**：基于形态学细化算法（如 `cv2.ximgproc.thinning` 中的 Zhang-Suen 算法），提取 CNT 连通域的单像素中心轴。
- **物理长度 (Physical Length)**：将骨架像素转化为无向图 $G(V, E)$，利用图遍历算法寻找图的直径（Diameter，即图中最长简单路径）作为 CNT 的实际物理长度，避免了简单周长法对复杂拓扑的高估。

### 2. 空间分布均匀性得分 (Uniformity Score)

传统的变异系数（CV）仅能反映单一维度的离散程度。系统采用多维度特征融合，并通过 Sigmoid 函数映射统一方向（0-100分，分数越高越均匀）：

$$
S_{base} = w_{NN} \cdot S_{sig}(CV_{NN}) + w_{Grid} \cdot S_{sig}(CV_{Grid}) + w_{Moran} \cdot S_{moran}(I)
$$

其中：
- **$CV_{NN}$ (最近邻距离CV)**：反映个体间的局部排斥程度。
- **$CV_{Grid}$ (网格密度CV)**：反映宏观上的空间占据均匀度。
- **$I$ (Moran's I 莫兰指数)**：反映空间自相关性（聚集或离散）。

**映射函数：**
对于“越小越好”的 CV 类指标，采用缩放的 Sigmoid 函数（参数 $k=4.0, m=0.6$）：
$$ S_{sig}(x) = \frac{100}{1 + \exp\left(k \cdot \frac{x - m}{m}\right)} $$
对于 Moran's I，映射中心偏移至随机分布点：
$$ S_{moran}(I) = \frac{100}{1 + \exp(5.0 \cdot (I - 0.1))} $$

**长管非线性惩罚 (Long Tube Penalty)：**
大量超长 CNT 的存在会破坏宏观均匀性。系统引入针对 $>40\mu m$ 长管比例 $r$ 的幂次放大惩罚（指数 $exp=0.6$，最高扣减 $P_{max}=25.0$）：
$$ Penalty = P_{max} \cdot \min(1.0, r)^{0.6} $$

最终均匀度主指标得分：
$$ Score_{uniformity} = \max(0, S_{base} - Penalty) $$

### 3. 四维评判框架与混合评分 (Hybrid Evaluation Framework)

为了全面评估 CNT 样品的质量，系统构建了四个独立维度的评价框架，并合成最终的**混合评分 (Hybrid Score)**。混合评分位于 $[0, 100]$，分数越高代表样品综合质量（从长度、细度、均匀度角度）越优。

$$ Score_{hybrid} = 0.30 \times Score_A + 0.20 \times Score_B + 0.30 \times Score_C + 0.20 \times Score_D $$

#### A. 均匀性主指标 (Uniformity, 权重 30%)
直接采用上述计算的综合均匀度得分：
$$ Score_A = Score_{uniformity} $$

#### B. 粗管/束化指标 (Thick Bundle, 权重 20%)
评估 CNT 的表观宽度与团束现象。基于平均宽度 $W_{mean}$ 和 90 分位数宽度 $W_{p90}$ 计算：
$$ S_{inv}(x) = \frac{100}{1 + x / 1.0\mu m} $$
$$ Score_B = \frac{S_{inv}(W_{mean}) + S_{inv}(W_{p90})}{2} $$

#### C. 长管指标 (Long Tube, 权重 30%)
评估 CNT 的长度优势。结合平均骨架长度 $L_{mean}$ 与超长管占比 $r_{>40\mu m}$：
$$ S_{mean} = 100 \times \min\left(1.0, \frac{L_{mean}}{80.0\mu m}\right) $$
$$ S_{ratio} = 100 \times (r_{>40\mu m})^{0.6} $$
$$ Score_C = \frac{S_{mean} + S_{ratio}}{2} $$
*(注：长管在均匀性维度作为惩罚项，而在本维度作为长度优势加分项，体现了物理性能与涂布工艺的 trade-off。)*

#### D. 团聚指标 (Agglomeration, 权重 20%)
通过多模态空间热点检测（点密度、覆盖率、阴影密度）划分出团聚区。基于团聚区总面积占比 $R_{area}$ 和最大单体团聚区占比 $R_{largest}$：
$$ Score_D = \frac{100(1 - R_{area}) + 100(1 - R_{largest})}{2} $$

## 项目结构

```text
OptCNT/
|-- main.py
|-- src/
|   |-- main.py
|   |-- core/
|   |   |-- analyzer_core.py
|   |   |-- models.py
|   |   |-- stats_compat.py
|   |   `-- utils.py
|   `-- gui/
|       |-- gui.py
|       |-- gui_styles.py
|       |-- panels.py
|       `-- widgets.py
|-- tests/
|-- benchmark.py
|-- benchmark_data.py
|-- build_minimal.py
|-- package_windows.ps1
|-- PACKAGING.md
|-- requirements.txt
|-- requirements-dev.txt
|-- requirements-build.txt
`-- requirements-optional.txt
```

## 依赖说明

- `requirements.txt`：桌面程序运行依赖
- `requirements-dev.txt`：运行依赖 + `pytest`
- `requirements-build.txt`：运行依赖 + PyInstaller 打包依赖
- `requirements-optional.txt`：可选扩展依赖，例如 `scikit-learn`

打包版本使用的是 `opencv-contrib-python-headless`，这样 `cv2.ximgproc.thinning` 这条更快的细化路径也能在最终 `exe` 中使用。

## 测试

```powershell
pip install -r requirements-dev.txt
pytest -q
```

## Windows 打包

如果要构建体积尽量小、并带 UPX 压缩的 Windows 可执行文件，可以直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\package_windows.ps1
```

这个脚本会自动：

1. 重新创建 `.venv-build`
2. 安装 `requirements-build.txt`
3. 复用或自动下载可用的 UPX
4. 构建 `dist\OptCNT.exe`

更多打包细节见 [PACKAGING.md](PACKAGING.md)。
