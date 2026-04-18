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

- 支持 CNT 检测、长度 / 宽度 / 长宽比测量
- 支持分散 CNT / 团聚 CNT 统计
- 支持骨架、轮廓、检测结果和空间分布可视化
- 结果表格默认一行对应一根 CNT，包含 ID、Length (μm)、Dispersed CNT、Agglomerated CNT
- 结果面板显示长度统计、空间分布、均匀度和评判框架摘要

### 对比分析

- 双图对比和组图对比都会沿用当前界面的预处理参数和识别参数
- 默认分析图像中部 `75%` 区域，尽量减小比例尺区域干扰
- 对比分析在后台执行，适合连续多次调参与复核
- 对比摘要会显示本次分析实际使用的参数快照
- 组图对比会显式区分 `base组（基准参考）` 与 `实验组（实验候选）`，并在摘要、图表、代表图中保持一致命名和固定配色
- 组图对比的主结论只围绕固定 `5` 项主指标展开：`总CNT数量`、`分散比例`、`网格CV`、`团聚面积占比`、`P90宽度`
- `平均骨架长度` 与 `综合均匀性得分` 作为补充展示项保留，但不作为组间主结论的核心判定依据
- 对比逻辑明确区分"数量"和"空间"两个维度：数量多不等于分布均匀，分散比例用于评价分散程度，网格CV用于评价空间均匀性

## 核心算法与数学模型

### 1. 骨架提取与长度测量

**骨架化 (Skeletonization)**：基于形态学细化算法（Zhang-Suen 算法），提取 CNT 连通域的单像素中心轴。

**物理长度 (Physical Length)**：将骨架像素转化为无向图 G(V, E)，利用图遍历算法寻找图的直径（Diameter，即图中最长简单路径）作为 CNT 的实际物理长度，避免了简单周长法对复杂拓扑的高估。

### 2. 空间分布均匀性得分

系统采用多维度特征融合，并通过 Sigmoid 函数映射统一方向（0-100分，分数越高越均匀）：

```
S_base = w_NN · S_sig(CV_NN) + w_Grid · S_sig(CV_Grid) + w_Moran · S_moran(I)
```

其中：
- `CV_NN`：最近邻距离 CV，反映个体间的局部排斥程度
- `CV_Grid`：网格密度 CV，反映宏观上的空间占据均匀度
- `I`：Moran's I 莫兰指数，反映空间自相关性（聚集或离散）

**映射函数**：
- 对于"越小越好"的 CV 类指标：`S_sig(x) = 100 / (1 + exp(k · (x - m) / m))`，其中 k=4.0, m=0.6
- 对于 Moran's I：`S_moran(I) = 100 / (1 + exp(5.0 · (I - 0.1)))`

**长管非线性惩罚**：大量超长 CNT（>40μm）的存在会破坏宏观均匀性，引入幂次放大惩罚：`Penalty = P_max · min(1.0, r)^0.6`，最高扣减 25 分。

**最终均匀度**：`Score_uniformity = max(0, S_base - Penalty)`

### 3. 四维评判框架与混合评分

系统构建了四个独立维度的评价框架，合成最终的混合评分（0-100分）：

```
Score_hybrid = 0.30 × Score_A + 0.20 × Score_B + 0.30 × Score_C + 0.20 × Score_D
```

| 维度 | 指标 | 权重 | 说明 |
|------|------|------|------|
| A | 均匀性主指标 | 30% | 直接采用综合均匀度得分 |
| B | 粗管/束化指标 | 20% | 基于平均宽度和 P90 宽度评估团束现象 |
| C | 长管指标 | 30% | 评估 CNT 的长度优势 |
| D | 团聚指标 | 20% | 评估团聚程度，团聚面积占比越低越好 |

## 核心能力概览

- 基于预处理、骨架和轮廓结果完成 CNT 检测与长度 / 宽度测量
- 提供分散 / 团聚统计、空间分布和热点区域分析
- 支持双图对比和组图对比，并输出摘要与图表
- 组图对比会显式区分 `base组（基准参考）` 与 `实验组（实验候选）`
- 当前组间主结论围绕固定 `5` 项指标：`总CNT数量`、`分散比例`、`网格CV`、`团聚面积占比`、`P90宽度`
- 对比解读遵循"数量维度"和"空间维度"分开看的原则：**数量多，不等于分布均匀**

## 项目结构

```text
OptCNT/
|-- main.py                 # 轻量入口，转发到 src.main
|-- src/
|   |-- main.py            # 主程序入口
|   |-- core/              # 核心分析模块
|   |   |-- analyzer_core.py   # CNTAnalyzer 类，图像处理核心
|   |   |-- models.py          # 数据模型（ROIRegion, CNTMeasurement）
|   |   |-- utils.py           # 工具函数和常量
|   |   |-- stats_compat.py    # 统计兼容性模块
|   `-- gui/               # GUI 模块
|       |-- gui.py             # 主 GUI 控制器
|       |-- panels.py          # 面板组件
|       |-- widgets.py         # 自定义控件
|       |-- comparison_view.py # 对比视图
|       |-- chart_manager.py   # 图表管理
|       |-- gui_layout.py      # 布局辅助
|       |-- gui_tasking.py     # 任务处理
|       `-- gui_styles.py      # 样式定义
|-- tests/                 # 测试用例
|-- package_windows.ps1    # Windows 打包脚本
|-- build_minimal.py       # 最小化构建脚本
`-- PACKAGING.md          # 打包详细说明
```

## 依赖说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 桌面程序运行依赖 |
| `requirements-dev.txt` | 运行依赖 + `pytest` |
| `requirements-build.txt` | 运行依赖 + PyInstaller 打包依赖 |
| `requirements-optional.txt` | 可选扩展依赖，例如 `scikit-learn` |

打包版本使用 `opencv-contrib-python-headless`，确保 `cv2.ximgproc.thinning` 细化路径在最终 exe 中可用。

核心依赖：
- `opencv-contrib-python-headless>=4.5.0` - 图像处理核心
- `numpy>=1.20.0` - 数值计算
- `Pillow>=8.0.0` - 图像显示
- `matplotlib>=3.3.0` - 图表绘制
- `numba>=0.60.0` - JIT 加速（可选，性能模式保留）

## 测试

```powershell
pip install -r requirements-dev.txt
pytest -q
```

## Windows 打包

### 一键打包

```powershell
powershell -ExecutionPolicy Bypass -File .\package_windows.ps1
```

默认使用 `slim` 配置（最小体积，排除 numba/llvmlite）。如需保留 JIT 加速：

```powershell
powershell -ExecutionPolicy Bypass -File .\package_windows.ps1 -Profile performance
```

### 打包流程

脚本会自动：

1. 创建 `.venv-build` 虚拟环境
2. 安装 `requirements-build.txt` 依赖
3. 复用或自动下载 UPX 压缩工具
4. 构建 `dist\OptCNT.exe`
5. 运行 UPX `--best` 压缩

### 打包配置

| 配置 | 说明 | 体积 | 性能 |
|------|------|------|------|
| `slim` | 最小体积，排除 numba/llvmlite | 较小 | 标准 |
| `performance` | 保留 numba/llvmlite，仅裁剪测试模块 | 较大 | JIT 加速 |

更多打包细节见 [PACKAGING.md](PACKAGING.md)。
