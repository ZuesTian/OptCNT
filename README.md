# OptCNT

OptCNT 是一个基于 Python、Tkinter 和 OpenCV 的 CNT 光学显微图像分析工具，面向单图分析、ROI 分析和多图对比分析场景。

**[在线工作台](https://opt-cnt.47.236.76.214.nip.io/)** · [全部 CNT 工具](https://zuestian.github.io/Cnanotools/)

## 版本信息

- 当前源码版本：`v2026.11.05`（已有打包文件不随源码自动更新）
- 更新时间：`2026-11-05`
- 当前分支：`main`

### 本次更新

- 修复全前景区域骨架提取不终止、边界宽度溢出与重叠轮廓面积被扣除的问题
- 最近邻统计使用固定大小分块；断裂合并使用端点空间索引筛选候选，保留距离与方向判定
- 拒绝非有限比例尺及检测参数；队列按分析引擎版本领取任务
- 审核与前后对照见 [CODE_REVIEW.md](CODE_REVIEW.md)。本次修改会影响触及 ROI 边界的检测结果，跨版本比较应重新分析原图
- 新增深色紧凑颗粒候选识别、逐颗粒尺寸/形态/置信度统计和红色标注视图
- 颗粒结果使用独立统计口径，不计入 CNT 数量、分散比例、均匀性或组间五项主指标
- 新增与桌面端共用 `CNTAnalyzer` 的 FastAPI Web 工作台，消除远端前端与项目算法分叉
- 对比分析界面切换时不再重排主三栏布局，减少标签切换跳变
- `comparison` 页收敛为单次延迟布局刷新，降低图表与摘要区连跳
- 结果面板内部 sash 仅在高度明显变化时重算，减少联动抖动
- 分散比例统计口径回退到 `2026-04-10` 版本，同时保留新版 `团聚面积占比` 与 `P90宽度`
- 已完成 Windows `slim` 打包，生成 `dist\\OptCNT.exe`

当前版本重点能力包括：

- 自动或手动比例尺标定
- 支持实时预览的预处理参数调节
- CNT 检测与长度 / 宽度 / 长宽比测量
- 深色紧凑颗粒候选识别与尺寸 / 形态 / 置信度测量
- 分散 CNT / 团聚 CNT 分类统计
- 骨架、轮廓和检测结果可视化
- 空间分布、热点区域和均匀度分析
- 双图对比、组图对比与摘要报告
- Windows 轻量打包流程
- 同源 Web API 与浏览器工作台

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
- 支持深色紧凑颗粒候选识别，输出面积、等效直径、长短轴、圆度、实心度、局部对比度和置信度
- 颗粒标注图使用红色轮廓与 `P1...Pn` 编号；颗粒统计不改变现有 CNT 指标
- 支持分散 CNT / 团聚 CNT 统计
- 支持骨架、轮廓、检测结果和空间分布可视化
- 结果表格默认一行对应一根 CNT，包含 ID、Length (μm)、Dispersed CNT、Agglomerated CNT
- 结果面板显示长度统计、空间分布、均匀度和评判框架摘要

### Web 工作台

Web 端与桌面端调用同一个 `CNTAnalyzer` 核心，不在浏览器中复制简化算法。单图流程支持比例尺、ROI、参数预览、CNT 检测、颗粒识别、统计表格以及 JSON / CSV / TXT / PNG 导出。

```powershell
pip install -r requirements-web.txt
python -m uvicorn src.webapp.app:app --host 127.0.0.1 --port 8000
```

然后访问 `http://127.0.0.1:8000`。部署到远端时建议由反向代理提供 HTTPS，并按环境设置上传、会话和并发限制；详见 [web/README.md](web/README.md)。

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

**骨架化 (Skeletonization)**：使用十字结构元素反复腐蚀，累积腐蚀图与其开运算结果之差，生成形态学骨架。当前实现不是 Zhang-Suen 细化；骨架可能存在分支或断开。ROI 外按背景处理，保证全前景区域也能终止。

**长度估计 (Length Estimation)**：将骨架像素构建为八邻接带权图。简单链直接遍历，分支图使用两次 Dijkstra 搜索估计端点间主路径；一般含环图不保证求得精确直径。图直径指最短路径距离的最大值，不等同于最长简单路径。若平滑轮廓的半周长减去宽度超过主路径长度的 1.25 倍，采用轮廓估计修正。结果是二维图像中的对象长度估计，不能直接视为真实单根 CNT 长度；接触 ROI 边界的对象还可能被截断。

### 2. 空间分布均匀性得分

当前综合均匀性得分使用覆盖密度 CV、热点面积比例、覆盖熵和 Moran's I（0–100 分，分数越高越均匀）：

```
S = clip(100 × [1 − 0.35 × clip(CV_coverage, 0, 2)
                 − 0.25 × A_hotspot − 0.20 × (1 − H_coverage)
                 − 0.20 × clip(I, 0, 1)], 0, 100)
```

其中 `CV_coverage` 为每格像素覆盖率的 CV，`A_hotspot` 为热点网格面积比例，`H_coverage` 为归一化覆盖熵，`I` 由计数网格计算。主指标中的“网格 CV”是中心点计数 CV，与这里的覆盖 CV 不同。

最近邻、计数网格及 Moran 分项仍提供 Sigmoid 诊断分数，但不按旧公式合成为总分。少于 2 个有效中心点时总分为 0；2–4 个时按 `50 + (S − 50) × n / 5` 向 50 收缩，并标记低可信度。

长管比例作为诊断值及四维框架的组成项保留，当前均匀性总分不再扣除旧版长管惩罚。评分权重和阈值是现有经验规则，本次没有重新拟合或据此验证材料性能。

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
- 基于原始光学图像的局部暗度、紧凑形态和多阈值稳定性生成颗粒候选
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
|   |-- webapp/            # 与桌面端同源的 FastAPI 会话、任务与序列化层
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
|-- web/                   # 浏览器工作台静态资源
|-- requirements-web.txt  # Web API 运行依赖
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
| `requirements-web.txt` | Web API 与远端工作台依赖 |

当前依赖锁定 `opencv-python-headless==4.13.0.92`，骨架处理不依赖 `cv2.ximgproc.thinning`。已有 exe 需另行重新打包才能包含本次改动。

核心依赖：
- `opencv-python-headless==4.13.0.92` - 图像处理核心
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
