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

### 空间分布与均匀度

当前版本将“主均匀度分数”和“诊断指标”分开处理。

主均匀度分数使用：

```text
UI = 100 × (1 - 0.5 × CV_d - 0.3 × A_c - 0.2 × R_v)
```

其中：

- `CV_d`：基于覆盖率网格 `coverage_ratio_grid` 的密度变异系数
- `A_c`：热点 / 团聚区域面积占比
- `R_v`：局部密度极差归一化项

分数会被限制在 `0-100` 区间，并给出等级标签：

- `90-100`：极均匀
- `80-89`：均匀
- `70-79`：较均匀
- `60-69`：一般
- `50-59`：较不均匀
- `<50`：明显不均匀

以下指标仍然保留输出，但主要作为诊断参考，不再直接决定主均匀度总分：

- 最近邻 CV
- 网格 CNT 数 CV
- Moran's I
- 长管比例

### 四维评判框架

系统同时输出四维评判框架，用于综合评价样品：

- `A. 均匀性主指标`
- `B. 粗管 / 束化指标`
- `C. 长管指标`
- `D. 团聚指标`

并进一步计算混合评分 `hybrid_score`，用于组间对比和代表图摘要。

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
