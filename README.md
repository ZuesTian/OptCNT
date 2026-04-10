# OptCNT

OptCNT 是一个基于 Python、Tkinter 和 OpenCV 的 CNT SEM 图像桌面分析工具。

它目前主要围绕单图分析、ROI 分析和对比分析这几类实用流程展开：

- 自动或手动比例尺校准
- 支持实时预览的预处理参数调节
- CNT 检测与长度 / 宽度 / 长宽比测量
- 单图测量列表中显示分散 CNT / 团聚 CNT 标记
- 轮廓与骨架叠加可视化
- 空间分布与热点分析
- 使用当前界面参数的双图 / 组图对比分析
- 使用 UPX 压缩的精简 Windows 打包流程

## 快速开始

先安装运行依赖，再启动程序：

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

可用启动方式：

- `python main.py`：推荐，从仓库根目录启动
- `python src/main.py`：直接使用源码入口，适合本地开发调试

根目录下的 `main.py` 是一个轻量入口，会转发到 `src.main`。

## 使用说明

### 单张图片分析

- 测量列表默认一行对应一条 CNT。
- 结果表格包含 `ID`、`Length (um)`、`Dispersed CNT`、`Agglomerated CNT` 四列。
- 结果面板上方显示的分散 / 团聚统计，和高级分析中的热点判断使用的是同一套空间分布分析结果。

### 对比分析

- 双图对比和组图对比都会沿用当前界面的预处理参数和识别参数。
- 对比分析仍然只取图像中部 `75%` 区域，尽量避开比例尺区域对统计结果的干扰。
- 对比分析现在在后台执行，连续多次对比时不需要重启软件。
- 对比结果摘要会显示该次分析实际使用的参数快照。

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

打包版本使用的是 `opencv-contrib-python-headless`，这样 `cv2.ximgproc.thinning` 这条更快的 OpenCV 细化路径也能在最终 `exe` 中使用。

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

当前打包流程会尽量压缩体积，同时保留 CNT 骨架处理依赖的 OpenCV thinning 快路径，避免因为打包而明显拖慢检测。

更多打包细节见 [PACKAGING.md](PACKAGING.md)。
