# OptCNT Web 工作台

该目录是 OptCNT FastAPI 服务提供的浏览器界面，不是独立的浏览器简化分析器。上传图像由服务端会话调用项目中的同一个 `CNTAnalyzer`，因此桌面端和远端使用同源的预处理、CNT 骨架测量、颗粒识别及统计口径。

## 功能

- 单图上传、自动/手动比例尺、多个 ROI、参数建议和二值/骨架预览。
- 完整 CNT 检测、逐根长度/宽度测量、空间均匀性与团聚分析。
- 深色紧凑颗粒候选识别，提供独立的数量、面积、等效直径、形态和置信度结果。
- 红色颗粒轮廓与 `P1...Pn` 编号视图，以及 JSON、CSV、TXT、CNT PNG、骨架 PNG、颗粒 PNG 导出。
- base 组与实验组批量对比。

颗粒结果不会计入 CNT 数量、分散比例、均匀性或组间五项主指标。单张弱标注样图不足以建立确定的颗粒分类器，当前结果应作为候选复核；外观相似的 CNT 交叉结点仍可能入选。

## 本地启动

从项目根目录运行：

```powershell
pip install -r requirements.txt
pip install -r requirements-web.txt
python -m uvicorn src.webapp.app:app --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000`。不能再用单独的静态文件服务器启动 `web/`，因为界面需要 `/api/v1` 下的同源分析 API。

## 远端部署边界

- 原图通过 HTTP(S) 上传到分析服务器；单图保存在内存会话中，批量任务使用受限临时目录。
- 默认单图会话 30 分钟过期。可用 `OPTCNT_SESSION_TTL_SECONDS`、`OPTCNT_MAX_UPLOAD_MB`、`OPTCNT_MAX_SESSIONS`、`OPTCNT_JOB_WORKERS` 等环境变量限制资源。
- 生产环境应使用 HTTPS、反向代理和访问控制，并根据机器内存限制上传大小、会话数及批量并发。
