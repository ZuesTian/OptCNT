"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  config: null,
  health: null,
  page: "analyze",
  session: null,
  activeRoiName: null,
  currentView: "original",
  result: null,
  suggestion: null,
  pendingScaleLine: null,
  drawMode: null,
  drawStart: null,
  drawCurrent: null,
  fitScale: 1,
  zoom: 1,
  renderWidth: 0,
  renderHeight: 0,
  viewSequence: 0,
  previewTimer: null,
  requestSequence: 0,
  measurementSort: { key: "id", direction: 1 },
  compare: {
    baseFiles: [],
    experimentFiles: [],
    jobId: null,
    timer: null,
    result: null,
  },
};

const els = {
  railStatusDot: $("#railStatusDot"),
  railStatusText: $("#railStatusText"),
  railVersion: $("#railVersion"),
  currentSection: $("#currentSection"),
  helpButton: $("#helpButton"),
  singleDropZone: $("#imageViewport"),
  singleImageInput: $("#singleImageInput"),
  uploadLimitText: $("#uploadLimitText"),
  analysisStudio: $("#analysisStudio"),
  sessionFileName: $("#sessionFileName"),
  sessionFileMeta: $("#sessionFileMeta"),
  sessionStateDot: $("#sessionStateDot"),
  sessionStateText: $("#sessionStateText"),
  sessionExpiryText: $("#sessionExpiryText"),
  replaceImageButton: $("#replaceImageButton"),
  viewTabs: $("#viewTabs"),
  drawScaleButton: $("#drawScaleButton"),
  drawRoiButton: $("#drawRoiButton"),
  drawerDrawRoiButton: $("#drawerDrawRoiButton"),
  fitImageButton: $("#fitImageButton"),
  imageViewport: $("#imageViewport"),
  imageTransform: $("#imageTransform"),
  analysisImage: $("#analysisImage"),
  interactionCanvas: $("#interactionCanvas"),
  viewerEmpty: $("#viewerEmpty"),
  viewerBusy: $("#viewerBusy"),
  viewerBusyTitle: $("#viewerBusyTitle"),
  viewerBusyDetail: $("#viewerBusyDetail"),
  drawHint: $("#drawHint"),
  drawHintText: $("#drawHintText"),
  cancelDrawButton: $("#cancelDrawButton"),
  activeViewLabel: $("#activeViewLabel"),
  activeRoiLabel: $("#activeRoiLabel"),
  zoomOutButton: $("#zoomOutButton"),
  zoomInButton: $("#zoomInButton"),
  zoomValue: $("#zoomValue"),
  scaleStatusCard: $("#scaleStatusCard"),
  scaleConfidenceBadge: $("#scaleConfidenceBadge"),
  scaleStatusTitle: $("#scaleStatusTitle"),
  scaleStatusDetail: $("#scaleStatusDetail"),
  scalePixelsInput: $("#scalePixelsInput"),
  scaleUmInput: $("#scaleUmInput"),
  applyScaleButton: $("#applyScaleButton"),
  useFullImageButton: $("#useFullImageButton"),
  roiList: $("#roiList"),
  detectionProfile: $("#detectionProfile"),
  suggestParamsButton: $("#suggestParamsButton"),
  suggestionCard: $("#suggestionCard"),
  blurInput: $("#blurInput"),
  blurValue: $("#blurValue"),
  blockInput: $("#blockInput"),
  blockValue: $("#blockValue"),
  adaptiveCInput: $("#adaptiveCInput"),
  adaptiveCValue: $("#adaptiveCValue"),
  bridgeInput: $("#bridgeInput"),
  bridgeValue: $("#bridgeValue"),
  thresholdInvertInput: $("#thresholdInvertInput"),
  minLengthInput: $("#minLengthInput"),
  maxLengthInput: $("#maxLengthInput"),
  minSlendernessInput: $("#minSlendernessInput"),
  mergeDistanceInput: $("#mergeDistanceInput"),
  splitModeInput: $("#splitModeInput"),
  previewButton: $("#previewButton"),
  analyzeButton: $("#analyzeButton"),
  resultsDashboard: $("#resultsDashboard"),
  resultSummaryLine: $("#resultSummaryLine"),
  scrollToViewerButton: $("#scrollToViewerButton"),
  exportMenuButton: $("#exportMenuButton"),
  exportMenu: $("#exportMenu"),
  uniformityRing: $("#uniformityRing"),
  uniformityScore: $("#uniformityScore"),
  uniformityGrade: $("#uniformityGrade"),
  uniformityConfidence: $("#uniformityConfidence"),
  resultCount: $("#resultCount"),
  resultDispersed: $("#resultDispersed"),
  resultDispersedGrade: $("#resultDispersedGrade"),
  resultGridCv: $("#resultGridCv"),
  resultGridGrade: $("#resultGridGrade"),
  resultAggArea: $("#resultAggArea"),
  resultAggGrade: $("#resultAggGrade"),
  resultP90Width: $("#resultP90Width"),
  resultWidthGrade: $("#resultWidthGrade"),
  resultMeanLength: $("#resultMeanLength"),
  resultHybridScore: $("#resultHybridScore"),
  resultParticleCount: $("#resultParticleCount"),
  particleCountBadge: $("#particleCountBadge"),
  particleResultCount: $("#particleResultCount"),
  particleAreaRatio: $("#particleAreaRatio"),
  particleMeanDiameter: $("#particleMeanDiameter"),
  particleMeanConfidence: $("#particleMeanConfidence"),
  showParticleViewButton: $("#showParticleViewButton"),
  particleTableBody: $("#particleTableBody"),
  measurementCountBadge: $("#measurementCountBadge"),
  lengthHistogram: $("#lengthHistogram"),
  dispersionChart: $("#dispersionChart"),
  morphologyChart: $("#morphologyChart"),
  histogramCaption: $("#histogramCaption"),
  dispersionCaption: $("#dispersionCaption"),
  resultInsights: $("#resultInsights"),
  measurementSearch: $("#measurementSearch"),
  measurementTableBody: $("#measurementTableBody"),
  pointHeatmap: $("#pointHeatmap"),
  coverageHeatmap: $("#coverageHeatmap"),
  shadowHeatmap: $("#shadowHeatmap"),
  gridSizeLabel: $("#gridSizeLabel"),
  spatialStatsList: $("#spatialStatsList"),
  frameworkUniformity: $("#frameworkUniformity"),
  frameworkUniformityDetail: $("#frameworkUniformityDetail"),
  frameworkBundle: $("#frameworkBundle"),
  frameworkBundleDetail: $("#frameworkBundleDetail"),
  frameworkLength: $("#frameworkLength"),
  frameworkLengthDetail: $("#frameworkLengthDetail"),
  frameworkAgglomeration: $("#frameworkAgglomeration"),
  frameworkAgglomerationDetail: $("#frameworkAgglomerationDetail"),
  frameworkFormula: $("#frameworkFormula"),
  frameworkHybrid: $("#frameworkHybrid"),
  baseFilesInput: $("#baseFilesInput"),
  experimentFilesInput: $("#experimentFilesInput"),
  baseFileCount: $("#baseFileCount"),
  experimentFileCount: $("#experimentFileCount"),
  baseFileList: $("#baseFileList"),
  experimentFileList: $("#experimentFileList"),
  compareLimitText: $("#compareLimitText"),
  compareScaleUm: $("#compareScaleUm"),
  compareScalePixels: $("#compareScalePixels"),
  compareCenterFraction: $("#compareCenterFraction"),
  compareProfile: $("#compareProfile"),
  compareMinLength: $("#compareMinLength"),
  compareMinSlenderness: $("#compareMinSlenderness"),
  compareParameterSummary: $("#compareParameterSummary"),
  startComparisonButton: $("#startComparisonButton"),
  compareProgress: $("#compareProgress"),
  compareProgressMessage: $("#compareProgressMessage"),
  compareProgressPercent: $("#compareProgressPercent"),
  compareProgressBar: $("#compareProgressBar"),
  compareDashboard: $("#compareDashboard"),
  compareVerdict: $("#compareVerdict"),
  comparisonExportLink: $("#comparisonExportLink"),
  baseAnalyzedCount: $("#baseAnalyzedCount"),
  baseUniformityText: $("#baseUniformityText"),
  experimentAnalyzedCount: $("#experimentAnalyzedCount"),
  experimentUniformityText: $("#experimentUniformityText"),
  compareVerdictTitle: $("#compareVerdictTitle"),
  compareVerdictDetail: $("#compareVerdictDetail"),
  comparisonChart: $("#comparisonChart"),
  baseRepresentative: $("#baseRepresentative"),
  experimentRepresentative: $("#experimentRepresentative"),
  comparisonTableBody: $("#comparisonTableBody"),
  toastRegion: $("#toastRegion"),
};

const VIEW_LABELS = {
  original: "原始图像",
  enhanced: "对比度增强",
  binary: "二值预处理",
  skeleton_preview: "骨架预览",
  overlay: "轮廓检测结果",
  skeleton: "骨架测量结果",
  particles: "颗粒候选标注",
};

const PAGE_LABELS = {
  analyze: "单图工作台",
  compare: "组间对比",
  method: "方法与边界",
};

function formatNumber(value, digits = 2, fallback = "—") {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : fallback;
}

function formatPercent(value, digits = 1, fallback = "—") {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(digits)}%` : fallback;
}

function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

function cacheBust(url) {
  return `${url}${url.includes("?") ? "&" : "?"}cache=${Date.now()}`;
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value ?? "");
  return div.innerHTML;
}

function showToast(message, type = "info", timeout = 4200) {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === "success" ? "✓" : type === "error" ? "!" : "i"}</span><p>${escapeHtml(message)}</p>`;
  els.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), timeout);
}

async function api(url, options = {}) {
  const timeoutMs = options.timeoutMs || 0;
  const controller = new AbortController();
  let timer = null;
  if (timeoutMs > 0) timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`请求超时（超过 ${Math.round(timeoutMs / 1000)} 秒）。图片过大或服务器繁忙，请改用「打开图像」选择较小的图片。`);
    }
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
  }
  if (response.ok) return response;
  let message = `${response.status} ${response.statusText}`;
  try {
    const payload = await response.json();
    message = payload.detail || payload.message || message;
  } catch (_) {
    const text = await response.text();
    if (text) message = text.slice(0, 240);
  }
  throw new Error(message);
}

async function apiJson(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && typeof options.body === "string") headers["Content-Type"] = "application/json";
  const response = await api(url, { ...options, headers });
  return response.json();
}

function switchPage(page) {
  if (!PAGE_LABELS[page]) return;
  state.page = page;
  $$('[data-page-panel]').forEach((panel) => panel.classList.toggle("active", panel.dataset.pagePanel === page));
  $$('[data-page]').forEach((button) => button.classList.toggle("active", button.dataset.page === page));
  els.currentSection.textContent = PAGE_LABELS[page];
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (page === "compare" && state.compare.result) window.setTimeout(renderComparisonCharts, 80);
  if (page === "analyze" && state.result) window.setTimeout(renderResultCharts, 80);
}

function setSessionState(text, kind = "ready", detail = null) {
  els.sessionStateText.textContent = text;
  els.sessionStateDot.className = `status-dot ${kind}`;
  if (detail) els.sessionExpiryText.textContent = detail;
}

function setViewerBusy(visible, title = "正在分析", detail = "请稍候") {
  els.viewerBusy.classList.toggle("hidden", !visible);
  els.viewerBusyTitle.textContent = title;
  els.viewerBusyDetail.textContent = detail;
}

async function bootstrap() {
  bindEvents();
  updateParameterOutputs();
  try {
    const [health, config] = await Promise.all([
      apiJson("/api/v1/health"),
      apiJson("/api/v1/config"),
    ]);
    state.health = health;
    state.config = config;
    els.railStatusDot.className = "status-dot ready";
    els.railStatusText.textContent = "完整分析引擎在线";
    els.railVersion.textContent = `CNTAnalyzer · ${health.version}`;
    els.uploadLimitText.textContent = `PNG、JPEG、TIFF、BMP、WebP · 单张不超过 ${config.max_upload_mb} MB`;
    els.compareLimitText.textContent = `${config.max_batch_files_per_group} 张、整批不超过 ${config.max_batch_total_mb} MB`;
    if (state.session) {
      els.sessionExpiryText.textContent = `${Math.round(config.session_ttl_seconds / 60)} 分钟无操作后清理`;
    } else {
      els.sessionExpiryText.textContent = `打开图像后创建 ${Math.round(config.session_ttl_seconds / 60)} 分钟临时会话`;
    }
  } catch (error) {
    els.railStatusDot.className = "status-dot error";
    els.railStatusText.textContent = "分析引擎未连接";
    els.railVersion.textContent = "请检查远端服务";
    showToast(`无法连接完整分析引擎：${error.message}`, "error", 8000);
  }
}

async function uploadSingleImage(file) {
  if (!file) return;
  const maxBytes = (state.config?.max_upload_mb || 25) * 1024 * 1024;
  if (file.size > maxBytes) {
    showToast(`文件超过 ${state.config?.max_upload_mb || 25} MB 限制。`, "error");
    return;
  }
  if (state.session) await deleteCurrentSession();
  els.resultsDashboard.classList.add("hidden");
  els.sessionFileName.textContent = file.name;
  els.sessionFileMeta.textContent = `${formatBytes(file.size)} · 正在上传`;
  const isLarge = file.size > 3 * 1024 * 1024;
  setSessionState("正在建立会话", "checking", "上传并检测比例尺");
  setViewerBusy(true, "正在建立完整分析会话", isLarge ? "大图上传与解码较慢，请稍候（比例尺检测 + 参数评估）" : "解码原图、检测比例尺并评估预处理参数");

  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const response = await apiJson("/api/v1/sessions", { method: "POST", body: form, timeoutMs: 60000 });
    state.session = response;
    state.activeRoiName = null;
    state.result = null;
    state.pendingScaleLine = null;
    state.zoom = 1;
    state.currentView = "original";
    els.sessionFileName.textContent = response.filename;
    els.sessionFileMeta.textContent = `${response.width} × ${response.height} px · ${formatBytes(file.size)}`;
    setSessionState("会话就绪", "ready", `${Math.round(response.expires_in_seconds / 60)} 分钟无操作后清理`);
    applyScaleStatus(response.scale_status);
    applySuggestion(response.suggestion, true);
    renderRois([]);
    enableView("binary", false);
    enableView("skeleton_preview", false);
    enableView("overlay", false);
    enableView("skeleton", false);
    enableView("particles", false);
    await loadView("original", { fit: true });
    showToast("图像已载入，比例尺与参数建议已生成。", "success");
  } catch (error) {
    resetSingleWorkspace();
    showToast(`图像载入失败：${error.message}`, "error", 7000);
  } finally {
    setViewerBusy(false);
    els.singleImageInput.value = "";
  }
}

async function deleteCurrentSession() {
  if (!state.session?.session_id) return;
  try {
    await fetch(`/api/v1/sessions/${state.session.session_id}`, { method: "DELETE", keepalive: true });
  } catch (_) {
    // The session TTL remains the final cleanup guard.
  }
}

function resetSingleWorkspace() {
  cancelDrawing();
  state.session = null;
  state.activeRoiName = null;
  state.result = null;
  state.pendingScaleLine = null;
  state.currentView = "original";
  state.zoom = 1;
  state.viewSequence += 1;
  els.resultsDashboard.classList.add("hidden");
  els.imageTransform.classList.remove("ready");
  els.analysisImage.removeAttribute("src");
  els.viewerEmpty.classList.remove("hidden");
  els.singleDropZone.classList.remove("dragging");
  els.sessionFileName.textContent = "未打开图像";
  els.sessionFileMeta.textContent = "选择文件或直接拖放到画布";
  const ttlMinutes = Math.round((state.config?.session_ttl_seconds || 1800) / 60);
  setSessionState("工具就绪", "ready", `打开图像后创建 ${ttlMinutes} 分钟临时会话`);
  els.activeViewLabel.textContent = VIEW_LABELS.original;
  els.activeRoiLabel.textContent = "分析范围：全图";
  $$('[data-view]', els.viewTabs).forEach((button) => {
    button.classList.toggle("active", button.dataset.view === "original");
    if (!["original", "enhanced"].includes(button.dataset.view)) button.disabled = true;
  });
  els.roiList.innerHTML = '<p class="empty-list">尚未创建 ROI</p>';
}

function applyScaleStatus(status) {
  const source = status?.source || "unset";
  const confidence = status?.confidence || "low";
  const applied = ["auto_detected", "manual", "batch_manual"].includes(source);
  els.scaleStatusCard.classList.toggle("success", applied);
  els.scaleStatusCard.classList.toggle("warning", !applied);
  els.scaleConfidenceBadge.textContent = applied ? (confidence === "high" ? "高置信度" : "已应用") : "待确认";
  if (source === "auto_detected") {
    els.scaleStatusTitle.textContent = "已自动检测并应用比例尺";
  } else if (source === "manual") {
    els.scaleStatusTitle.textContent = "已应用手动比例尺";
  } else {
    els.scaleStatusTitle.textContent = "未可靠检测到比例尺";
  }
  const px = Number(status?.pixels);
  const um = Number(status?.micrometers);
  const ratio = Number(status?.um_per_pixel);
  els.scaleStatusDetail.textContent = Number.isFinite(px) && Number.isFinite(um)
    ? `${px.toFixed(1)} px = ${um.toFixed(3)} μm · ${ratio.toFixed(6)} μm/px${status.exclusion_enabled ? " · 已排除标尺区域" : ""}`
    : `当前临时比例 ${Number.isFinite(ratio) ? ratio.toFixed(6) : "0.100000"} μm/px，建议画线确认。`;
  if (Number.isFinite(px) && px > 0) els.scalePixelsInput.value = px.toFixed(2);
  if (Number.isFinite(um) && um > 0) els.scaleUmInput.value = um.toFixed(3).replace(/\.0+$/, "");
}

async function applyScale() {
  if (!state.session) return;
  const pixels = Number(els.scalePixelsInput.value);
  const micrometers = Number(els.scaleUmInput.value);
  if (!(pixels > 0) || !(micrometers > 0)) {
    showToast("比例尺像素数和微米数必须大于 0。", "error");
    return;
  }
  const body = { pixels, micrometers };
  if (state.pendingScaleLine) {
    body.start = state.pendingScaleLine.start;
    body.end = state.pendingScaleLine.end;
  }
  setViewerBusy(true, "正在应用比例尺", "重建排除区域与参数建议");
  try {
    const response = await apiJson(`/api/v1/sessions/${state.session.session_id}/scale`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    state.session.image_urls = response.image_urls || state.session.image_urls;
    applyScaleStatus(response.scale_status);
    applySuggestion(response.suggestion, true);
    state.result = null;
    els.resultsDashboard.classList.add("hidden");
    disableDerivedViews();
    state.pendingScaleLine = null;
    drawInteractionOverlay();
    await loadView("original");
    showToast("比例尺已应用，已有测量将在下次检测中按新尺度重算。", "success");
  } catch (error) {
    showToast(`比例尺应用失败：${error.message}`, "error");
  } finally {
    setViewerBusy(false);
  }
}

function switchDrawer(name) {
  $$("[data-drawer]").forEach((button) => button.classList.toggle("active", button.dataset.drawer === name));
  $$("[data-drawer-panel]").forEach((panel) => panel.classList.toggle("active", panel.dataset.drawerPanel === name));
}

function renderRois(rois) {
  state.session.rois = rois;
  els.roiList.innerHTML = "";
  if (!rois.length) {
    els.roiList.innerHTML = '<p class="empty-list">尚未创建 ROI</p>';
  } else {
    rois.forEach((roi) => {
      const row = document.createElement("div");
      row.className = `roi-item${state.activeRoiName === roi.name ? " active" : ""}`;
      row.dataset.roiName = roi.name;
      row.innerHTML = `<span class="roi-swatch"></span><div><strong>${escapeHtml(roi.name)}</strong><small>${roi.width} × ${roi.height} px · (${roi.x}, ${roi.y})</small></div><button class="delete-roi" type="button" aria-label="删除 ${escapeHtml(roi.name)}">×</button>`;
      row.addEventListener("click", (event) => {
        if (event.target.closest(".delete-roi")) return;
        selectRoi(roi.name);
      });
      $(".delete-roi", row).addEventListener("click", () => deleteRoi(roi.name));
      els.roiList.append(row);
    });
  }
  els.useFullImageButton.classList.toggle("active", !state.activeRoiName);
  els.useFullImageButton.querySelector("b").textContent = !state.activeRoiName ? "✓" : "";
  updateActiveRoiLabel();
  drawInteractionOverlay();
}

function selectRoi(name) {
  state.activeRoiName = name || null;
  state.result = null;
  els.resultsDashboard.classList.add("hidden");
  disableDerivedViews();
  renderRois(state.session?.rois || []);
  switchDrawer("parameters");
  requestSuggestion(false);
}

async function createRoi(rect) {
  if (!state.session) return;
  setViewerBusy(true, "正在创建 ROI", "保存画布选择区域");
  try {
    const response = await apiJson(`/api/v1/sessions/${state.session.session_id}/rois`, {
      method: "POST",
      body: JSON.stringify({
        name: `ROI_${(state.session.rois?.length || 0) + 1}`,
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
      }),
    });
    state.session.image_urls = response.image_urls || state.session.image_urls;
    state.activeRoiName = response.roi.name;
    state.result = null;
    els.resultsDashboard.classList.add("hidden");
    disableDerivedViews();
    renderRois(response.rois);
    switchDrawer("roi");
    showToast(`${response.roi.name} 已创建并设为当前分析范围。`, "success");
    await requestSuggestion(false);
  } catch (error) {
    showToast(`ROI 创建失败：${error.message}`, "error");
  } finally {
    setViewerBusy(false);
  }
}

async function deleteRoi(name) {
  if (!state.session) return;
  try {
    const response = await apiJson(`/api/v1/sessions/${state.session.session_id}/rois/${encodeURIComponent(name)}`, { method: "DELETE" });
    state.session.image_urls = response.image_urls || state.session.image_urls;
    if (state.activeRoiName === name) state.activeRoiName = null;
    renderRois(response.rois);
    state.result = null;
    els.resultsDashboard.classList.add("hidden");
    disableDerivedViews();
    showToast(`${name} 已删除。`, "success");
  } catch (error) {
    showToast(`ROI 删除失败：${error.message}`, "error");
  }
}

function updateActiveRoiLabel() {
  const roi = state.session?.rois?.find((item) => item.name === state.activeRoiName);
  els.activeRoiLabel.textContent = roi ? `分析范围：${roi.name} · ${roi.width} × ${roi.height} px` : "分析范围：全图";
}

async function requestSuggestion(showBusy = true) {
  if (!state.session) return;
  if (showBusy) setViewerBusy(true, "正在分析图像特征", "评估噪点、对比度、边缘与前景比例");
  try {
    const suggestion = await apiJson(`/api/v1/sessions/${state.session.session_id}/suggest`, {
      method: "POST",
      body: JSON.stringify({
        roi_name: state.activeRoiName,
        detection_profile: els.detectionProfile.value,
      }),
    });
    applySuggestion(suggestion, true);
    showToast("已根据当前分析范围更新预处理参数。", "success");
  } catch (error) {
    showToast(`参数建议失败：${error.message}`, "error");
  } finally {
    if (showBusy) setViewerBusy(false);
  }
}

function applySuggestion(suggestion, updateInputs) {
  if (!suggestion) return;
  state.suggestion = suggestion;
  if (updateInputs) {
    if (Number.isFinite(Number(suggestion.blur_kernel))) els.blurInput.value = suggestion.blur_kernel;
    if (Number.isFinite(Number(suggestion.adaptive_block))) els.blockInput.value = suggestion.adaptive_block;
    if (Number.isFinite(Number(suggestion.adaptive_c))) els.adaptiveCInput.value = suggestion.adaptive_c;
    updateParameterOutputs();
  }
  const metrics = suggestion.metrics || {};
  const detail = [
    Number.isFinite(Number(metrics.noise_score)) ? `噪点 ${Number(metrics.noise_score).toFixed(2)}` : null,
    Number.isFinite(Number(metrics.edge_density)) ? `边缘 ${Number(metrics.edge_density).toFixed(3)}` : null,
    Number.isFinite(Number(metrics.foreground_ratio)) ? `前景 ${(Number(metrics.foreground_ratio) * 100).toFixed(1)}%` : null,
  ].filter(Boolean).join(" · ");
  els.suggestionCard.innerHTML = `<span>AI</span><p><strong>${escapeHtml(suggestion.reason_summary || "参数建议已生成")}</strong><small>${escapeHtml(detail || "已采用校准基线参数")}</small></p>`;
}

function getPreprocessPayload(generateSkeleton = true) {
  return {
    roi_name: state.activeRoiName,
    blur_kernel: Number(els.blurInput.value),
    adaptive_block: Number(els.blockInput.value),
    adaptive_c: Number(els.adaptiveCInput.value),
    bridge_strength: Number(els.bridgeInput.value),
    threshold_invert: els.thresholdInvertInput.checked,
    generate_skeleton: Boolean(generateSkeleton),
  };
}

function getDetectionPayload() {
  return {
    min_length_um: Number(els.minLengthInput.value),
    max_length_um: Number(els.maxLengthInput.value),
    min_slenderness: Number(els.minSlendernessInput.value),
    detection_profile: els.detectionProfile.value,
    split_mode: els.splitModeInput.value,
    merge_distance_px: Number(els.mergeDistanceInput.value),
  };
}

function validateParameters() {
  const p = getPreprocessPayload(true);
  const d = getDetectionPayload();
  if (p.blur_kernel % 2 === 0 || p.adaptive_block % 2 === 0) throw new Error("模糊核和自适应块大小必须为奇数");
  if (!(d.max_length_um > 0) || d.max_length_um < d.min_length_um) throw new Error("最大长度必须大于等于最小长度");
  return { preprocess: p, detection: d };
}

async function runPreview({ skeleton = true, quiet = false } = {}) {
  if (!state.session) return;
  const sequence = ++state.requestSequence;
  if (!quiet) setViewerBusy(true, skeleton ? "正在生成骨架预览" : "正在更新二值预览", skeleton ? "执行同源预处理与骨架化" : "快速重建二值分割");
  try {
    const response = await apiJson(`/api/v1/sessions/${state.session.session_id}/preprocess`, {
      method: "POST",
      body: JSON.stringify(getPreprocessPayload(skeleton)),
    });
    if (sequence !== state.requestSequence) return;
    state.session.image_urls = response.image_urls || state.session.image_urls;
    state.result = null;
    els.resultsDashboard.classList.add("hidden");
    enableView("binary", true);
    enableView("skeleton_preview", skeleton);
    enableView("overlay", false);
    enableView("skeleton", false);
    enableView("particles", false);
    await loadView(skeleton ? "skeleton_preview" : "binary");
    if (!quiet) showToast(skeleton ? "骨架预览已生成，可确认后开始完整检测。" : "二值预览已更新。", "success");
  } catch (error) {
    if (sequence === state.requestSequence) showToast(`预处理失败：${error.message}`, "error");
  } finally {
    if (!quiet && sequence === state.requestSequence) setViewerBusy(false);
  }
}

function scheduleLivePreview() {
  updateParameterOutputs();
  if (!state.session) return;
  window.clearTimeout(state.previewTimer);
  state.previewTimer = window.setTimeout(() => runPreview({ skeleton: false, quiet: true }), 650);
}

async function runFullAnalysis() {
  if (!state.session) return;
  let payload;
  try {
    payload = validateParameters();
  } catch (error) {
    showToast(error.message, "error");
    return;
  }
  window.clearTimeout(state.previewTimer);
  ++state.requestSequence;
  const largePixelCount = state.session && state.session.width * state.session.height > 6_000_000;
  setSessionState("正在完整检测", "checking", largePixelCount ? "大图分析较慢（超过 600 万像素），请耐心等待" : "执行骨架路径与空间统计");
  setViewerBusy(true, "正在完整检测 CNT", largePixelCount ? "高分辨率图像分析较慢，可能需要 30 秒以上，请勿重复点击" : "骨架路径、宽度、空间分布与团聚判定同步计算");
  try {
    payload.preprocess.roi_name = state.activeRoiName;
    const result = await apiJson(`/api/v1/sessions/${state.session.session_id}/analyze`, {
      method: "POST",
      body: JSON.stringify({
        roi_name: state.activeRoiName,
        preprocess: payload.preprocess,
        detection: payload.detection,
      }),
      timeoutMs: 180000,
    });
    state.session.image_urls = result.image_urls || state.session.image_urls;
    state.result = result;
    enableView("binary", true);
    enableView("skeleton_preview", true);
    enableView("overlay", true);
    enableView("skeleton", true);
    enableView("particles", true);
    renderResults(result);
    setSessionState("分析完成", "ready", `${result.stats.count || 0} 个 CNT · 可继续调参复核`);
    els.resultsDashboard.classList.remove("hidden");
    showToast(`完整检测完成，共测量 ${result.stats.count || 0} 个 CNT。`, "success");
    loadView("skeleton").catch(() => {});
  } catch (error) {
    setSessionState("分析失败", "error", error.message);
    showToast(`完整检测失败：${error.message}`, "error", 7000);
  } finally {
    setViewerBusy(false);
  }
}

function enableView(view, enabled) {
  const button = $(`[data-view="${view}"]`, els.viewTabs);
  if (button) button.disabled = !enabled;
}

function disableResultViews() {
  enableView("overlay", false);
  enableView("skeleton", false);
  enableView("particles", false);
}

function disableDerivedViews() {
  enableView("binary", false);
  enableView("skeleton_preview", false);
  disableResultViews();
}

async function loadView(view, { fit = false } = {}) {
  if (!state.session) return;
  const url = state.session.image_urls?.[view];
  if (!url) return;
  const previousView = state.currentView;
  const viewSequence = ++state.viewSequence;
  state.currentView = view;
  $$("[data-view]", els.viewTabs).forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  els.activeViewLabel.textContent = VIEW_LABELS[view] || view;

  const resolvedUrl = new URL(url, window.location.href).href;
  if (
    els.analysisImage.complete &&
    els.analysisImage.naturalWidth > 0 &&
    els.analysisImage.currentSrc === resolvedUrl
  ) {
    els.viewerEmpty.classList.add("hidden");
    els.imageTransform.classList.add("ready");
    if (fit) state.zoom = 1;
    fitImageToViewport(false);
    return;
  }

  const hasVisibleImage = els.imageTransform.classList.contains("ready") && els.analysisImage.naturalWidth > 0;
  if (!hasVisibleImage) {
    els.viewerEmpty.classList.remove("hidden");
    els.imageTransform.classList.remove("ready");
  }

  await new Promise((resolve, reject) => {
    const preload = new Image();
    preload.onload = resolve;
    preload.onerror = () => reject(new Error(`${VIEW_LABELS[view] || view}尚未生成`));
    preload.src = resolvedUrl;
  }).then(() => {
    if (viewSequence !== state.viewSequence) return false;
    return new Promise((resolve, reject) => {
    const onLoad = () => {
      els.analysisImage.removeEventListener("error", onError);
      resolve(true);
    };
    const onError = () => {
      els.analysisImage.removeEventListener("load", onLoad);
      reject(new Error(`${VIEW_LABELS[view] || view}尚未生成`));
    };
    els.analysisImage.addEventListener("load", onLoad, { once: true });
    els.analysisImage.addEventListener("error", onError, { once: true });
    els.analysisImage.src = resolvedUrl;
    });
  }).then((applied) => {
    if (!applied || viewSequence !== state.viewSequence) return;
    els.viewerEmpty.classList.add("hidden");
    els.imageTransform.classList.add("ready");
    if (fit) state.zoom = 1;
    fitImageToViewport(false);
  }).catch((error) => {
    if (viewSequence !== state.viewSequence) return;
    state.currentView = previousView;
    $$("[data-view]", els.viewTabs).forEach((button) => button.classList.toggle("active", button.dataset.view === previousView));
    els.activeViewLabel.textContent = VIEW_LABELS[previousView] || previousView;
    showToast(error.message, "error");
  });
}

function fitImageToViewport(resetZoom = true) {
  if (!els.analysisImage.naturalWidth || !els.analysisImage.naturalHeight) return;
  const availableWidth = Math.max(100, els.imageViewport.clientWidth - 30);
  const availableHeight = Math.max(100, els.imageViewport.clientHeight - 30);
  state.fitScale = Math.min(
    availableWidth / els.analysisImage.naturalWidth,
    availableHeight / els.analysisImage.naturalHeight,
    1,
  );
  if (resetZoom) state.zoom = 1;
  state.renderWidth = Math.max(1, Math.round(els.analysisImage.naturalWidth * state.fitScale));
  state.renderHeight = Math.max(1, Math.round(els.analysisImage.naturalHeight * state.fitScale));
  els.imageTransform.style.width = `${state.renderWidth}px`;
  els.imageTransform.style.height = `${state.renderHeight}px`;
  els.imageTransform.style.left = "50%";
  els.imageTransform.style.top = "50%";
  els.imageTransform.style.transform = `translate(-50%, -50%) scale(${state.zoom})`;
  els.analysisImage.style.width = `${state.renderWidth}px`;
  els.analysisImage.style.height = `${state.renderHeight}px`;
  resizeInteractionCanvas();
  updateZoomLabel();
}

function updateZoomLabel() {
  const percent = Math.round(state.fitScale * state.zoom * 100);
  els.zoomValue.textContent = state.zoom === 1 ? `适应 · ${percent}%` : `${percent}%`;
}

function setZoom(next) {
  state.zoom = Math.max(.5, Math.min(4, next));
  els.imageTransform.style.transform = `translate(-50%, -50%) scale(${state.zoom})`;
  updateZoomLabel();
}

function resizeInteractionCanvas() {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  els.interactionCanvas.width = Math.max(1, Math.round(state.renderWidth * ratio));
  els.interactionCanvas.height = Math.max(1, Math.round(state.renderHeight * ratio));
  els.interactionCanvas.style.width = `${state.renderWidth}px`;
  els.interactionCanvas.style.height = `${state.renderHeight}px`;
  const ctx = els.interactionCanvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawInteractionOverlay();
}

function canvasPoint(event) {
  const rect = els.interactionCanvas.getBoundingClientRect();
  const nx = (event.clientX - rect.left) / rect.width;
  const ny = (event.clientY - rect.top) / rect.height;
  return {
    x: Math.max(0, Math.min(els.analysisImage.naturalWidth, nx * els.analysisImage.naturalWidth)),
    y: Math.max(0, Math.min(els.analysisImage.naturalHeight, ny * els.analysisImage.naturalHeight)),
  };
}

function originalToCanvas(point) {
  return {
    x: point.x / els.analysisImage.naturalWidth * state.renderWidth,
    y: point.y / els.analysisImage.naturalHeight * state.renderHeight,
  };
}

function startDrawing(mode) {
  if (!state.session) return;
  state.drawMode = mode;
  state.drawStart = null;
  state.drawCurrent = null;
  els.drawScaleButton.classList.toggle("active", mode === "scale");
  els.drawRoiButton.classList.toggle("active", mode === "roi");
  els.drawHint.classList.remove("hidden");
  els.drawHintText.textContent = mode === "scale" ? "沿比例尺线段拖动" : "拖动绘制 ROI 矩形";
}

function cancelDrawing() {
  state.drawMode = null;
  state.drawStart = null;
  state.drawCurrent = null;
  els.drawScaleButton?.classList.remove("active");
  els.drawRoiButton?.classList.remove("active");
  els.drawHint?.classList.add("hidden");
  drawInteractionOverlay();
}

function drawInteractionOverlay() {
  const canvas = els.interactionCanvas;
  if (!canvas.width || !state.renderWidth) return;
  const ctx = canvas.getContext("2d");
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, state.renderWidth, state.renderHeight);

  (state.session?.rois || []).forEach((roi) => {
    const start = originalToCanvas({ x: roi.x, y: roi.y });
    const end = originalToCanvas({ x: roi.x + roi.width, y: roi.y + roi.height });
    const active = roi.name === state.activeRoiName;
    ctx.save();
    ctx.strokeStyle = active ? "#d7f276" : "rgba(255, 210, 70, .85)";
    ctx.fillStyle = active ? "rgba(215, 242, 118, .08)" : "rgba(255, 210, 70, .04)";
    ctx.lineWidth = active ? 2.2 : 1.4;
    ctx.setLineDash(active ? [] : [5, 4]);
    ctx.fillRect(start.x, start.y, end.x - start.x, end.y - start.y);
    ctx.strokeRect(start.x, start.y, end.x - start.x, end.y - start.y);
    ctx.setLineDash([]);
    ctx.fillStyle = active ? "#d7f276" : "#ffd246";
    ctx.font = "700 9px Segoe UI, sans-serif";
    ctx.fillText(roi.name, start.x + 5, Math.max(11, start.y - 5));
    ctx.restore();
  });

  if (state.pendingScaleLine) {
    drawScaleLine(ctx, state.pendingScaleLine.start, state.pendingScaleLine.end, false);
  }
  if (state.drawStart && state.drawCurrent) {
    if (state.drawMode === "scale") {
      drawScaleLine(ctx, state.drawStart, state.drawCurrent, true);
    } else if (state.drawMode === "roi") {
      const a = originalToCanvas(state.drawStart);
      const b = originalToCanvas(state.drawCurrent);
      ctx.save();
      ctx.strokeStyle = "#d7f276";
      ctx.fillStyle = "rgba(215, 242, 118, .09)";
      ctx.lineWidth = 2;
      ctx.fillRect(a.x, a.y, b.x - a.x, b.y - a.y);
      ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
      ctx.restore();
    }
  }
}

function drawScaleLine(ctx, start, end, provisional) {
  const a = originalToCanvas(start);
  const b = originalToCanvas(end);
  const pixels = Math.hypot(end.x - start.x, end.y - start.y);
  ctx.save();
  ctx.strokeStyle = provisional ? "#d7f276" : "#76e0d0";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  for (const point of [a, b]) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = provisional ? "#d7f276" : "#76e0d0";
    ctx.fill();
  }
  ctx.font = "700 9px Segoe UI, sans-serif";
  const label = `${pixels.toFixed(1)} px`;
  const x = (a.x + b.x) / 2;
  const y = (a.y + b.y) / 2 - 9;
  const width = ctx.measureText(label).width + 10;
  ctx.fillStyle = "rgba(10, 30, 27, .82)";
  ctx.fillRect(x - width / 2, y - 10, width, 16);
  ctx.fillStyle = "white";
  ctx.fillText(label, x - width / 2 + 5, y + 1);
  ctx.restore();
}

function onCanvasPointerDown(event) {
  if (!state.drawMode) return;
  els.interactionCanvas.setPointerCapture(event.pointerId);
  state.drawStart = canvasPoint(event);
  state.drawCurrent = state.drawStart;
  drawInteractionOverlay();
}

function onCanvasPointerMove(event) {
  if (!state.drawMode || !state.drawStart) return;
  state.drawCurrent = canvasPoint(event);
  drawInteractionOverlay();
}

function onCanvasPointerUp(event) {
  if (!state.drawMode || !state.drawStart) return;
  const end = canvasPoint(event);
  const start = state.drawStart;
  const mode = state.drawMode;
  if (mode === "scale") {
    const pixels = Math.hypot(end.x - start.x, end.y - start.y);
    if (pixels >= 2) {
      state.pendingScaleLine = { start, end };
      els.scalePixelsInput.value = pixels.toFixed(2);
      switchDrawer("calibration");
      showToast("比例尺线段已测量，请填写对应微米数并应用。", "info");
    }
  } else {
    const rect = {
      x: Math.min(start.x, end.x),
      y: Math.min(start.y, end.y),
      width: Math.abs(end.x - start.x),
      height: Math.abs(end.y - start.y),
    };
    if (rect.width >= 10 && rect.height >= 10) createRoi(rect);
    else showToast("ROI 宽高至少需要 10 像素。", "error");
  }
  cancelDrawing();
}

function renderResults(result) {
  const stats = result.stats || {};
  const spatial = stats.spatial_distribution || {};
  const uniformity = spatial.uniformity_scores || {};
  const core = result.core_metrics || {};
  const framework = result.evaluation_framework || {};
  const dispersed = result.dispersed_stats || {};
  const particles = result.particle_statistics || {};
  const score = Number(core.uniformity_score || uniformity.overall || 0);
  const grade = uniformity.grade || (score >= 80 ? "优秀" : score >= 65 ? "良好" : score >= 50 ? "一般" : "需复核");

  els.uniformityScore.textContent = Math.round(score);
  els.uniformityRing.style.setProperty("--score-angle", `${Math.max(0, Math.min(100, score)) * 3.6}deg`);
  els.uniformityGrade.textContent = grade;
  els.uniformityConfidence.textContent = `${uniformity.confidence || "样本自适应"}置信度 · 越高越均匀`;
  els.resultCount.textContent = core.total_count ?? stats.count ?? 0;
  els.resultDispersed.textContent = formatPercent(core.dispersed_ratio, 1);
  els.resultDispersedGrade.textContent = `评级 ${core.dispersed_ratio_grade || "—"}`;
  els.resultGridCv.textContent = formatNumber(core.grid_density_cv, 3);
  els.resultGridGrade.textContent = `评级 ${core.grid_density_cv_grade || "—"}`;
  els.resultAggArea.textContent = formatPercent(core.agglomerated_area_ratio, 1);
  els.resultAggGrade.textContent = `评级 ${core.agglomerated_area_ratio_grade || "—"}`;
  els.resultP90Width.textContent = `${formatNumber(core.width_p90_um, 3)} μm`;
  els.resultWidthGrade.textContent = `评级 ${core.width_p90_um_grade || "—"}`;
  els.resultMeanLength.textContent = formatNumber(core.skeleton_length_mean_um, 2);
  els.resultHybridScore.textContent = formatNumber(framework.hybrid_score, 1);
  els.resultParticleCount.textContent = particles.count ?? 0;
  els.particleCountBadge.textContent = particles.count ?? 0;
  els.particleResultCount.textContent = particles.count ?? 0;
  els.particleAreaRatio.textContent = formatPercent(particles.area_ratio, 2);
  els.particleMeanDiameter.textContent = formatNumber(particles.equivalent_diameter_mean_um, 3);
  els.particleMeanConfidence.textContent = formatPercent(particles.confidence_mean, 1);
  els.measurementCountBadge.textContent = result.measurement_count ?? result.measurements?.length ?? 0;
  els.resultSummaryLine.textContent = `${result.name} · ${result.analysis_roi?.name || "全图"} · ${result.measurement_count ?? result.measurements?.length ?? 0} 个 CNT 测量 · ${particles.count || 0} 个颗粒候选`;
  els.histogramCaption.textContent = `均值 ${formatNumber(stats.length_mean, 2)} μm · SD ${formatNumber(stats.length_std, 2)} μm`;
  els.dispersionCaption.textContent = `${dispersed.dispersed_count || 0} 分散 · ${dispersed.agglomerated_count || 0} 团聚`;
  els.gridSizeLabel.textContent = `${spatial.grid_size || 0} × ${spatial.grid_size || 0}`;

  $$('[data-export]').forEach((link) => {
    link.href = result.export_urls?.[link.dataset.export] || "#";
  });
  renderInsights(result);
  renderParticles(result.particle_measurements || []);
  renderSpatialStats(spatial);
  renderFramework(framework);
  renderResultCharts();
  ensureMeasurementsLoaded();
}

let measurementsLoadPromise = null;

async function ensureMeasurementsLoaded() {
  if (!state.result || state.result.measurements?.length) return;
  const url = state.result.measurements_url;
  if (!url) return;
  if (measurementsLoadPromise) return measurementsLoadPromise;
  measurementsLoadPromise = (async () => {
    try {
      const detail = await apiJson(url);
      if (!state.result || !detail?.measurements) return;
      state.result.measurements = detail.measurements;
      renderMeasurementTable();
      drawMorphologyChart();
    } catch (error) {
      // Non-blocking: summary stays rendered; the table falls back to its empty state.
      if (window.__opcntDebug) console.warn("measurements lazy-load failed:", error.message);
    } finally {
      measurementsLoadPromise = null;
    }
  })();
  return measurementsLoadPromise;
}

function renderParticles(particles) {
  els.particleTableBody.innerHTML = particles.map((item) => `<tr>
    <td>P${Number(item.id) + 1}</td>
    <td>${formatNumber(item.centroid?.x, 1)} / ${formatNumber(item.centroid?.y, 1)}</td>
    <td>${formatNumber(item.area_um2, 4)}</td>
    <td>${formatNumber(item.equivalent_diameter_um, 4)}</td>
    <td>${formatNumber(item.major_axis_um, 4)} / ${formatNumber(item.minor_axis_um, 4)}</td>
    <td>${formatNumber(item.circularity, 3)}</td>
    <td>${formatNumber(item.solidity, 3)}</td>
    <td>${formatNumber(item.aspect_ratio, 3)}</td>
    <td>${formatNumber(item.mean_local_contrast, 3)}</td>
    <td>${formatPercent(item.confidence, 1)}</td>
  </tr>`).join("") || '<tr><td colspan="10">当前分析区域未检出颗粒候选</td></tr>';
}

function renderInsights(result) {
  const core = result.core_metrics || {};
  const spatial = result.stats?.spatial_distribution || {};
  const items = [];
  items.push({
    type: core.dispersed_ratio >= .8 ? "good" : core.dispersed_ratio >= .6 ? "info" : "warn",
    icon: core.dispersed_ratio >= .8 ? "✓" : "!",
    text: `分散比例为 ${formatPercent(core.dispersed_ratio, 1)}，${core.dispersed_ratio >= .8 ? "分散 CNT 占比较高" : "建议结合团聚热点复核局部区域"}。`,
  });
  items.push({
    type: core.grid_density_cv < .6 ? "good" : "warn",
    icon: core.grid_density_cv < .6 ? "✓" : "!",
    text: `网格 CV 为 ${formatNumber(core.grid_density_cv, 3)}，${core.grid_density_cv < .6 ? "宏观空间占据较均匀" : "不同区域的 CNT 密度存在明显差异"}。`,
  });
  items.push({
    type: "info",
    icon: "i",
    text: `Moran's I 为 ${formatNumber(spatial.morans_i, 3)}，最近邻指数为 ${formatNumber(spatial.nearest_neighbor_index, 3)}；数量多不等于空间均匀。`,
  });
  els.resultInsights.innerHTML = items.map((item) => `<li class="${item.type}"><span>${item.icon}</span><p>${escapeHtml(item.text)}</p></li>`).join("");
}

function renderMeasurementTable() {
  if (!state.result) return;
  const query = els.measurementSearch.value.trim().toLowerCase();
  const classification = new Map();
  (state.result.dispersed_stats?.dispersed_ids || []).forEach((id) => classification.set(Number(id), "dispersed"));
  (state.result.dispersed_stats?.agglomerated_ids || []).forEach((id) => classification.set(Number(id), "agglomerated"));
  const { key, direction } = state.measurementSort;
  const rows = [...(state.result.measurements || [])]
    .filter((item) => !query || Object.values(item).some((value) => String(value ?? "").toLowerCase().includes(query)))
    .sort((a, b) => {
      const av = Number(a[key]);
      const bv = Number(b[key]);
      if (Number.isFinite(av) && Number.isFinite(bv)) return (av - bv) * direction;
      return String(a[key] ?? "").localeCompare(String(b[key] ?? "")) * direction;
    });
  els.measurementTableBody.innerHTML = rows.map((item) => {
    const kind = classification.get(Number(item.id)) || "dispersed";
    return `<tr>
      <td>${item.id}</td>
      <td>${formatNumber(item.length_um, 4)}</td>
      <td>${formatNumber(item.width_mean_um, 4)}</td>
      <td>${formatNumber(item.width_median_um, 4)}</td>
      <td>${formatNumber(item.width_iqr_um, 4)}</td>
      <td>${formatNumber(item.slenderness, 3)}</td>
      <td><span class="classification-badge ${kind}">${kind === "agglomerated" ? "团聚" : "分散"}</span></td>
    </tr>`;
  }).join("") || '<tr><td colspan="7">没有匹配的测量结果</td></tr>';
}

function renderSpatialStats(spatial) {
  const items = [
    ["最近邻距离均值", `${formatNumber(spatial.nearest_neighbor_mean, 3)} px`],
    ["最近邻 CV", formatNumber(spatial.nearest_neighbor_cv, 4)],
    ["最近邻指数 NNI", formatNumber(spatial.nearest_neighbor_index, 4)],
    ["网格密度 CV", formatNumber(spatial.grid_density_cv, 4)],
    ["覆盖率 CV", formatNumber(spatial.coverage_density_cv, 4)],
    ["网格熵", formatNumber(spatial.grid_entropy, 4)],
    ["Moran's I", formatNumber(spatial.morans_i, 4)],
    ["网格占用率", formatPercent(spatial.occupancy_ratio, 1)],
    ["热点面积占比", formatPercent(spatial.hotspot_area_ratio, 1)],
    ["密度极差比", formatNumber(spatial.density_range_ratio, 4)],
  ];
  els.spatialStatsList.innerHTML = items.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("");
}

function renderFramework(framework) {
  const uniformity = framework.uniformity || {};
  const bundle = framework.thick_bundle || {};
  const length = framework.long_cnt || {};
  const agglomeration = framework.agglomeration || {};
  els.frameworkUniformity.textContent = formatNumber(uniformity.score, 1);
  els.frameworkUniformityDetail.textContent = `网格 CV ${formatNumber(uniformity.grid_density_cv, 3)} · 等级 ${uniformity.grade || "—"}`;
  els.frameworkBundle.textContent = formatNumber(bundle.score, 1);
  els.frameworkBundleDetail.textContent = `平均宽度 ${formatNumber(bundle.apparent_width_mean_um, 3)} μm · P90 ${formatNumber(bundle.width_p90_um, 3)} μm`;
  els.frameworkLength.textContent = formatNumber(length.score, 1);
  els.frameworkLengthDetail.textContent = `平均骨架 ${formatNumber(length.skeleton_length_mean_um, 2)} μm · 超长占比 ${formatPercent(length.ultra_long_ratio, 1)}`;
  els.frameworkAgglomeration.textContent = formatNumber(agglomeration.score, 1);
  els.frameworkAgglomerationDetail.textContent = `团聚面积 ${formatPercent(agglomeration.agglomerated_area_ratio, 1)} · 最大区域 ${formatNumber(agglomeration.largest_agglomerate_area_um2, 2)} μm²`;
  const weights = framework.score_weights || {};
  els.frameworkFormula.textContent = `A ${formatPercent(weights.uniformity, 0)} + B ${formatPercent(weights.thick_bundle, 0)} + C ${formatPercent(weights.long_cnt, 0)} + D ${formatPercent(weights.agglomeration, 0)}`;
  els.frameworkHybrid.textContent = formatNumber(framework.hybrid_score, 1);
}

function prepareCanvas(canvas, minHeight = 220) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(120, Math.floor(rect.width || canvas.parentElement?.clientWidth || 500));
  const height = Math.max(minHeight, Math.floor(rect.height || minHeight));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function renderResultCharts() {
  if (!state.result || els.resultsDashboard.classList.contains("hidden")) return;
  drawLengthHistogram();
  drawDispersionChart();
  drawMorphologyChart();
  const spatial = state.result.stats?.spatial_distribution || {};
  drawHeatmap(els.pointHeatmap, spatial.point_density_grid || spatial.density_grid || [], "count");
  drawHeatmap(els.coverageHeatmap, spatial.coverage_density_grid || [], "ratio");
  drawHeatmap(els.shadowHeatmap, spatial.shadow_density_grid || [], "ratio");
}

function drawAxes(ctx, width, height, pad) {
  ctx.strokeStyle = "#dfe7e4";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();
}

function drawLengthHistogram() {
  const summaryLengths = state.result?.stats?.lengths;
  const values = Array.isArray(summaryLengths) && summaryLengths.length
    ? summaryLengths.map(Number).filter(Number.isFinite)
    : (state.result?.measurements || []).map((item) => Number(item.length_um)).filter(Number.isFinite);
  const { ctx, width, height } = prepareCanvas(els.lengthHistogram, 240);
  const pad = { left: 38, right: 14, top: 18, bottom: 33 };
  drawAxes(ctx, width, height, pad);
  if (!values.length) return drawEmptyChart(ctx, width, height, "无长度数据");
  const sorted = [...values].sort((a, b) => a - b);
  const cap = percentile(sorted, .97) * 1.08 || Math.max(...sorted, 1);
  const bins = Array(12).fill(0);
  values.forEach((value) => bins[Math.min(bins.length - 1, Math.floor(Math.min(value, cap) / cap * bins.length))]++);
  const maxCount = Math.max(...bins, 1);
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const slot = plotWidth / bins.length;
  bins.forEach((count, index) => {
    const barHeight = count / maxCount * (plotHeight - 5);
    const x = pad.left + index * slot + 2;
    const y = pad.top + plotHeight - barHeight;
    const gradient = ctx.createLinearGradient(0, y, 0, height - pad.bottom);
    gradient.addColorStop(0, "#2f8778");
    gradient.addColorStop(1, "#b9ded5");
    ctx.fillStyle = gradient;
    roundedRect(ctx, x, y, Math.max(2, slot - 4), barHeight, 3);
    ctx.fill();
  });
  ctx.fillStyle = "#7b8c88";
  ctx.font = "8px Segoe UI";
  [0, .25, .5, .75, 1].forEach((fraction) => {
    const x = pad.left + plotWidth * fraction;
    ctx.fillText(`${(cap * fraction).toFixed(cap > 100 ? 0 : 1)}`, x - 7, height - 12);
  });
  ctx.fillText("长度 (μm)", width - 66, height - 12);
}

function drawDispersionChart() {
  const dispersed = Number(state.result?.dispersed_stats?.dispersed_count || 0);
  const agglomerated = Number(state.result?.dispersed_stats?.agglomerated_count || 0);
  const total = Math.max(1, dispersed + agglomerated);
  const { ctx, width, height } = prepareCanvas(els.dispersionChart, 240);
  const cx = width * .42;
  const cy = height * .52;
  const radius = Math.min(width, height) * .29;
  const inner = radius * .63;
  let angle = -Math.PI / 2;
  [[dispersed, "#2f8778"], [agglomerated, "#d75067"]].forEach(([value, color]) => {
    const next = angle + value / total * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, angle, next);
    ctx.arc(cx, cy, inner, next, angle, true);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    angle = next;
  });
  ctx.fillStyle = "#172623";
  ctx.font = "400 24px Georgia";
  ctx.textAlign = "center";
  ctx.fillText(String(dispersed + agglomerated), cx, cy + 2);
  ctx.fillStyle = "#7b8c88";
  ctx.font = "8px Segoe UI";
  ctx.fillText("CNT", cx, cy + 18);
  ctx.textAlign = "left";
  [["分散", dispersed, "#2f8778"], ["团聚", agglomerated, "#d75067"]].forEach(([label, value, color], index) => {
    const y = height * .42 + index * 42;
    ctx.fillStyle = color;
    ctx.fillRect(width * .72, y, 8, 8);
    ctx.fillStyle = "#667773";
    ctx.font = "8px Segoe UI";
    ctx.fillText(`${label} ${value} · ${(value / total * 100).toFixed(1)}%`, width * .72 + 14, y + 8);
  });
}

function drawMorphologyChart() {
  const points = (state.result?.measurements || [])
    .map((item) => ({ x: Number(item.length_um), y: Number(item.width_median_um ?? item.width_mean_um) }))
    .filter((item) => Number.isFinite(item.x) && Number.isFinite(item.y));
  const { ctx, width, height } = prepareCanvas(els.morphologyChart, 240);
  const pad = { left: 39, right: 16, top: 18, bottom: 33 };
  drawAxes(ctx, width, height, pad);
  if (!points.length) return drawEmptyChart(ctx, width, height, "无宽度数据");
  const maxX = Math.max(...points.map((item) => item.x), 1) * 1.06;
  const maxY = Math.max(...points.map((item) => item.y), 1) * 1.12;
  ctx.fillStyle = "rgba(47, 135, 120, .58)";
  points.forEach((point) => {
    const x = pad.left + point.x / maxX * (width - pad.left - pad.right);
    const y = height - pad.bottom - point.y / maxY * (height - pad.top - pad.bottom);
    ctx.beginPath();
    ctx.arc(x, y, 2.6, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = "#7b8c88";
  ctx.font = "8px Segoe UI";
  ctx.fillText("长度 (μm)", width - 64, height - 12);
  ctx.save();
  ctx.translate(12, pad.top + 52);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("宽度 (μm)", 0, 0);
  ctx.restore();
}

function drawHeatmap(canvas, rawGrid, mode = "count") {
  const grid = Array.isArray(rawGrid) ? rawGrid : [];
  const { ctx, width, height } = prepareCanvas(canvas, 270);
  if (!grid.length || !grid[0]?.length) return drawEmptyChart(ctx, width, height, "无网格数据");
  const rows = grid.length;
  const cols = grid[0].length;
  const padding = 18;
  const size = Math.min(width - padding * 2, height - padding * 2);
  const cellWidth = size / cols;
  const cellHeight = size / rows;
  const startX = (width - size) / 2;
  const startY = (height - size) / 2;
  const values = grid.flat().map(Number).filter(Number.isFinite);
  const max = Math.max(...values, 1e-9);
  grid.forEach((row, y) => row.forEach((raw, x) => {
    const value = Number(raw) || 0;
    const t = Math.max(0, Math.min(1, value / max));
    const hue = 165 - t * 142;
    const light = 95 - t * 45;
    ctx.fillStyle = `hsl(${hue} 48% ${light}%)`;
    ctx.fillRect(startX + x * cellWidth, startY + y * cellHeight, cellWidth - 1, cellHeight - 1);
    if (rows <= 10 && cols <= 10) {
      ctx.fillStyle = t > .58 ? "white" : "#39504b";
      ctx.font = "7px Segoe UI";
      ctx.textAlign = "center";
      ctx.fillText(mode === "ratio" ? value.toFixed(2) : value.toFixed(0), startX + (x + .5) * cellWidth, startY + (y + .55) * cellHeight);
    }
  }));
  ctx.textAlign = "left";
}

function drawEmptyChart(ctx, width, height, label) {
  ctx.fillStyle = "#8a9895";
  ctx.font = "9px Segoe UI";
  ctx.textAlign = "center";
  ctx.fillText(label, width / 2, height / 2);
  ctx.textAlign = "left";
}

function percentile(sorted, fraction) {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))];
}

function roundedRect(ctx, x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius, Math.abs(width) / 2, Math.abs(height) / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function updateParameterOutputs() {
  els.blurValue.textContent = els.blurInput.value;
  els.blockValue.textContent = els.blockInput.value;
  els.adaptiveCValue.textContent = els.adaptiveCInput.value;
  els.bridgeValue.textContent = els.bridgeInput.value;
  updateCompareSummary();
}

function updateCompareSummary() {
  const fraction = Math.round(Number(els.compareCenterFraction.value) * 100);
  const profile = { balanced: "标准", precision: "少误检", recall: "少漏检" }[els.compareProfile.value] || "标准";
  els.compareParameterSummary.textContent = `${els.compareScaleUm.value} μm · ${fraction === 100 ? "完整图像" : `中部 ${fraction}%`} · ${profile}检测`;
}

function updateFileGroup(kind, files) {
  const limit = state.config?.max_batch_files_per_group || 20;
  const list = [...files].slice(0, limit);
  state.compare[kind === "base" ? "baseFiles" : "experimentFiles"] = list;
  const count = kind === "base" ? els.baseFileCount : els.experimentFileCount;
  const target = kind === "base" ? els.baseFileList : els.experimentFileList;
  count.textContent = `${list.length} 张`;
  target.innerHTML = list.length
    ? list.map((file) => `<li><span>${escapeHtml(file.name)}</span><b>${formatBytes(file.size)}</b></li>`).join("")
    : "<li>尚未选择图像</li>";
  els.startComparisonButton.disabled = !(state.compare.baseFiles.length && state.compare.experimentFiles.length);
  if (files.length > limit) showToast(`每组最多 ${limit} 张，已保留前 ${limit} 张。`, "info");
}

function getComparisonSettings() {
  return {
    preprocess: {
      blur_kernel: Number(els.blurInput.value),
      adaptive_block: Number(els.blockInput.value),
      adaptive_c: Number(els.adaptiveCInput.value),
      bridge_strength: Number(els.bridgeInput.value),
      threshold_invert: els.thresholdInvertInput.checked,
      generate_skeleton: true,
    },
    detection: {
      min_length_um: Number(els.compareMinLength.value),
      max_length_um: Number(els.maxLengthInput.value || 1000),
      min_slenderness: Number(els.compareMinSlenderness.value),
      detection_profile: els.compareProfile.value,
      split_mode: els.splitModeInput.value,
      merge_distance_px: Number(els.mergeDistanceInput.value || 0),
    },
    scale_um: Number(els.compareScaleUm.value),
    manual_scale_pixels: Number(els.compareScalePixels.value),
    center_fraction: Number(els.compareCenterFraction.value),
    recognize_scale_text: false,
  };
}

async function startComparison() {
  if (!state.compare.baseFiles.length || !state.compare.experimentFiles.length) return;
  window.clearTimeout(state.compare.timer);
  state.compare.result = null;
  els.compareDashboard.classList.add("hidden");
  els.compareProgress.classList.remove("hidden");
  els.startComparisonButton.disabled = true;
  els.compareProgressMessage.textContent = "正在上传图像并建立后台任务…";
  els.compareProgressPercent.textContent = "0%";
  els.compareProgressBar.style.width = "0%";
  try {
    const form = new FormData();
    state.compare.baseFiles.forEach((file) => form.append("base_files", file, file.name));
    state.compare.experimentFiles.forEach((file) => form.append("experiment_files", file, file.name));
    form.append("settings", JSON.stringify(getComparisonSettings()));
    const response = await apiJson("/api/v1/comparisons", { method: "POST", body: form });
    state.compare.jobId = response.job_id;
    pollComparison();
  } catch (error) {
    els.compareProgress.classList.add("hidden");
    els.startComparisonButton.disabled = false;
    showToast(`对比任务创建失败：${error.message}`, "error", 7000);
  }
}

async function pollComparison() {
  if (!state.compare.jobId) return;
  try {
    const snapshot = await apiJson(`/api/v1/comparisons/${state.compare.jobId}`);
    const percent = Math.round((snapshot.progress || 0) * 100);
    els.compareProgressMessage.textContent = snapshot.message || "正在分析";
    els.compareProgressPercent.textContent = `${percent}%`;
    els.compareProgressBar.style.width = `${percent}%`;
    if (snapshot.status === "complete") {
      state.compare.result = snapshot.result;
      els.compareProgress.classList.add("hidden");
      els.startComparisonButton.disabled = false;
      renderComparison(snapshot.result);
      els.compareDashboard.classList.remove("hidden");
      showToast("组间完整分析已完成。", "success");
      return;
    }
    if (snapshot.status === "failed") throw new Error(snapshot.error || "后台任务失败");
    state.compare.timer = window.setTimeout(pollComparison, 1100);
  } catch (error) {
    els.compareProgress.classList.add("hidden");
    els.startComparisonButton.disabled = false;
    showToast(`组间对比失败：${error.message}`, "error", 7000);
  }
}

function renderComparison(result) {
  const base = result.base || {};
  const experiment = result.experiment || {};
  const metrics = result.metrics || [];
  const scoreBase = Number(base.uniformity_score_stats?.mean || 0);
  const scoreExperiment = Number(experiment.uniformity_score_stats?.mean || 0);
  const dispersedDelta = Number(experiment.dispersed_ratio_stats?.mean || 0) - Number(base.dispersed_ratio_stats?.mean || 0);
  const gridImprovement = Number(base.grid_density_cv_stats?.mean || 0) - Number(experiment.grid_density_cv_stats?.mean || 0);
  const scoreDelta = scoreExperiment - scoreBase;
  let title = "两组表现接近，建议结合逐图波动";
  if (scoreDelta > 5 && dispersedDelta > 0 && gridImprovement > 0) title = "实验组整体呈现更好的均匀性";
  else if (scoreDelta < -5) title = "实验组整体均匀性低于 base 组";
  else if (gridImprovement > 0 && dispersedDelta > 0) title = "实验组在分散与空间维度均有改善";
  const detail = `均匀性变化 ${signed(scoreDelta, 1)} 分；分散比例变化 ${signed(dispersedDelta * 100, 1)} 个百分点；网格 CV ${gridImprovement >= 0 ? "改善" : "升高"} ${Math.abs(gridImprovement).toFixed(3)}。`;
  els.compareVerdict.textContent = detail;
  els.compareVerdictTitle.textContent = title;
  els.compareVerdictDetail.textContent = detail;
  els.baseAnalyzedCount.textContent = base.image_count || 0;
  els.experimentAnalyzedCount.textContent = experiment.image_count || 0;
  els.baseUniformityText.textContent = `均匀性 ${formatNumber(scoreBase, 1)} · CNT ${formatNumber(base.count_stats?.mean, 1)}/图`;
  els.experimentUniformityText.textContent = `均匀性 ${formatNumber(scoreExperiment, 1)} · CNT ${formatNumber(experiment.count_stats?.mean, 1)}/图`;
  els.comparisonExportLink.href = `/api/v1/comparisons/${state.compare.jobId}/export.csv`;
  els.baseRepresentative.src = cacheBust(result.representative_urls?.base || "");
  els.experimentRepresentative.src = cacheBust(result.representative_urls?.experiment || "");
  els.comparisonTableBody.innerHTML = metrics.map((metric) => {
    const baseMean = Number(metric.base?.mean || 0);
    const expMean = Number(metric.experiment?.mean || 0);
    const delta = baseMean ? (expMean - baseMean) / Math.abs(baseMean) * 100 : 0;
    const deltaClass = delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : "";
    return `<tr><td>${escapeHtml(metric.label)}</td><td>${formatNumber(baseMean, 4)} ± ${formatNumber(metric.base?.std, 4)}</td><td>${formatNumber(expMean, 4)} ± ${formatNumber(metric.experiment?.std, 4)}</td><td class="${deltaClass}">${signed(delta, 1)}%</td><td>${formatP(metric.t_pvalue)}</td><td>${formatP(metric.mw_pvalue)}</td><td>${escapeHtml(metric.direction)}</td></tr>`;
  }).join("");
  renderComparisonCharts();
}

function signed(value, digits = 1) {
  const numeric = Number(value) || 0;
  return `${numeric > 0 ? "+" : ""}${numeric.toFixed(digits)}`;
}

function formatP(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/A";
  if (numeric < .001) return "<0.001***";
  if (numeric < .01) return `${numeric.toFixed(3)}**`;
  if (numeric < .05) return `${numeric.toFixed(3)}*`;
  return `${numeric.toFixed(3)} n.s.`;
}

function renderComparisonCharts() {
  if (!state.compare.result || els.compareDashboard.classList.contains("hidden")) return;
  const metrics = (state.compare.result.metrics || []).slice(0, 5);
  const { ctx, width, height } = prepareCanvas(els.comparisonChart, 360);
  const pad = { left: Math.min(118, width * .28), right: 20, top: 18, bottom: 18 };
  const plotWidth = width - pad.left - pad.right;
  const rowHeight = (height - pad.top - pad.bottom) / Math.max(metrics.length, 1);
  ctx.font = "8px Segoe UI";
  metrics.forEach((metric, index) => {
    const base = Math.abs(Number(metric.base?.mean || 0));
    const experiment = Math.abs(Number(metric.experiment?.mean || 0));
    const max = Math.max(base, experiment, 1e-9);
    const y = pad.top + index * rowHeight;
    ctx.fillStyle = "#667773";
    ctx.textAlign = "right";
    ctx.fillText(metric.label, pad.left - 10, y + rowHeight * .53);
    ctx.fillStyle = "#edf2ef";
    roundedRect(ctx, pad.left, y + rowHeight * .20, plotWidth, rowHeight * .22, 4);
    ctx.fill();
    roundedRect(ctx, pad.left, y + rowHeight * .52, plotWidth, rowHeight * .22, 4);
    ctx.fill();
    ctx.fillStyle = "#2f8778";
    roundedRect(ctx, pad.left, y + rowHeight * .20, plotWidth * base / max, rowHeight * .22, 4);
    ctx.fill();
    ctx.fillStyle = "#e7873b";
    roundedRect(ctx, pad.left, y + rowHeight * .52, plotWidth * experiment / max, rowHeight * .22, 4);
    ctx.fill();
  });
  ctx.textAlign = "left";
}

function bindEvents() {
  $$('[data-page]').forEach((button) => button.addEventListener("click", () => switchPage(button.dataset.page)));
  els.helpButton.addEventListener("click", () => switchPage("method"));
  els.singleImageInput.addEventListener("change", () => uploadSingleImage(els.singleImageInput.files[0]));
  els.replaceImageButton.addEventListener("click", () => els.singleImageInput.click());
  ["dragenter", "dragover"].forEach((name) => els.singleDropZone.addEventListener(name, (event) => {
    event.preventDefault();
    els.singleDropZone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((name) => els.singleDropZone.addEventListener(name, (event) => {
    event.preventDefault();
    els.singleDropZone.classList.remove("dragging");
  }));
  els.singleDropZone.addEventListener("drop", (event) => uploadSingleImage(event.dataTransfer.files[0]));

  document.addEventListener("paste", async (event) => {
    const cd = event.clipboardData;
    if (!cd) {
      showToast("浏览器未提供剪贴板数据，请重新截图后按 Ctrl+V。", "error", 6000);
      return;
    }
    // 输入框里的文字粘贴不干预
    const target = event.target;
    const isEditable = !!(target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable));

    const pickImage = (blob) => {
      if (!blob) return false;
      if (!blob.size) {
        showToast("剪贴板图片内容为空，请重新截图后粘贴。", "error", 6000);
        return true;
      }
      // 微信/QQ 截图等经浏览器转换后 MIME 可能不标准，放宽判断：image/* 或空类型都按图片处理
      const isImage = blob.type.startsWith("image/") || !blob.type;
      if (!isImage) return false;
      const name = blob.name || `粘贴图像-${Date.now()}.png`;
      const file = blob.name ? blob : new File([blob], name, { type: blob.type || "image/png" });
      uploadSingleImage(file);
      event.preventDefault();
      return true;
    };

    // 1) clipboardData.files（多数浏览器截图粘贴的入口）
    for (const f of Array.from(cd.files || [])) {
      if (pickImage(f)) return;
    }
    // 2) clipboardData.items 中的图片文件（微信截图可能 type 非标准，不预判 MIME）
    for (const item of Array.from(cd.items || [])) {
      if (item.kind === "file" && pickImage(item.getAsFile())) return;
    }
    // 3) navigator.clipboard.read() 兜底（HTTPS 下 Edge/Chrome 可用，首次会请求剪贴板权限）
    if (navigator.clipboard?.read) {
      try {
        const entries = await navigator.clipboard.read();
        for (const entry of entries) {
          for (const type of entry.types) {
            if (type.startsWith("image/") && pickImage(await entry.getType(type))) return;
          }
        }
      } catch (_) {
        // 无权限，落到下方统一提示
      }
    }
    // 走到这里说明收到了粘贴事件但没拿到图片
    if (!isEditable) {
      showToast("未在剪贴板中检测到图片。微信截图请先点「√ 完成」，再回到页面按 Ctrl+V。", "error", 7000);
    }
  });

  $$("[data-view]", els.viewTabs).forEach((button) => button.addEventListener("click", () => loadView(button.dataset.view)));
  $$("[data-drawer]").forEach((button) => button.addEventListener("click", () => switchDrawer(button.dataset.drawer)));
  els.drawScaleButton.addEventListener("click", () => startDrawing("scale"));
  els.drawRoiButton.addEventListener("click", () => startDrawing("roi"));
  els.drawerDrawRoiButton.addEventListener("click", () => startDrawing("roi"));
  els.cancelDrawButton.addEventListener("click", cancelDrawing);
  els.interactionCanvas.addEventListener("pointerdown", onCanvasPointerDown);
  els.interactionCanvas.addEventListener("pointermove", onCanvasPointerMove);
  els.interactionCanvas.addEventListener("pointerup", onCanvasPointerUp);
  els.interactionCanvas.addEventListener("pointercancel", cancelDrawing);
  els.fitImageButton.addEventListener("click", () => fitImageToViewport(true));
  els.zoomOutButton.addEventListener("click", () => setZoom(state.zoom / 1.2));
  els.zoomInButton.addEventListener("click", () => setZoom(state.zoom * 1.2));
  els.applyScaleButton.addEventListener("click", applyScale);
  els.useFullImageButton.addEventListener("click", () => selectRoi(null));
  els.suggestParamsButton.addEventListener("click", () => requestSuggestion(true));
  els.detectionProfile.addEventListener("change", () => requestSuggestion(false));

  [els.blurInput, els.blockInput, els.adaptiveCInput, els.bridgeInput, els.thresholdInvertInput].forEach((input) => input.addEventListener("input", scheduleLivePreview));
  [els.minLengthInput, els.maxLengthInput, els.minSlendernessInput, els.mergeDistanceInput, els.splitModeInput].forEach((input) => input.addEventListener("change", () => {
    state.result = null;
    els.resultsDashboard.classList.add("hidden");
    disableResultViews();
  }));
  els.previewButton.addEventListener("click", () => runPreview({ skeleton: true }));
  els.analyzeButton.addEventListener("click", runFullAnalysis);
  els.scrollToViewerButton.addEventListener("click", () => els.analysisStudio.scrollIntoView({ behavior: "smooth", block: "start" }));
  els.showParticleViewButton.addEventListener("click", async () => {
    await loadView("particles");
    els.analysisStudio.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  els.exportMenuButton.addEventListener("click", () => els.exportMenu.classList.toggle("hidden"));
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".export-menu")) els.exportMenu.classList.add("hidden");
  });
  $$("[data-result-tab]").forEach((button) => button.addEventListener("click", () => {
    const tab = button.dataset.resultTab;
    $$("[data-result-tab]").forEach((item) => item.classList.toggle("active", item.dataset.resultTab === tab));
    $$("[data-result-panel]").forEach((panel) => panel.classList.toggle("active", panel.dataset.resultPanel === tab));
    if (["overview", "spatial"].includes(tab)) window.setTimeout(renderResultCharts, 50);
    if (tab === "measurements") ensureMeasurementsLoaded();
  }));
  els.measurementSearch.addEventListener("input", renderMeasurementTable);
  $$("th[data-sort]").forEach((header) => header.addEventListener("click", () => {
    const key = header.dataset.sort;
    state.measurementSort = state.measurementSort.key === key
      ? { key, direction: state.measurementSort.direction * -1 }
      : { key, direction: 1 };
    renderMeasurementTable();
  }));

  els.baseFilesInput.addEventListener("change", () => updateFileGroup("base", els.baseFilesInput.files));
  els.experimentFilesInput.addEventListener("change", () => updateFileGroup("experiment", els.experimentFilesInput.files));
  [els.compareScaleUm, els.compareScalePixels, els.compareCenterFraction, els.compareProfile, els.compareMinLength, els.compareMinSlenderness].forEach((input) => input.addEventListener("input", updateCompareSummary));
  els.startComparisonButton.addEventListener("click", startComparison);

  let resizeTimer;
  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      if (state.session) fitImageToViewport(false);
      if (state.result) renderResultCharts();
      if (state.compare.result) renderComparisonCharts();
    }, 140);
  });
}

bootstrap();
