// js/app.js — Main UI, state, pipeline, and init for MultiespectralFusion.
// Depends on: app-math.js (window.FusionMath), app-compute.js (window.FusionCompute).
(() => {
  'use strict';

  /* ════════════════════════════════════════════════════════════════
     CONSTANTS
  ════════════════════════════════════════════════════════════════ */

  // Preloaded example pairs (paths relative to the tool root)
  const EXAMPLES = [
    { label: 'KAIST Day', rgb: 'media/KAIST_day_visible_image.png', lwir: 'media/KAIST_day_lwir_image.png' },
    { label: 'LLVIP', rgb: 'media/LLVIP_visible_image.png', lwir: 'media/LLVIP_lwir_image.png' },
  ];

  const METHODS_STATIC = ['rgbt', 'hsvt', 'vths', 'vt'];
  const METHODS_REPR   = ['wavelet', 'waveletmax', 'curvelet', 'curveletmax', 'pca', 'fa'];
  const METHODS_ALPHA  = ['sobel', 'ssim', 'superpixel'];
  const ALL_METHODS    = [...METHODS_STATIC, ...METHODS_REPR, ...METHODS_ALPHA];

  /* ════════════════════════════════════════════════════════════════
     I18N
  ════════════════════════════════════════════════════════════════ */

  const i18nApi = window.FusionI18n;
  let _i18nCopy = null;   // full translations dict for the active language (set in applyTranslations)

  /* ════════════════════════════════════════════════════════════════
     IMPORTS FROM COMPANION MODULES
  ════════════════════════════════════════════════════════════════ */

  const { bitmapToImageData, workingSize, equalizeToOffscreenCanvas } = window.FusionMath;
  const { COMPUTE_FN } = window.FusionCompute;

  /* ════════════════════════════════════════════════════════════════
     STATE
  ════════════════════════════════════════════════════════════════ */

  const state = {
    lang: 'en',
    activeExample: null,
    activeStrategy: 'static-early',
    activeMethod: 'rgbt',
    pipelineResults: {},
    pipelineImageData: {},  // canvasId → full-res ImageData
    generating: false,
    maxWorkSide: 512,   // resolution cap (longer side); user-selectable, 0 = full res
    equalizeRgb:  false,
    equalizeLwir: false,
    rawRgb:       null,
    rawRgbLabel:  '',
    rawLwir:      null,
    rawLwirLabel: '',
    histCharts:   { rgb: null, lwir: null, modal: null },
    histLog:      true,    // histogram Y axis: false=linear, true=logarithmic (shared by thumbnails + modal).
                           // Default log: image histograms have a huge dynamic range (a spike at one
                           // intensity dwarfs the rest) so linear collapses most channels to a flat line.
    pipelineConnectionData: {},  // method → { cardEls, opNodeEls, connections }
    isCustomMode:     false,
    customRgbBitmap:  null,
    customRgbLabel:   '',
    customLwirBitmap: null,
    customLwirLabel:  '',
    // Counter-based example load guard (async-safe).
    // Set to 2 before a loadExample call; each card's onLoad decrements it.
    // While > 0, onLoad is treated as a programmatic example load, not user-initiated.
    _exampleLoadsExpected: 0,
  };

  ALL_METHODS.forEach((m) => { state.pipelineResults[m] = null; });

  /* ════════════════════════════════════════════════════════════════
     DOM CACHE
  ════════════════════════════════════════════════════════════════ */

  const dom = {
    subtitle:         document.getElementById('subtitle'),
    introTitle:       document.getElementById('intro-title'),
    introText:        document.getElementById('intro-text'),
    cfgTitle:         document.getElementById('cfg-title'),

    selectExample:    document.getElementById('select-example'),
    lblExamples:      document.getElementById('lbl-examples'),
    selectResolution: document.getElementById('select-resolution'),
    badgeRgb:         document.getElementById('badge-rgb'),
    badgeLwir:        document.getElementById('badge-lwir'),
    srcRgbTitle:      document.getElementById('src-rgb-title'),
    srcLwirTitle:     document.getElementById('src-lwir-title'),
    histRgb:          document.getElementById('hist-rgb'),
    histLwir:         document.getElementById('hist-lwir'),
    btnExpandHistRgb: document.getElementById('btn-expand-hist-rgb'),
    btnExpandHistLwir:document.getElementById('btn-expand-hist-lwir'),
    histLogScale:     document.getElementById('hist-log-scale'),
    lblHistLog:       document.getElementById('lbl-hist-log'),
    chkEqualizeRgb:   document.getElementById('chk-equalize-rgb'),
    chkEqualizeLwir:  document.getElementById('chk-equalize-lwir'),
    lblEqualizeRgb:   document.getElementById('lbl-equalize-rgb'),
    lblEqualizeLwir:  document.getElementById('lbl-equalize-lwir'),

    btnGenerate:      document.getElementById('btn-generate'),
    btnDownload:      document.getElementById('btn-download-active'),
    status:           document.getElementById('status'),

    strategyHint:     document.getElementById('strategy-hint'),

    pipelinePanels:   Object.fromEntries(
      ALL_METHODS.map((m) => [m, document.getElementById(`pipeline-${m}`)])
    ),

    modalOverlay:     document.getElementById('image-modal-overlay'),
    modalCanvas:      document.getElementById('image-modal-canvas'),
    modalTitle:       document.getElementById('image-modal-title'),
    btnCloseModal:    document.getElementById('btn-close-image-modal'),
    btnDownloadModal: document.getElementById('btn-download-image-modal'),

    histModalOverlay:  document.getElementById('hist-modal-overlay'),
    histModalCanvas:   document.getElementById('hist-modal-canvas'),
    histModalTitle:    document.getElementById('hist-modal-title'),
    btnCloseHistModal: document.getElementById('btn-close-hist-modal'),

    footerAuthor:     document.getElementById('footer-author'),
  };

  /* ════════════════════════════════════════════════════════════════
     HISTOGRAM  (Chart.js)
  ════════════════════════════════════════════════════════════════ */

  /**
   * Compute per-channel pixel histograms from a bitmap.
   * Returns { hR, hG, hB } Uint32Array[256] each.
   */
  function computePixelHistogram(bitmap) {
    if (!bitmap) return null;
    const oc = new OffscreenCanvas(bitmap.width, bitmap.height);
    oc.getContext('2d').drawImage(bitmap, 0, 0);
    const { data, width, height } = oc.getContext('2d').getImageData(0, 0, bitmap.width, bitmap.height);
    const n = width * height;
    const hR = new Uint32Array(256), hG = new Uint32Array(256), hB = new Uint32Array(256);
    for (let i = 0; i < n; i++) { hR[data[i*4]]++; hG[data[i*4+1]]++; hB[data[i*4+2]]++; }
    return { hR, hG, hB };
  }

  function histPointData(hist, log) {
    // In log mode zero counts must become gaps (null) — a logarithmic axis can't
    // plot 0 and would otherwise break the line / fill.
    return Array.from(hist, (y, x) => ({ x, y: (log && y <= 0) ? null : y }));
  }

  /** Read a CSS custom property value from the document root. */
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || name;
  }

  function createHistogramChart(canvas, type, { minimized = true, log = false } = {}) {
    const yType = log ? 'logarithmic' : 'linear';
    const yMin  = log ? 1 : 0;
    if (!canvas || !window.Chart || !window.SharedChartLegend) return null;
    const L = window.SharedChartLegend;
    const theme = L.getChartTheme();
    const empty = () => Array.from({ length: 256 }, (_, x) => ({ x, y: 0 }));

    const clrR = cssVar('--clr-ch-r');
    const clrG = cssVar('--clr-ch-g');
    const clrB = cssVar('--clr-ch-b');
    const clrT = cssVar('--clr-ch-t');

    const FILL_BASE = 0.12;
    const baseDs = { parsing: false, fill: true, pointRadius: 0, tension: 0.1, borderWidth: 1.5 };

    const datasets = type === 'rgb'
      ? [
          { ...baseDs, label: 'R Channel', data: empty(), borderColor: clrR, backgroundColor: L.withAlpha(clrR, FILL_BASE) },
          { ...baseDs, label: 'G Channel', data: empty(), borderColor: clrG, backgroundColor: L.withAlpha(clrG, FILL_BASE) },
          { ...baseDs, label: 'B Channel', data: empty(), borderColor: clrB, backgroundColor: L.withAlpha(clrB, FILL_BASE) },
        ]
      : [
          { ...baseDs, label: 'Thermal Channel', data: empty(), borderColor: clrT, backgroundColor: L.withAlpha(clrT, 0.22) },
        ];

    // Start from the shared legend (theme-aware labels) but REPLACE its hover
    // handlers with a much stronger highlight: the hovered channel goes fully
    // solid + thicker line + brought to front, the others fade right back.
    // (We OVERRIDE, not stack onto, the shared onHover/onLeave — stacking caused a
    // double-mutation that captured the dimmed colour as "original" and left
    // channels stuck. _baseBorder is captured once, before any mutation.)
    const legendWithHighlight = L.createLegendOptions({ position: 'top' });
    legendWithHighlight.onHover = (_e, item, legend) => {
      const ci = legend.chart;
      ci.data.datasets.forEach((ds) => { if (ds._baseBorder == null) ds._baseBorder = ds.borderColor; });
      ci.data.datasets.forEach((ds, i) => {
        const active = i === item.datasetIndex;
        ds.borderColor     = active ? ds._baseBorder : L.withAlpha(ds._baseBorder, 0.12);
        ds.backgroundColor = L.withAlpha(ds._baseBorder, active ? 0.45 : 0.03);
        ds.borderWidth     = active ? 2.6 : 1;
        // NOTE: do NOT touch ds.order — Chart.js sorts the LEGEND items by order,
        // so raising the hovered dataset shoved its label to the end (the legend
        // "jumped to the right" on hover). The solid-vs-faded colours already make
        // the active channel dominant, so no z-reordering is needed.
      });
      ci.update('none');
    };
    legendWithHighlight.onLeave = (_e, _item, legend) => {
      const ci = legend.chart;
      ci.data.datasets.forEach((ds) => {
        if (ds._baseBorder == null) ds._baseBorder = ds.borderColor;
        ds.borderColor     = ds._baseBorder;
        ds.backgroundColor = L.withAlpha(ds._baseBorder, FILL_BASE);
        ds.borderWidth     = 1.5;
      });
      ci.update('none');
    };

    const chart = new window.Chart(canvas, {
      type: 'line',
      data: { datasets },
      options: L.buildChartOptions({
        maintainAspectRatio: false,
        parsing: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: minimized ? { display: false } : legendWithHighlight,
          tooltip: L.createTooltipOptions({
            mode: 'index',
            intersect: false,
            callbacks: {
              title: (items) => items[0] ? `Intensity: ${items[0].parsed.x}` : '',
              label: (item) => `${item.dataset.label}: ${item.parsed.y.toLocaleString()}`,
            },
          }),
        },
        scales: minimized
          ? { x: { display: false, min: 0, max: 255 }, y: { display: false, type: yType, min: yMin } }
          : {
              x: L.buildLinearScale('Intensity', 0, 255, { ticks: { maxTicksLimit: 9, color: theme.text } }),
              y: Object.assign(L.buildLinearScale('Count', yMin, undefined, { ticks: { maxTicksLimit: 5, color: theme.text } }), { type: yType }),
            },
      }),
    });

    // Which state slot holds the live chart for this canvas (thumbnails are
    // recreated on every update, so listeners must read the slot, not a stale
    // closure over `chart`).
    const slot = minimized ? type : 'modal';

    if (minimized) {
      canvas.style.cursor = 'zoom-in';
    } else {
      window.SharedChartInteractions?.attach({
        canvas,
        getChart: () => state.histCharts.modal,
        defaults: { xMin: 0, xMax: 255, mode: 'x' },
      });
    }

    // Bind canvas listeners exactly once per canvas (the chart instance behind a
    // canvas is replaced on update; the listeners always resolve the current one).
    if (!canvas._histBound) {
      canvas._histBound = true;
      if (minimized) {
        canvas.addEventListener('click', () => openHistogramModal(type));
      }
      canvas.addEventListener('mouseleave', () => {
        const c = state.histCharts[slot];
        if (!c) return;
        if (c.tooltip) c.tooltip.setActiveElements([], { x: 0, y: 0 });
        c.setActiveElements([]);
        c.update('none');
      });
    }

    return chart;
  }

  /**
   * Draw the per-channel histogram directly on the small thumbnail canvas.
   * Chart.js mis-renders the line on this tiny canvas (the line drew flat at the
   * top regardless of the — verified correct — scale mapping). A direct 2D draw
   * sidesteps that entirely and is all a non-interactive preview needs. The expand
   * modal keeps Chart.js (which renders fine at full size).
   */
  function drawThumbnailHistogram(canvas, channels, colors, log) {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 200;
    const cssH = canvas.clientHeight || 100;
    canvas.width = Math.max(1, Math.round(cssW * dpr));
    canvas.height = Math.max(1, Math.round(cssH * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    let peak = 0;
    for (const ch of channels) for (let i = 0; i < ch.length; i++) if (ch[i] > peak) peak = ch[i];
    if (peak <= 0) return;

    const pad = 2, W = cssW, H = cssH, n = channels[0].length;
    const lmax = Math.log10(peak + 1);
    const yOf = (v) => log
      ? H - pad - (Math.log10(v + 1) / lmax) * (H - 2 * pad)
      : H - pad - (v / peak) * (H - 2 * pad);
    const xOf = (i) => pad + (i / (n - 1)) * (W - 2 * pad);
    const withA = window.SharedChartLegend ? window.SharedChartLegend.withAlpha : (c) => c;

    channels.forEach((ch, ci) => {
      const col = colors[ci];
      ctx.beginPath();
      ctx.moveTo(xOf(0), H - pad);
      for (let i = 0; i < n; i++) ctx.lineTo(xOf(i), yOf(ch[i]));
      ctx.lineTo(xOf(n - 1), H - pad);
      ctx.closePath();
      ctx.fillStyle = withA(col, 0.12);
      ctx.fill();
      ctx.beginPath();
      for (let i = 0; i < n; i++) { const x = xOf(i), y = yOf(ch[i]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    });
  }

  function updateHistogramChart(type, bitmap) {
    const canvas = type === 'rgb' ? dom.histRgb : dom.histLwir;
    if (!canvas) return;

    // Thumbnails are hand-drawn (see drawThumbnailHistogram). Tear down any leftover
    // Chart.js instance bound to this canvas from an earlier build.
    const existing = window.Chart && window.Chart.getChart ? window.Chart.getChart(canvas) : null;
    if (existing) existing.destroy();
    state.histCharts[type] = null;

    if (!bitmap) { const c = canvas.getContext('2d'); c.clearRect(0, 0, canvas.width, canvas.height); return; }

    const hist = computePixelHistogram(bitmap);
    if (!hist) return;

    const channels = type === 'rgb' ? [hist.hR, hist.hG, hist.hB] : [hist.hR];
    const colors = type === 'rgb'
      ? [cssVar('--clr-ch-r'), cssVar('--clr-ch-g'), cssVar('--clr-ch-b')]
      : [cssVar('--clr-ch-t')];
    drawThumbnailHistogram(canvas, channels, colors, state.histLog);
  }

  /**
   * Configure a histogram chart's Y axis for the active scale mode.
   * Linear: cap at a high percentile of non-zero bin counts so one dominant bin
   *   (e.g. B spiking at intensity 0) doesn't squash the other channels into a
   *   near-flat line.
   * Log: full range [1, peak] so both the spike and the long tail are legible —
   *   the natural fix for image histograms with a huge dynamic range.
   * `channels` is an array of per-series count arrays.
   */
  function applyHistYScale(chart, channels, minimized) {
    const y = chart.options.scales?.y;
    if (!y) return;
    let peak = 0;
    for (const ch of channels) for (let i = 0; i < ch.length; i++) if (ch[i] > peak) peak = ch[i];
    // y.type is fixed at chart creation (switching it at runtime renders broken on
    // the small thumbnail canvas); here we only set the range.
    if (state.histLog) {
      y.min = 1;
      y.max = peak > 1 ? peak : undefined;
    } else {
      y.min = 0;
      if (minimized) {
        // Tiny thumbnail: cap at a high percentile so one dominant bin doesn't
        // squash the rest into a flat line.
        const counts = [];
        for (const ch of channels) for (let i = 0; i < ch.length; i++) if (ch[i] > 0) counts.push(ch[i]);
        counts.sort((a, b) => a - b);
        y.max = counts.length
          ? Math.max(1, Math.ceil(counts[Math.min(counts.length - 1, Math.floor(counts.length * 0.98))] * 1.1))
          : undefined;
      } else {
        // Expand modal: full range, the dominant peak is fine to show at full height.
        y.max = peak > 0 ? Math.ceil(peak * 1.02) : undefined;
      }
    }
  }

  function clearHistModalChart() {
    if (state.histCharts.modal) {
      window.SharedChartInteractions?.detach(dom.histModalCanvas);
      state.histCharts.modal.destroy();
      state.histCharts.modal = null;
    }
  }

  function openHistogramModal(type) {
    const raw = type === 'rgb' ? state.rawRgb : state.rawLwir;
    if (!raw || !dom.histModalCanvas) return;
    const eq = type === 'rgb' ? state.equalizeRgb : state.equalizeLwir;
    const bitmap = eq ? equalizeToOffscreenCanvas(raw, type) : raw;

    clearHistModalChart();

    const chart = createHistogramChart(dom.histModalCanvas, type, { minimized: false, log: state.histLog });
    state.histCharts.modal = chart;

    if (chart) {
      const hist = computePixelHistogram(bitmap);
      if (hist) {
        const channels = type === 'rgb' ? [hist.hR, hist.hG, hist.hB] : [hist.hR];
        channels.forEach((ch, i) => { chart.data.datasets[i].data = histPointData(ch, state.histLog); });
        applyHistYScale(chart, channels, false);
        chart.update('none');
      }
    }
    state.histModalType = type;

    if (dom.histModalTitle) {
      dom.histModalTitle.textContent = type === 'rgb' ? 'RGB Histogram' : 'LWIR Histogram';
    }
    dom.histModalOverlay.classList.remove('hidden');
    dom.histModalOverlay.setAttribute('aria-hidden', 'false');
    requestAnimationFrame(() => state.histCharts.modal?.resize());
  }

  /* ════════════════════════════════════════════════════════════════
     EQUALIZATION MANAGEMENT
  ════════════════════════════════════════════════════════════════ */

  function updateCardEqualization(type) {
    const raw   = type === 'rgb' ? state.rawRgb   : state.rawLwir;
    const label = type === 'rgb' ? state.rawRgbLabel : state.rawLwirLabel;
    const card  = type === 'rgb' ? cardRgb  : cardLwir;
    if (!raw) return;
    const eq = type === 'rgb' ? state.equalizeRgb : state.equalizeLwir;
    const display = eq ? equalizeToOffscreenCanvas(raw, type) : raw;
    card.setImage(display, label);
    updateHistogramChart(type, display);
  }

  function setEqualization(type, enabled) {
    if (type === 'rgb') {
      state.equalizeRgb = enabled;
      if (dom.chkEqualizeRgb) dom.chkEqualizeRgb.checked = enabled;
    } else {
      state.equalizeLwir = enabled;
      if (dom.chkEqualizeLwir) dom.chkEqualizeLwir.checked = enabled;
    }
    updateCardEqualization(type);
    if (cardRgb?.hasImage() && cardLwir?.hasImage()) Promise.resolve().then(runFusion);
  }

  /* ════════════════════════════════════════════════════════════════
     MODE MANAGEMENT (example vs custom)
  ════════════════════════════════════════════════════════════════ */

  function applyMode(isCustom) {
    state.isCustomMode = isCustom;
    document.getElementById('card-rgb') ?.classList.toggle('mode-example', !isCustom);
    document.getElementById('card-lwir')?.classList.toggle('mode-example', !isCustom);
  }

  /* ════════════════════════════════════════════════════════════════
     IMAGE CARDS (shared widget wiring)
  ════════════════════════════════════════════════════════════════ */

  let cardRgb, cardLwir;

  function onImageChange() {
    ALL_METHODS.forEach((m) => { state.pipelineResults[m] = null; });
    state.pipelineConnectionData = {};
    state.pipelineImageData = {};
    dom.btnDownload.disabled = true;
    updateStatus();
    if (cardRgb.hasImage() && cardLwir.hasImage()) {
      Promise.resolve().then(runFusion);
    }
  }

  function initImageCards() {
    cardRgb = window.SharedImageCard.create({
      canvasId:    'preview-rgb',
      fileInputId: 'file-rgb',
      clearBtn:    document.querySelector('[data-clear-slot="rgb"]'),
      expandBtn:   document.querySelector('[data-expand-slot="rgb"]'),
      dropHintId:  'drop-hint-rgb',
      onLoad: (bitmap, fileName) => {
        state.rawRgb = bitmap; state.rawRgbLabel = fileName;
        if (state._exampleLoadsExpected > 0) {
          state._exampleLoadsExpected--;
        } else {
          state.customRgbBitmap = bitmap;
          state.customRgbLabel  = fileName;
          state.activeExample = null;
          syncExampleSelect();
        }
        updateCardEqualization('rgb');
        onImageChange();
      },
      onClear: () => {
        state.rawRgb = null; state.rawRgbLabel = '';
        if (state.isCustomMode) { state.customRgbBitmap = null; state.customRgbLabel = ''; }
        state.activeExample = null; syncExampleSelect();
        updateHistogramChart('rgb', null);
        onImageChange();
      },
      onExpand: (bmp) => openModal(bmp),
    });

    cardLwir = window.SharedImageCard.create({
      canvasId:    'preview-lwir',
      fileInputId: 'file-lwir',
      clearBtn:    document.querySelector('[data-clear-slot="lwir"]'),
      expandBtn:   document.querySelector('[data-expand-slot="lwir"]'),
      dropHintId:  'drop-hint-lwir',
      onLoad: (bitmap, fileName) => {
        state.rawLwir = bitmap; state.rawLwirLabel = fileName;
        if (state._exampleLoadsExpected > 0) {
          state._exampleLoadsExpected--;
        } else {
          state.customLwirBitmap = bitmap;
          state.customLwirLabel  = fileName;
          state.activeExample = null;
          syncExampleSelect();
        }
        updateCardEqualization('lwir');
        onImageChange();
      },
      onClear: () => {
        state.rawLwir = null; state.rawLwirLabel = '';
        if (state.isCustomMode) { state.customLwirBitmap = null; state.customLwirLabel = ''; }
        state.activeExample = null; syncExampleSelect();
        updateHistogramChart('lwir', null);
        onImageChange();
      },
      onExpand: (bmp) => openModal(bmp),
    });

    // In example mode: intercept preview-wrap clicks to open expand modal.
    [
      { wrap: document.querySelector('#card-rgb .preview-wrap'),  getRaw: () => state.rawRgb,  type: 'rgb'  },
      { wrap: document.querySelector('#card-lwir .preview-wrap'), getRaw: () => state.rawLwir, type: 'lwir' },
    ].forEach(({ wrap, getRaw, type }) => {
      if (!wrap) return;
      wrap.addEventListener('click', (e) => {
        if (state.isCustomMode) return;
        if (e.target.closest('.preview-actions')) return;
        const raw = getRaw();
        if (!raw) return;
        e.stopPropagation();
        const eq = type === 'rgb' ? state.equalizeRgb : state.equalizeLwir;
        openModal(eq ? equalizeToOffscreenCanvas(raw, type) : raw);
      }, true);
    });
  }

  /* ════════════════════════════════════════════════════════════════
     EXAMPLE SELECTOR
  ════════════════════════════════════════════════════════════════ */

  function syncExampleSelect() {
    if (!dom.selectExample) return;
    dom.selectExample.value = state.activeExample === null ? 'custom' : String(state.activeExample);
  }

  function loadExample(idx) {
    const ex = EXAMPLES[idx];
    if (!ex) return;
    setStatus(i18nApi.t('statusLoading'));
    state.activeExample = idx;
    syncExampleSelect();
    applyMode(false);

    state._exampleLoadsExpected = 2;
    cardRgb.loadUrl(ex.rgb, `${ex.label}_rgb`);
    cardLwir.loadUrl(ex.lwir, `${ex.label}_lwir`);

    setTimeout(() => {
      if (!cardRgb.hasImage() || !cardLwir.hasImage()) {
        setStatus(i18nApi.t('statusLoadError'));
      }
    }, 4000);
  }

  /* ════════════════════════════════════════════════════════════════
     INPUT SECTION CONNECTION LINES
  ════════════════════════════════════════════════════════════════ */

  function drawInputConnections() {
    const container = document.querySelector('.fusion-diagram-wrapper');
    if (!container) return;
    container.querySelectorAll('.input-conn-svg').forEach((e) => e.remove());

    const cardRgbEl  = document.getElementById('card-rgb');
    const cardLwirEl = document.getElementById('card-lwir');
    const histRgbEl  = document.getElementById('hist-rgb-card');
    const histLwirEl = document.getElementById('hist-lwir-card');
    if (!cardRgbEl || !cardLwirEl || !histRgbEl || !histLwirEl) return;

    const cRect = container.getBoundingClientRect();
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'input-conn-svg');
    svg.setAttribute('aria-hidden', 'true');

    [
      { from: cardRgbEl,  to: histRgbEl,  color: 'var(--clr-visible)',  fromSide: 'left'  },
      { from: cardLwirEl, to: histLwirEl, color: 'var(--clr-thermal)',  fromSide: 'right' },
    ].forEach(({ from, to, color, fromSide }) => {
      const fRect = from.getBoundingClientRect();
      const tRect = to.getBoundingClientRect();

      const x1 = (fromSide === 'left' ? fRect.left : fRect.right) - cRect.left;
      const y1 = fRect.top + fRect.height / 2 - cRect.top;
      const x2 = tRect.left + tRect.width / 2 - cRect.left;
      const y2 = tRect.top - cRect.top;

      const dx = x2 - x1;
      const dy = y2 - y1;
      const r  = Math.min(10, Math.abs(dx) / 2, Math.abs(dy) / 2);
      const sx = dx >= 0 ? 1 : -1;
      let pathD;
      if (Math.abs(dx) < 2) {
        pathD = `M ${x1} ${y1} L ${x2} ${y2}`;
      } else {
        pathD = [
          `M ${x1} ${y1}`,
          `L ${x1 + sx * (Math.abs(dx) - r)} ${y1}`,
          `Q ${x2} ${y1} ${x2} ${y1 + r}`,
          `L ${x2} ${y2}`,
        ].join(' ');
      }

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', pathD);
      path.setAttribute('stroke', color);
      path.setAttribute('stroke-width', '2');
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke-linecap', 'round');
      svg.appendChild(path);

      const aw = 5, ah = 7;
      const arr = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      arr.setAttribute('points', `${x2} ${y2} ${x2 - aw} ${y2 - ah} ${x2 + aw} ${y2 - ah}`);
      arr.setAttribute('fill', color);
      svg.appendChild(arr);
    });

    container.appendChild(svg);
  }

  /* ════════════════════════════════════════════════════════════════
     PIPELINE DIAGRAM BUILDER
  ════════════════════════════════════════════════════════════════ */

  function buildPipelineDiagram(method, stages, connections) {
    const container = dom.pipelinePanels[method];
    container.innerHTML = '';

    const cardEls   = {};
    const opNodeEls = {};
    const stageEls  = {};

    stages.forEach((stage) => {
      if (stage.opNode) {
        const opRowEl = document.createElement('div');
        opRowEl.className = 'pipeline-op-row';
        const opEl = makePipeOpNode(`ps-${method}-${stage.opNode.id}`, stage.opNode.labelKey);
        opRowEl.appendChild(opEl);
        container.appendChild(opRowEl);
        opNodeEls[stage.opNode.id] = opEl;
      }

      const stageEl = makePipelineStage(stage.key);

      stage.cards.forEach((card) => {
        if (card.separateBefore) {
          const spacer = document.createElement('div');
          spacer.className = 'pipe-stage-spacer';
          stageEl.appendChild(spacer);
        }

        const canvasId = `ps-${method}-${card.id}`;
        const cardEl = makePipeCard(card.cardClass, card.desc, canvasId,
          card.isOutput ? 'pipe-card-canvas--xl' : 'pipe-card-canvas--sm',
          card.isOutput, method, card.imageData);
        stageEl.appendChild(cardEl);
        cardEls[card.id] = cardEl;
      });

      container.appendChild(stageEl);
      stageEls[stage.key] = stageEl;
    });

    if (connections?.length) {
      const cardStageIndex   = {};
      const opNodeStageIndex = {};
      stages.forEach((stage, idx) => {
        stage.cards.forEach((c) => { cardStageIndex[c.id] = idx; });
        if (stage.opNode) opNodeStageIndex[stage.opNode.id] = idx;
      });
      state.pipelineConnectionData[method] = {
        cardEls, opNodeEls, connections, stageEls, stages,
        cardStageIndex, opNodeStageIndex,
      };
      requestAnimationFrame(redrawAllConnections);
    }
  }

  function alignPipelineStages() {
    const method = state.activeMethod;
    const data   = state.pipelineConnectionData[method];
    if (!data?.stageEls) return;

    const { cardEls, opNodeEls, connections, stageEls, stages } = data;

    const srcMap = {
      'src-rgb':  document.getElementById('card-rgb'),
      'src-lwir': document.getElementById('card-lwir'),
    };
    function resolveEl(id) {
      return id.startsWith('src-') ? srcMap[id]
           : id.startsWith('op-')  ? opNodeEls[id]
           : cardEls[id];
    }
    function midX(el) {
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return r.width ? r.left + r.width / 2 : null;
    }
    function avg(xs) { return xs.reduce((a, b) => a + b, 0) / xs.length; }

    // Reset all transforms (stages + op-node rows)
    Object.values(stageEls).forEach((el) => { el.style.transform = ''; });
    Object.values(opNodeEls).forEach((el) => {
      if (el.parentElement) el.parentElement.style.transform = '';
    });

    // ── Align card stages (non-opNode stages) ──
    const offsets = {};
    stages.forEach((stage) => {
      if (stage.opNode) return;
      const stageEl = stageEls[stage.key];
      if (!stageEl) return;

      const cardIdSet = new Set(stage.cards.map((c) => c.id));
      const upXs = [];
      connections.forEach((conn) => {
        if (cardIdSet.has(conn.to)) {
          const x = midX(resolveEl(conn.from));
          if (x !== null) upXs.push(x);
        }
      });

      const downXs = [];
      connections.forEach((conn) => {
        if (cardIdSet.has(conn.from)) {
          const x = midX(resolveEl(conn.to));
          if (x !== null) downXs.push(x);
        }
      });

      if (!upXs.length && !downXs.length) return;

      const targetX = (upXs.length && downXs.length)
        ? (avg(upXs) + avg(downXs)) / 2
        : upXs.length ? avg(upXs) : avg(downXs);

      const sRect = stageEl.getBoundingClientRect();
      if (!sRect.width) return;
      const currentX = sRect.left + sRect.width / 2;
      offsets[stage.key] = targetX - currentX;
    });

    Object.entries(offsets).forEach(([key, dx]) => {
      if (Math.abs(dx) > 0.5 && stageEls[key]) {
        stageEls[key].style.transform = `translateX(${dx.toFixed(1)}px)`;
      }
    });

    // ── Align op-node rows based on their upstream/downstream connections ──
    stages.forEach((stage) => {
      if (!stage.opNode) return;
      const opId = stage.opNode.id;
      const opEl = opNodeEls[opId];
      if (!opEl) return;

      const upXs = [];
      connections.forEach((conn) => {
        if (conn.to === opId) {
          const x = midX(resolveEl(conn.from));
          if (x !== null) upXs.push(x);
        }
      });
      const downXs = [];
      connections.forEach((conn) => {
        if (conn.from === opId) {
          const x = midX(resolveEl(conn.to));
          if (x !== null) downXs.push(x);
        }
      });

      if (!upXs.length && !downXs.length) return;

      // A SINGLE downstream (e.g. a merge / inverse-transform node feeding only
      // the Output card) → sit directly above it so the connector is a straight
      // vertical (the upstream cards fan in). Averaging up+down pulled the node
      // off-centre from the Output when the upstream cards were lopsided (e.g. FA
      // factors), bending the merge→output link. Multi-downstream nodes keep the
      // balanced average.
      const targetX = downXs.length === 1 ? downXs[0]
        : (upXs.length && downXs.length) ? (avg(upXs) + avg(downXs)) / 2
        : upXs.length ? avg(upXs) : avg(downXs);

      const opRect = opEl.getBoundingClientRect();
      if (!opRect.width) return;
      const currentX = opRect.left + opRect.width / 2;
      const dx = targetX - currentX;
      if (Math.abs(dx) > 0.5 && opEl.parentElement) {
        opEl.parentElement.style.transform = `translateX(${dx.toFixed(1)}px)`;
      }
    });
  }

  function redrawAllConnections() {
    const wrapper = document.querySelector('.fusion-diagram-wrapper');
    if (!wrapper) return;
    wrapper.querySelectorAll('.pipeline-flow-svg').forEach((e) => e.remove());

    const method = state.activeMethod;
    const data   = state.pipelineConnectionData[method];
    if (!data) return;

    const pipelineEl = dom.pipelinePanels[method];
    if (!pipelineEl || pipelineEl.classList.contains('hidden')) return;

    /* ── Tuning constants ─────────────────────────────────────── */
    const CORNER_R    = 8;   // rounded-corner radius for path bends
    const CHANNEL_GAP = 8;   // min vertical separation between channels
    const OBS_PAD     = 4;   // clearance around obstacle bounding boxes
    const MAX_SPREAD  = 30;  // max fan-out spread at a shared port (px)
    const PORT_STEP   = 13;  // per-port step when spreading a fan across a shared node
    const GUTTER      = 14;  // offset of a side lane from the cards it bypasses
    const LANE_STEP   = 12;  // spacing between parallel side lanes

    /* ── Adaptive vertical spacing ─────────────────────────────
       Count how many connections cross each inter-element gap.
       Adjust the margin-bottom of DOM children so dense corridors
       get more room and sparse ones stay compact.
    ──────────────────────────────────────────────────────────── */
    const pipeChildren = Array.from(pipelineEl.children);
    pipeChildren.forEach((ch) => { ch.style.marginBottom = ''; });
    pipelineEl.style.gap = '';   // reset adaptive gap override

    // Run alignment with default spacing first
    alignPipelineStages();

    // Count edges crossing each gap between consecutive pipeline children
    const srcMap0 = {
      'src-rgb':  document.getElementById('card-rgb'),
      'src-lwir': document.getElementById('card-lwir'),
    };
    function resolveEl0(id) {
      return id.startsWith('src-') ? srcMap0[id]
           : id.startsWith('op-')  ? data.opNodeEls[id]
           : data.cardEls[id];
    }

    if (pipeChildren.length > 1 && data.connections.length > 0) {
      // Collect bottom Y of each child
      const childBottoms = pipeChildren.map((ch) => ch.getBoundingClientRect().bottom);
      const childTops    = pipeChildren.map((ch) => ch.getBoundingClientRect().top);
      const gapCounts    = new Array(pipeChildren.length - 1).fill(0);

      // Which pipeline child (row) contains an element's centre.
      function childIndexOf(el) {
        if (!el) return -1;
        const r = el.getBoundingClientRect();
        const cy = (r.top + r.bottom) / 2;
        for (let i = 0; i < pipeChildren.length; i++) {
          if (cy >= childTops[i] - 4 && cy <= childBottoms[i] + 4) return i;
        }
        return -1;
      }
      // Count, per gap, only the connections that actually TURN there — i.e. that
      // have a horizontal run in that corridor. An edge turns leaving its source
      // child (gap = srcChild) and entering its target child (gap = tgtChild-1);
      // the gaps strictly between are a vertical pass-through (a side-lane drop)
      // that does NOT add a horizontal channel. Counting spans (the old way)
      // over-inflated tall diagrams whose long R/G/B lanes pass through every
      // corridor, leaving big empty vertical gaps between blocks.
      data.connections.filter((c) => !c.bundle).forEach((conn) => {
        const sc = childIndexOf(resolveEl0(conn.from));
        const tc = childIndexOf(resolveEl0(conn.to));
        if (sc < 0 || tc < 0) return;
        const lo = Math.min(sc, tc), hi = Math.max(sc, tc);
        if (lo < gapCounts.length) gapCounts[lo]++;          // turn leaving source
        const tgtGap = hi - 1;
        if (tgtGap !== lo && tgtGap >= 0 && tgtGap < gapCounts.length) gapCounts[tgtGap]++; // turn entering target
      });

      // Apply adaptive margins: a corridor must be tall enough for ALL the
      // horizontal runs that stack inside it (Phase 5) to sit at least
      // CHANNEL_GAP apart, plus stub clearance at each band for the rounded
      // corner turns. Without this geometry-aware floor, dense fan-in/out
      // corridors (e.g. T + wT fanning into R/G/B) collapsed every run onto
      // ~2px spacing — visually "pegadas". n ≈ runs that turn in this gap.
      const BASE_GAP_REM = 1.5;
      const MIN_GAP_REM  = 0.9;
      const REM_PX       = 16;
      const STUB_EST     = CORNER_R + 3;   // corner-rounding clearance at each band
      let needsReflow = false;
      // Override CSS gap with 0 — margins handle all spacing
      pipelineEl.style.gap = '0px';
      for (let i = 0; i < gapCounts.length; i++) {
        const n = gapCounts[i];
        // ≤1 run turns here → no stacking, keep the base gap (avoids bloating
        // trivial corridors). ≥2 runs → grow to fit them at CHANNEL_GAP spacing.
        const needRem = n >= 2 ? ((n - 1) * CHANNEL_GAP + 2 * STUB_EST) / REM_PX : 0;
        const gap = Math.max(MIN_GAP_REM, BASE_GAP_REM, needRem);
        pipeChildren[i].style.marginBottom = `${gap.toFixed(2)}rem`;
        needsReflow = true;
      }

      if (needsReflow) {
        // Force reflow so subsequent rect queries reflect new spacing
        pipelineEl.offsetHeight; // eslint-disable-line no-unused-expressions
        alignPipelineStages();
      }
    }

    const wRect = wrapper.getBoundingClientRect();

    /* ── Element resolution ───────────────────────────────────── */
    const srcMap = {
      'src-rgb':  document.getElementById('card-rgb'),
      'src-lwir': document.getElementById('card-lwir'),
    };

    function resolveEl(id) {
      return id.startsWith('src-') ? srcMap[id]
           : id.startsWith('op-')  ? data.opNodeEls[id]
           : data.cardEls[id];
    }

    /** Bounding rect of `el` relative to the wrapper. */
    function rr(el) {
      if (!el) return null;
      const r = el.getBoundingClientRect();
      if (!r.width) return null;
      return {
        l: r.left - wRect.left,  r: r.right  - wRect.left,
        t: r.top  - wRect.top,   b: r.bottom - wRect.top,
        cx: (r.left + r.right) / 2 - wRect.left,
        w: r.width, h: r.height,
      };
    }

    /** Convert an array of waypoints to an SVG path string
     *  with smooth quadratic curves at every orthogonal corner. */
    /** Drop consecutive points <1px apart: zero-length segments otherwise hit
     *  the straight-L branch below (no corner rounding → pointy corners) and
     *  produce degenerate all-same-point paths. */
    function dedupePts(pts) {
      if (!pts || pts.length < 2) return pts || [];
      const out = [pts[0]];
      for (let i = 1; i < pts.length; i++) {
        const p = out[out.length - 1], c = pts[i];
        if (Math.hypot(c.x - p.x, c.y - p.y) >= 1) out.push(c);
      }
      return out;
    }

    function ptsToD(pts) {
      pts = dedupePts(pts);
      if (pts.length < 2) return '';
      if (pts.length === 2) {
        return `M ${pts[0].x} ${pts[0].y} L ${pts[1].x} ${pts[1].y}`;
      }
      let d = `M ${pts[0].x} ${pts[0].y}`;
      for (let i = 1; i < pts.length; i++) {
        if (i < pts.length - 1) {
          const p = pts[i - 1], c = pts[i], n = pts[i + 1];
          const d1x = c.x - p.x, d1y = c.y - p.y;
          const d2x = n.x - c.x, d2y = n.y - c.y;
          const l1 = Math.hypot(d1x, d1y), l2 = Math.hypot(d2x, d2y);
          if (l1 < 1 || l2 < 1) { d += ` L ${c.x} ${c.y}`; continue; }
          const cr = Math.min(CORNER_R, l1 * 0.45, l2 * 0.45);
          d += ` L ${c.x - (d1x / l1) * cr} ${c.y - (d1y / l1) * cr}`
            +  ` Q ${c.x} ${c.y} ${c.x + (d2x / l2) * cr} ${c.y + (d2y / l2) * cr}`;
        } else {
          d += ` L ${pts[i].x} ${pts[i].y}`;
        }
      }
      return d;
    }

    /* ════════════════════════════════════════════════════════════
       PHASE 1 — Obstacle collection
    ════════════════════════════════════════════════════════════ */
    const obstacles = [];
    for (const el of Object.values(data.cardEls))   { const rc = rr(el); if (rc) obstacles.push(rc); }
    for (const el of Object.values(data.opNodeEls)) { const rc = rr(el); if (rc) obstacles.push(rc); }

    /* ════════════════════════════════════════════════════════════
       PHASE 2 — Detect horizontal element bands & routing corridors
       A "band" is a horizontal row of elements.
       A "corridor" is the vertical gap between two consecutive bands
       where horizontal routing segments are allowed.
    ════════════════════════════════════════════════════════════ */
    const bands = [];
    const BAND_TOL = 12;

    function addToBand(rect) {
      const cy = (rect.t + rect.b) / 2;
      for (const band of bands) {
        if (Math.abs(band.cy - cy) < BAND_TOL) {
          band.t = Math.min(band.t, rect.t);
          band.b = Math.max(band.b, rect.b);
          return;
        }
      }
      bands.push({ t: rect.t, b: rect.b, cy });
    }

    obstacles.forEach(addToBand);
    for (const id of Object.keys(srcMap)) {
      const rc = rr(srcMap[id]);
      if (rc) addToBand(rc);
    }
    bands.sort((a, b) => a.cy - b.cy);

    const corridors = [];
    for (let i = 0; i < bands.length - 1; i++) {
      const gapH = bands[i + 1].t - bands[i].b;
      if (gapH > 4) {
        corridors.push({
          t: bands[i].b,  b: bands[i + 1].t,
          cy: (bands[i].b + bands[i + 1].t) / 2,
          h: gapH,  idx: corridors.length,
          above: i,  below: i + 1,
        });
      }
    }

    /* ════════════════════════════════════════════════════════════
       PHASE 3 — Build edges & apply port spread
    ════════════════════════════════════════════════════════════ */
    const bundleConns = data.connections.filter((c) => c.bundle);
    const edges = data.connections.filter((c) => !c.bundle).map((conn) => {
      const fromR = rr(resolveEl(conn.from));
      const toR   = rr(resolveEl(conn.to));
      if (!fromR || !toR) return null;
      return {
        conn,
        fid: conn.from, tid: conn.to,
        x1: fromR.cx, y1: fromR.b,
        x2: toR.cx,   y2: toR.t,
        fromW: fromR.w, toW: toR.w,
        fromCx: fromR.cx, toCx: toR.cx,
        fromCy: (fromR.t + fromR.b) / 2, toCy: (toR.t + toR.b) / 2,
      };
    }).filter(Boolean);

    /* ── Routing helpers (shared by classification & path building) ── */
    /** Horizontal line at `y` spanning [xMin,xMax] hits an obstacle? */
    function isBlockedAt(y, xMin, xMax) {
      return obstacles.some((o) =>
        y >= o.t - OBS_PAD && y <= o.b + OBS_PAD &&
        xMax >= o.l - OBS_PAD && xMin <= o.r + OBS_PAD);
    }
    /** Vertical line at `x` spanning [yTop,yBot] is clear of obstacles? */
    function clearVertical(x, yTop, yBot) {
      return !obstacles.some((o) =>
        x >= o.l - OBS_PAD && x <= o.r + OBS_PAD &&
        yBot >= o.t - OBS_PAD && yTop <= o.b + OBS_PAD);
    }
    function bandIndexOf(y) {
      let bi = 0, bd = Infinity;
      for (let i = 0; i < bands.length; i++) {
        const d = Math.abs(bands[i].cy - y);
        if (d < bd) { bd = d; bi = i; }
      }
      return bi;
    }
    const srcCorrOf = (bi) => corridors.find((c) => c.above === bi);
    const tgtCorrOf = (bi) => corridors.find((c) => c.below === bi);
    const pipeCx = obstacles.length
      ? obstacles.reduce((s, o) => s + o.cx, 0) / obstacles.length
      : wRect.width / 2;
    /** Obstacle x-extent crossing the vertical span [top,bot]. */
    function laneSpan(top, bot) {
      let l = Infinity, r = -Infinity;
      obstacles.forEach((o) => {
        if (o.b > top && o.t < bot) { l = Math.min(l, o.l); r = Math.max(r, o.r); }
      });
      return { l, r };
    }

    /* ── Classify each edge BEFORE spreading ports ──────────────────
       We work out where every edge first heads (its target column for a
       single jog, or the vertical lane it will drop down for a band-skip).
       Ports are then ordered by that heading X so the line leaving a node
       on the side nearest its destination never crosses a sibling — the
       generic "near/far" rule that is invariant to left vs right side.
    ──────────────────────────────────────────────────────────────── */
    edges.forEach((e) => {
      // Classify by each node's CENTRE (tall source/output cards would be
      // mis-assigned if we used their top/bottom edge).
      e.bS = bandIndexOf(e.fromCy);
      e.bT = bandIndexOf(e.toCy);
      e.sCorr = srcCorrOf(e.bS);
      e.tCorr = tgtCorrOf(e.bT);

      if (Math.abs(e.fromCx - e.toCx) < 3) { e.kind = 'v'; }
      else if (e.sCorr && e.tCorr && e.sCorr === e.tCorr) { e.kind = 'jog'; }
      else if (!e.sCorr || !e.tCorr) { e.kind = 'jog'; }
      else { e.kind = 'lane'; }

      if (e.kind === 'lane') {
        const top = e.sCorr.cy, bot = e.tCorr.cy;
        if (clearVertical(e.fromCx, top, bot))      e.laneHint = e.fromCx;
        else if (clearVertical(e.toCx, top, bot))   e.laneHint = e.toCx;
        else {
          const span = laneSpan(top, bot);
          e.laneHint = e.fromCx <= pipeCx ? span.l - GUTTER : span.r + GUTTER;
        }
        e.headFrom = e.laneHint;   // heads toward its lane
        e.headTo   = e.laneHint;   // arrives from its lane
      } else {
        e.headFrom = e.toCx;       // heads toward the target column
        e.headTo   = e.fromCx;     // arrives from the source column
      }
    });

    // Fan-out / fan-in: spread exit/entry ports across a shared node
    const grpFrom = new Map(), grpTo = new Map();
    edges.forEach((e, i) => {
      if (!grpFrom.has(e.fid)) grpFrom.set(e.fid, []);
      grpFrom.get(e.fid).push(i);
      if (!grpTo.has(e.tid)) grpTo.set(e.tid, []);
      grpTo.get(e.tid).push(i);
    });

    /* ── Coordinated band-skip fan-out (channel card → fused cards) ──
       A non-op source card that band-skips to ≥2 cards all on ONE side
       (e.g. thermal T → fus-r/g/b). Phase 5's conv/far ordering is
       unstable for these (the source ties with a 3-input fused card,
       flipping opAbove), so its run-Y order would not match whatever
       port order Phase 3 picks → same-colour drops cut sibling runs.
       We stamp a STABLE rank from destination distance only (never from
       the spread x1/x2). Both spreadPorts (below) and Phase 5 read this
       same rank, so port order and channel order can no longer disagree.
       fanRank 0 = farthest-reaching target → away-side exit port (Phase 3)
       + run closest to the cards (Phase 5), so its long drop nests on the
       outside and clears every shorter sibling run. */
    grpFrom.forEach((idxArr, fid) => {
      if (fid.startsWith('op-')) return;                 // leave op-node fans untouched
      // Stamp only the LANE (band-skip) edges of this source. A channel/thermal
      // card frequently ALSO jogs to an op-node (e.g. T → SOBELEDGEOP) — that jog
      // must NOT disqualify the fused-card lane fan that actually crosses. The
      // earlier all-lane requirement saw the jog and skipped stamping entirely,
      // so the fan fell back to the generic order and its same-colour lines
      // crossed (the whole fanRank machinery was effectively dead for sobel/
      // ssim/superpixel — confirmed: every ch-t group was rejected 'not-all-lane').
      const laneIdx = idxArr.filter((i) => edges[i].kind === 'lane');
      if (laneIdx.length < 2) return;
      const fromCx = edges[laneIdx[0]].fromCx;
      const sides  = laneIdx.map((i) => Math.sign(edges[i].toCx - fromCx));
      if (!sides.every((s) => s === sides[0]) || sides[0] === 0) return; // must be one-sided
      const dir = sides[0];                              // -1 targets left, +1 targets right
      // Rank by |toCx - fromCx| desc → farthest target = rank 0.
      const order = laneIdx.slice().sort((a, b) =>
        Math.abs(edges[b].toCx - fromCx) - Math.abs(edges[a].toCx - fromCx));
      order.forEach((i, rank) => {
        edges[i].fanRank  = rank;     // stable, destination-only ordering key
        edges[i].fanGroup = fid;      // tag so Phase 5 can re-nest this group's runs
        edges[i].fanDir   = dir;      // side the targets lie on
      });
    });

    function spreadPorts(group, isSrc) {
      group.forEach((idxArr) => {
        if (idxArr.length < 2) return;
        const ref    = edges[idxArr[0]];
        const nodeW  = isSrc ? ref.fromW : ref.toW;
        const nodeCx = isSrc ? ref.x1    : ref.x2;

        // Coordinated fan-out exit ports for a stamped one-sided band-skip fan
        // (channel card → fused cards). MONOTONIC with target side: rank 0 (the
        // farthest-reaching target) takes the exit port on the SAME side as its
        // target and turns on the TOP run (ASC channel re-deal in Phase 5);
        // nearer targets take inner ports and lower runs. Each clustered-source
        // drop then descends past — never through — the shorter siblings' runs
        // (hand-enumerated crossing-free). Only fires when EVERY edge of the
        // group is stamped (pure lane fan); mixed groups that also jog to an
        // op-node fall through to the generic head-order, which is already
        // monotonic. fanRank is destination-only, so this never reads x1 → no
        // Phase3↔Phase5 feedback loop.
        if (isSrc && idxArr.every((i) => edges[i].fanRank !== undefined)) {
          idxArr.sort((a, b) =>
            edges[a].fanDir < 0 ? edges[a].fanRank - edges[b].fanRank
                                : edges[b].fanRank - edges[a].fanRank);
        } else {
          // Order ports by where each line actually heads (its lane / target
          // column), so the run nearest a side takes the port on that side.
          idxArr.sort((a, b) =>
            isSrc ? edges[a].headFrom - edges[b].headFrom
                  : edges[a].headTo   - edges[b].headTo);
        }

        const n = idxArr.length;
        // Scale spread with the port count so 2-port fans stay tight and
        // the inputs don't land on the far corners of a wide card.
        let spread = Math.min(nodeW * 0.8, MAX_SPREAD, (n - 1) * PORT_STEP);

        // Narrow op-node targets (e.g. CURVFUSEOP/MERGE badges) have a tiny
        // nodeW, which would stack every incoming arrow on the centre even
        // when the source cards span a wide band. Widen the entry spread to
        // the span of the other endpoints (capped) so the arrows fan out and
        // stop overlapping; the lines still converge toward the node centre.
        const otherX = idxArr.map((idx) => (isSrc ? edges[idx].x2 : edges[idx].x1));
        const otherSpan = Math.max(...otherX) - Math.min(...otherX);
        if (otherSpan > spread) {
          spread = Math.min(otherSpan, MAX_SPREAD, (n - 1) * PORT_STEP);
        }

        idxArr.forEach((idx, j) => {
          const off = n === 1 ? 0 : -spread / 2 + (j / (n - 1)) * spread;
          if (isSrc) edges[idx].x1 = nodeCx + off;
          else       edges[idx].x2 = nodeCx + off;
        });
      });
    }
    spreadPorts(grpFrom, true);
    spreadPorts(grpTo,   false);

    // Straighten genuine 1-to-1 links (both endpoints sole connections),
    // by sliding the exit/entry onto the wider node's column. Avoids the
    // little "hook" on links like source→split or blend→output. Fan-out
    // children are skipped so their arrows still land on their own card.
    edges.forEach((e) => {
      if (Math.abs(e.x1 - e.x2) < 1) return;
      if (grpFrom.get(e.fid).length !== 1 || grpTo.get(e.tid).length !== 1) return;
      if (Math.abs(e.x2 - e.fromCx) <= e.fromW / 2 - 2)      e.x1 = e.x2; // wide source
      else if (Math.abs(e.x1 - e.toCx) <= e.toW / 2 - 2)     e.x2 = e.x1; // wide target
    });

    /* ════════════════════════════════════════════════════════════
       PHASE 4 — Build each edge's path through corridors / side lanes
         • jog   → a single horizontal run in the shared corridor
         • lane  → drop down a clear vertical lane so the line never
                   crosses the cards/op-nodes in between
       Edges were already classified (kind, sCorr, tCorr) before the
       ports were spread; here we finalise the actual lane X.
    ════════════════════════════════════════════════════════════ */
    corridors.forEach((c) => { c.segs = []; });
    // Register a horizontal run [xa,xb] at corridor `c`, tying its two
    // waypoint indices so Phase 5 can set their shared Y.
    function regSeg(c, e, ptA, ptB, xa, xb) {
      if (Math.abs(xa - xb) < 0.5) return;          // degenerate, stays vertical
      c.segs.push({ edge: e, ptA, ptB,
        x1: Math.min(xa, xb), x2: Math.max(xa, xb) });
    }

    const gutterEdges = [];
    edges.forEach((e) => {
      e.arcJumps = [];

      // Straight vertical when aligned and the column is clear.
      if (Math.abs(e.x1 - e.x2) < 3 && clearVertical(e.x1, e.y1, e.y2)) {
        e.pts = [{ x: e.x1, y: e.y1 }, { x: e.x2, y: e.y2 }];
        return;
      }

      // Single jog (adjacent bands, or no usable corridor pair).
      if (e.kind !== 'lane' || !e.sCorr || !e.tCorr) {
        const corr  = (e.sCorr && e.sCorr === e.tCorr) ? e.sCorr : null;
        const forkY = corr ? corr.cy : (e.y1 + e.y2) / 2;
        e.pts = [
          { x: e.x1, y: e.y1 },
          { x: e.x1, y: forkY },
          { x: e.x2, y: forkY },
          { x: e.x2, y: e.y2 },
        ];
        if (corr) regSeg(corr, e, 1, 2, e.x1, e.x2);
        return;
      }

      // Band-skip → choose the final lane from the spread ports.
      // Prefer dropping straight at the source/target column; otherwise
      // detour into a side gutter (assigned below).
      if (clearVertical(e.x1, e.sCorr.cy, e.tCorr.cy))      e.laneX = e.x1;
      else if (clearVertical(e.x2, e.sCorr.cy, e.tCorr.cy)) e.laneX = e.x2;
      else { e.laneX = null; gutterEdges.push(e); }
    });

    // Assign gutter lanes (only the edges that couldn't drop straight).
    const leftG  = gutterEdges.filter((e) => e.x1 <= pipeCx)
      .sort((a, b) => Math.min(a.x1, a.x2) - Math.min(b.x1, b.x2));
    const rightG = gutterEdges.filter((e) => e.x1 > pipeCx)
      .sort((a, b) => Math.max(b.x1, b.x2) - Math.max(a.x1, a.x2));
    function assignGutter(group, side) {
      group.forEach((e, rank) => {
        const span = laneSpan(e.sCorr.cy, e.tCorr.cy);
        const base = side < 0 ? span.l - GUTTER : span.r + GUTTER;
        e.laneX = base + side * rank * LANE_STEP; // outermost first
      });
    }
    assignGutter(leftG, -1);
    assignGutter(rightG, 1);

    // Build the skip-edge paths now that every lane is known.
    edges.forEach((e) => {
      if (e.kind !== 'lane' || e.laneX == null || !e.sCorr) return;
      const lx = e.laneX;
      e.pts = [
        { x: e.x1, y: e.y1 },
        { x: e.x1, y: e.sCorr.cy },
        { x: lx,   y: e.sCorr.cy },
        { x: lx,   y: e.tCorr.cy },
        { x: e.x2, y: e.tCorr.cy },
        { x: e.x2, y: e.y2 },
      ];
      regSeg(e.sCorr, e, 1, 2, e.x1, lx);
      regSeg(e.tCorr, e, 3, 4, lx, e.x2);
    });

    /* ════════════════════════════════════════════════════════════
       PHASE 5 — Stack horizontal runs on ordered channels
       Within a corridor the runs all converge on (or diverge from)
       one op-node. They are nested so the run reaching the card
       farthest from that node hugs the node itself, while inner runs
       turn nearer the cards. This is the order that lets every
       perpendicular drop clear the others — no avoidable line-jumps.
       The nesting flips depending on whether the op-node sits above
       the corridor (fan-out) or below it (fan-in).
    ════════════════════════════════════════════════════════════ */
    corridors.forEach((c) => {
      const segs = c.segs;
      if (!segs.length) return;

      // Convergence node = the one shared by the most runs in this corridor.
      const tally = new Map();
      function bump(id, cx, asFrom) {
        let t = tally.get(id);
        if (!t) { t = { n: 0, cx, from: 0, to: 0 }; tally.set(id, t); }
        t.n++; if (asFrom) t.from++; else t.to++;
      }
      segs.forEach(({ edge: e }) => { bump(e.fid, e.fromCx, true); bump(e.tid, e.toCx, false); });
      let conv = null, bestN = -1;
      tally.forEach((t) => { if (t.n > bestN) { bestN = t.n; conv = t; } });
      const convergeX = conv ? conv.cx : c.cy;
      // op-node mostly a source → it is above the corridor → fan-out.
      const opAbove = conv ? conv.from >= conv.to : false;

      // Distance of each run's far end from the convergence point. The run
      // reaching farthest must hug the op-node, so it is placed first.
      segs.forEach((s) => {
        s.far = Math.max(Math.abs(s.x1 - convergeX), Math.abs(s.x2 - convergeX));
      });
      segs.sort((a, b) => b.far - a.far);

      // Interval packing: runs that do NOT overlap horizontally SHARE a Y
      // channel — so symmetric, non-touching runs line up at the SAME height and
      // the corridor stays compact; only runs that would actually overlap open a
      // new channel. Channel 0 hugs the op-node; segs are sorted by far-distance
      // desc so the farthest-reaching run anchors channel 0. Any crossing between
      // runs on different channels is resolved cleanly by arc-jumps at render
      // time (the arc-jump dedupe bug that used to tangle these is fixed), so we
      // no longer need one-channel-per-run to avoid weaving.
      const HGAP = 8;        // min horizontal clearance to treat two runs as disjoint
      const channels = [];   // each: array of placed segs
      segs.forEach((s) => {
        let ci = 0;
        for (; ci < channels.length; ci++) {
          const clash = channels[ci].some((o) => s.x1 <= o.x2 + HGAP && o.x1 <= s.x2 + HGAP);
          if (!clash) break;
        }
        if (ci === channels.length) channels.push([]);
        channels[ci].push(s);
        s.ch = ci;
      });

      // Keep a stub clear of each band so the turns into a node always
      // have room for a rounded corner (no squashed right-angles).
      const K    = channels.length;
      // Stub clearance at each band for the rounded-corner turn. Must MATCH the
      // adaptive-margin STUB_EST so the corridor height provisioned there yields
      // gap ≥ CHANNEL_GAP here (otherwise runs collapse → pegadas). CORNER_R+3
      // still exceeds CORNER_R so corners round fully, while keeping corridors
      // compact (less empty vertical space between blocks).
      const STUB = Math.min(CORNER_R + 3, c.h * 0.4);
      const gap  = K > 1 ? (c.h - 2 * STUB) / (K - 1) : 0;
      segs.forEach((s) => {
        // Channel 0 hugs the op-node (top for fan-out, bottom for fan-in).
        const y = K === 1 ? c.cy
          : opAbove ? c.t + STUB + s.ch * gap
                    : c.b - STUB - s.ch * gap;
        s.edge.pts[s.ptA].y = y;
        s.edge.pts[s.ptB].y = y;
      });

      /* ── Stable nesting for coordinated band-skip fan-outs ──────────
         The generic conv/far ordering above is UNSTABLE for a channel
         card that band-skips to several fused cards on one side (e.g.
         thermal ch-t → fus-r/g/b): ch-t (from=3) ties with each 3-input
         fused card (to=3), so `conv`/`opAbove` flip arbitrarily and the
         run-Y order no longer matches the exit-port order Phase 3 chose.
         The same-colour result is ch-t's own lines crossing each other.

         Fix WITHOUT disturbing any other run: take only this group's segs
         in THIS corridor, keep the exact set of Y slots they already
         occupy (so no other-colour run moves and the corridor envelope is
         unchanged), and re-deal those Y values in the destination-stable
         fanRank order. The farthest-reaching target (fanRank 0) gets the
         slot CLOSEST to the cards — which, paired with the matching away-side
         exit port from Phase 3, makes every vertical drop nest instead of
         cutting a sibling run. fanRank derives only from destination X, so
         this is immune to the Phase3↔Phase5 x1/x2 feedback that undid the
         earlier port-only attempt. op-node fans are never stamped, so they
         are byte-identical here. */
      const fanIds = new Set(
        segs.filter((s) => s.edge.fanGroup !== undefined)
            .map((s) => s.edge.fanGroup));
      fanIds.forEach((fid) => {
        const gSegs = segs.filter((s) => s.edge.fanGroup === fid);
        if (gSegs.length < 2) return;
        const ys = gSegs.map((s) => s.edge.pts[s.ptA].y);
        // Slots ordered farthest-from-cards (top) → nearest-cards (bottom).
        ys.sort((a, b) => a - b);
        // Pair with fanRank ASCENDING: rank 0 (farthest target) turns at the TOP
        // run (ys[0], nearest the source/op-node) with the shortest drop on its
        // target side; nearer targets take progressively lower runs and ports
        // further toward the source. So each clustered-source vertical drop
        // descends to the right (for a left fan) of every higher sibling run and
        // clears it — hand-verified crossing-free, and matches the monotonic
        // exit-port order set in Phase 3.
        gSegs.slice().sort((a, b) => a.edge.fanRank - b.edge.fanRank)
          .forEach((s, i) => {
            s.edge.pts[s.ptA].y = ys[i];
            s.edge.pts[s.ptB].y = ys[i];
          });
      });

      // De-coincidence: the fan re-deal can land a (wide) fan run on a channel the
      // interval packing had filled with narrow X-disjoint NON-fan runs (e.g. the
      // thermal rank-0 run spans over the weight jogs sharing the top channel) →
      // two different-colour lines drawn on top of each other. Nudge such a fan run
      // a fraction of the gap toward the corridor interior so the thermal fan and
      // the weight/channel runs stay visually separate. Only fan runs move; the
      // small offset keeps them between their own rank siblings (no new same-colour
      // crossing) and the corridor envelope is unchanged.
      if (gap > 0) {
        const nonFan = segs.filter((s) => s.edge.fanGroup === undefined)
          .map((s) => ({ y: s.edge.pts[s.ptA].y, x1: s.x1, x2: s.x2 }));
        segs.filter((s) => s.edge.fanGroup !== undefined).forEach((s) => {
          const y = s.edge.pts[s.ptA].y;
          const hit = nonFan.some((o) => Math.abs(o.y - y) < 3
            && s.x1 <= o.x2 + HGAP && o.x1 <= s.x2 + HGAP);
          if (!hit) return;
          const ny = y + (y <= c.cy ? 1 : -1) * gap * 0.45; // toward corridor interior
          s.edge.pts[s.ptA].y = ny;
          s.edge.pts[s.ptB].y = ny;
        });
      }
    });

    /* ════════════════════════════════════════════════════════════
       PHASE 6 — Nudge any horizontal run still hitting an obstacle
    ════════════════════════════════════════════════════════════ */
    edges.forEach((e) => {
      if (!e.pts || e.pts.length < 4) return;
      for (let i = 0; i < e.pts.length - 1; i++) {
        const a = e.pts[i], b = e.pts[i + 1];
        if (Math.abs(a.y - b.y) > 0.5) continue;        // skip verticals
        const xMin = Math.min(a.x, b.x), xMax = Math.max(a.x, b.x);
        if (!isBlockedAt(a.y, xMin, xMax)) continue;
        for (let d = 1; d < 80; d++) {
          let moved = false;
          for (const cand of [a.y - d, a.y + d]) {
            if (!isBlockedAt(cand, xMin, xMax)) { a.y = b.y = cand; moved = true; break; }
          }
          if (moved) break;
        }
      }
    });

    // Dedupe each path's waypoints ONCE here so the arc-jump segment indices
    // computed next match what ptsWithArcs renders. Previously ptsWithArcs
    // re-deduped at render time, shifting indices so arcs attached to the wrong
    // segment → tangled paths and diagonal closing segments in dense fan-ins.
    edges.forEach((e) => { if (e.pts) e.pts = dedupePts(e.pts); });

    /* ════════════════════════════════════════════════════════════
       PHASE 6c — Monotone turn-height repair
       Shift interior horizontal runs to a better turn-height, KEEPING a
       shift only if it lowers a cost = (#inter-edge crossings) + a small
       penalty for parallel overlaps. By construction this can only REDUCE
       crossings, never add them — it "manages where each line turns" to
       undo the avoidable double-cuts at dense fan-ins. Skipped entirely
       when a diagram is already crossing-free (the clean ones cost nothing).
    ════════════════════════════════════════════════════════════ */
    function routeStats() {
      let coinc = 0, crossSum = 0, dbl = 0, len = 0, kink = 0;
      for (let i = 0; i < edges.length; i++) {
        const pi = edges[i].pts; if (!pi) continue;
        for (let k = 0; k < pi.length - 1; k++)           // total orthogonal path length
          len += Math.abs(pi[k].x - pi[k + 1].x) + Math.abs(pi[k].y - pi[k + 1].y);
        // Cramped corners: an interior segment with a real bend at BOTH ends that
        // is shorter than 2·CORNER_R can't fit the full corner radius on each side
        // → a tight "quiebro apurado" (arrow + radius colliding). Penalise it.
        for (let k = 1; k < pi.length - 2; k++) {
          const s1h = Math.abs(pi[k - 1].y - pi[k].y) < 0.5;
          const s2h = Math.abs(pi[k].y - pi[k + 1].y) < 0.5;
          const s3h = Math.abs(pi[k + 1].y - pi[k + 2].y) < 0.5;
          if (s1h === s2h || s2h === s3h) continue;       // need a bend at both ends of seg k
          if (Math.abs(pi[k].x - pi[k + 1].x) + Math.abs(pi[k].y - pi[k + 1].y) < 2 * CORNER_R) kink++;
        }
        for (let j = i + 1; j < edges.length; j++) {
          const pj = edges[j].pts; if (!pj) continue;
          let pc = 0;                                       // crossings between THIS pair
          for (let a = 0; a < pi.length - 1; a++) {
            const aH = Math.abs(pi[a].y - pi[a + 1].y) < 0.5;
            for (let b = 0; b < pj.length - 1; b++) {
              const bH = Math.abs(pj[b].y - pj[b + 1].y) < 0.5;
              if (aH !== bH) {                              // perpendicular → crossing
                const h = aH ? pi : pj, hi = aH ? a : b, v = aH ? pj : pi, vi = aH ? b : a;
                const hy = h[hi].y, hx1 = Math.min(h[hi].x, h[hi + 1].x), hx2 = Math.max(h[hi].x, h[hi + 1].x);
                const vx = v[vi].x, vy1 = Math.min(v[vi].y, v[vi + 1].y), vy2 = Math.max(v[vi].y, v[vi + 1].y);
                if (vx > hx1 + 1 && vx < hx2 - 1 && hy > vy1 + 1 && hy < vy2 - 1) pc++;
              } else if (aH) {                              // both horizontal → X-extent overlap (Y coincidence)
                if (Math.abs(pi[a].y - pj[b].y) < 5) {
                  const ov = Math.min(Math.max(pi[a].x, pi[a + 1].x), Math.max(pj[b].x, pj[b + 1].x))
                           - Math.max(Math.min(pi[a].x, pi[a + 1].x), Math.min(pj[b].x, pj[b + 1].x));
                  if (ov > 10) coinc++;
                }
              } else {                                      // both vertical → Y-extent overlap (X coincidence)
                if (Math.abs(pi[a].x - pj[b].x) < 5) {
                  const ov = Math.min(Math.max(pi[a].y, pi[a + 1].y), Math.max(pj[b].y, pj[b + 1].y))
                           - Math.max(Math.min(pi[a].y, pi[a + 1].y), Math.min(pj[b].y, pj[b + 1].y));
                  if (ov > 10) coinc++;
                }
              }
            }
          }
          crossSum += pc;
          if (pc >= 2) dbl += pc - 1;                       // pair crossing ≥2× = double-cross; the extra crossings are pointless
        }
      }
      return { coinc, crossSum, dbl, len, kink };
    }
    // Priority order = the user's: (1) NO X/Y overlaps (always avoidable), (2) NO
    // double-crosses (two lines crossing each other twice — always avoidable),
    // (3) minimise normal crossings (some forced by genuine interleaves → clean arc-hops).
    function routeCost() {
      const s = routeStats();
      // + tiny length term: among equal-crossing options, prefer the SHORTER /
      // cleaner path — kills gratuitous kinks and wandering detours (e.g. a line
      // entering an op-node from below, or an entry-port slid too far). Weight is
      // small enough that it never overrides a crossing/overlap/double-cross.
      return 1000 * s.coinc + 100 * s.dbl + 2 * s.kink + s.crossSum + 0.002 * s.len;
    }
    // Per-edge cost: edge e's OWN length+kinks plus its crossings/overlaps with the
    // OTHER edges. Shifting one of e's segments changes ONLY this (every other-other
    // pair is constant), so the repair evaluates a candidate in O(E·S²) instead of
    // re-scanning all pairs O(E²·S²) — the heavy full-scan per candidate is what
    // made dense diagrams hang.
    function edgeStats(e) {
      const pe = e.pts; let coinc = 0, crossSum = 0, dbl = 0, len = 0, kink = 0;
      if (!pe) return { coinc, crossSum, dbl, len, kink };
      for (let k = 0; k < pe.length - 1; k++) len += Math.abs(pe[k].x - pe[k + 1].x) + Math.abs(pe[k].y - pe[k + 1].y);
      for (let k = 1; k < pe.length - 2; k++) {
        const s1h = Math.abs(pe[k - 1].y - pe[k].y) < 0.5, s2h = Math.abs(pe[k].y - pe[k + 1].y) < 0.5, s3h = Math.abs(pe[k + 1].y - pe[k + 2].y) < 0.5;
        if (s1h === s2h || s2h === s3h) continue;
        if (Math.abs(pe[k].x - pe[k + 1].x) + Math.abs(pe[k].y - pe[k + 1].y) < 2 * CORNER_R) kink++;
      }
      for (const o of edges) {
        const po = o.pts; if (o === e || !po) continue;
        let pc = 0;
        for (let a = 0; a < pe.length - 1; a++) {
          const aH = Math.abs(pe[a].y - pe[a + 1].y) < 0.5;
          for (let b = 0; b < po.length - 1; b++) {
            const bH = Math.abs(po[b].y - po[b + 1].y) < 0.5;
            if (aH !== bH) {
              const h = aH ? pe : po, hi = aH ? a : b, v = aH ? po : pe, vi = aH ? b : a;
              const hy = h[hi].y, hx1 = Math.min(h[hi].x, h[hi + 1].x), hx2 = Math.max(h[hi].x, h[hi + 1].x);
              const vx = v[vi].x, vy1 = Math.min(v[vi].y, v[vi + 1].y), vy2 = Math.max(v[vi].y, v[vi + 1].y);
              if (vx > hx1 + 1 && vx < hx2 - 1 && hy > vy1 + 1 && hy < vy2 - 1) pc++;
            } else if (aH) {
              if (Math.abs(pe[a].y - po[b].y) < 5) {
                const ov = Math.min(Math.max(pe[a].x, pe[a + 1].x), Math.max(po[b].x, po[b + 1].x)) - Math.max(Math.min(pe[a].x, pe[a + 1].x), Math.min(po[b].x, po[b + 1].x));
                if (ov > 10) coinc++;
              }
            } else if (Math.abs(pe[a].x - po[b].x) < 5) {
              const ov = Math.min(Math.max(pe[a].y, pe[a + 1].y), Math.max(po[b].y, po[b + 1].y)) - Math.max(Math.min(pe[a].y, pe[a + 1].y), Math.min(po[b].y, po[b + 1].y));
              if (ov > 10) coinc++;
            }
          }
        }
        crossSum += pc; if (pc >= 2) dbl += pc - 1;
      }
      return { coinc, crossSum, dbl, len, kink };
    }
    function edgeCost(e) { const s = edgeStats(e); return 1000 * s.coinc + 100 * s.dbl + 2 * s.kink + s.crossSum + 0.002 * s.len; }
    const s0 = routeStats();
    if (s0.coinc + s0.dbl + s0.kink + s0.crossSum > 0) {   // skip diagrams with no structural issue
      const OFFS = [-70, -55, -40, -30, -20, -12, -8, -5, 5, 8, 12, 20, 30, 40, 55, 70];
      let pass = 0, improved = true;
      while (improved && pass++ < 4) {
        improved = false;
        for (const e of edges) {
          const p = e.pts; if (!p || p.length < 4) continue;
          // Only repair edges that ARE part of a problem (both partners of a
          // crossing/overlap qualify, so no fix is missed). Skipping the many clean
          // edges is the bulk of the speed-up.
          const es0 = edgeStats(e);
          if (es0.coinc + es0.dbl + es0.kink + es0.crossSum === 0) continue;
          for (let i = 1; i < p.length - 2; i++) {
            if (Math.abs(p[i].y - p[i + 1].y) > 0.5) continue;     // run must be horizontal
            if (Math.abs(p[i].x - p[i + 1].x) < 4) continue;       // long enough to matter
            if (Math.abs(p[i - 1].x - p[i].x) > 0.5) continue;     // flanked by verticals (true interior run)
            if (Math.abs(p[i + 2].x - p[i + 1].x) > 0.5) continue;
            const y0 = p[i].y;
            const xMin = Math.min(p[i].x, p[i + 1].x), xMax = Math.max(p[i].x, p[i + 1].x);
            let bestY = y0, bestC = edgeCost(e);
            for (const dy of OFFS) {
              const cy = y0 + dy;
              if (isBlockedAt(cy, xMin, xMax)) continue;
              p[i].y = cy; p[i + 1].y = cy;
              const cc = edgeCost(e);
              if (cc < bestC - 1e-6) { bestC = cc; bestY = cy; }
            }
            p[i].y = bestY; p[i + 1].y = bestY;            // commit best (or restore y0)
            if (bestY !== y0) improved = true;
          }
          // Mirror in X: shift interior VERTICAL segments to a clear lane, same
          // monotone rule. Y-shifts alone can't always undo an overlap/double-cross
          // in a crowded corridor; the X freedom lets a lane step aside.
          for (let i = 1; i < p.length - 2; i++) {
            if (Math.abs(p[i].x - p[i + 1].x) > 0.5) continue;   // segment must be vertical
            if (Math.abs(p[i].y - p[i + 1].y) < 4) continue;     // long enough to matter
            if (Math.abs(p[i - 1].y - p[i].y) > 0.5) continue;   // flanked by horizontals
            if (Math.abs(p[i + 2].y - p[i + 1].y) > 0.5) continue;
            const x0 = p[i].x;
            const yMin = Math.min(p[i].y, p[i + 1].y), yMax = Math.max(p[i].y, p[i + 1].y);
            let bestX = x0, bestC = edgeCost(e);
            for (const dx of OFFS) {
              const cx = x0 + dx;
              if (!clearVertical(cx, yMin, yMax)) continue;
              p[i].x = cx; p[i + 1].x = cx;
              const cc = edgeCost(e);
              if (cc < bestC - 1e-6) { bestC = cc; bestX = cx; }
            }
            p[i].x = bestX; p[i + 1].x = bestX;
            if (bestX !== x0) improved = true;
          }
          // Entry-port slide, CONSTRAINED: move the arrival point within the target
          // card ONLY when it removes a STRUCTURAL problem (overlap/double-cross/
          // crossing) — never merely to shorten. This unties same-target double-
          // crosses (two lines into one card crossing twice) WITHOUT pulling the
          // many clean, single-input arrivals off their card centre (the bug that
          // made "lines arrive anywhere"). Among structure-reducing spots, the
          // shortest is chosen so the fixed arrival still sits sensibly.
          const L = p.length;
          if (L >= 3 && e.toW && Math.abs(p[L - 1].x - p[L - 2].x) < 0.5
              && Math.abs(p[L - 2].y - p[L - 3].y) < 0.5) {
            const x0 = p[L - 1].x, lo = e.toCx - e.toW / 2 + 6, hi = e.toCx + e.toW / 2 - 6;
            // struct = overlaps + double-crosses ONLY (NOT plain crossings): the
            // arrival is moved only to undo a nonsensical overlap/double-cross,
            // never to dodge a tolerable single crossing (which would needlessly
            // pull the arrow off its card centre).
            const struct = () => { const s = edgeStats(e); return { st: 1000 * s.coinc + 100 * s.dbl + 2 * s.kink, len: s.len }; };
            const base = struct().st;
            let bestX = null, bestSt = base, bestLen = Infinity;
            for (const dx of OFFS) {
              const cx = x0 + dx; if (cx < lo || cx > hi) continue;
              p[L - 1].x = cx; p[L - 2].x = cx;
              const r = struct();
              if (r.st < base - 1e-6 && (r.st < bestSt - 1e-6 || (Math.abs(r.st - bestSt) < 1e-6 && r.len < bestLen))) {
                bestSt = r.st; bestLen = r.len; bestX = cx;
              }
            }
            p[L - 1].x = bestX != null ? bestX : x0;       // move ONLY on a structural win, else stay put
            p[L - 2].x = p[L - 1].x;
            if (bestX != null) improved = true;
          }
        }
      }
      edges.forEach((e) => { if (e.pts) e.pts = dedupePts(e.pts); }); // shifts may collinearise
    }
    try { window.__rstats = routeStats(); } catch (_) {} // dev metric (overlaps/double-cross/crossings)

    /* ════════════════════════════════════════════════════════════
       PHASE 6b — Detect edge crossings & mark arc-jump points
       For each pair of edges, find segment intersections.
       On the edge with the higher Y horizontal segment (drawn
       "on top"), insert a small semicircle arc at the crossing.
    ════════════════════════════════════════════════════════════ */
    const ARC_R = 6;  // arc-jump radius cap in px (semicircle; shrinks adaptively where crossings are tight)

    /** Test if two axis-aligned segments intersect. Returns intersection point or null. */
    function segIntersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) {
      // Only handle orthogonal cases: one horizontal + one vertical
      const aHoriz = Math.abs(ay1 - ay2) < 0.5;
      const bHoriz = Math.abs(by1 - by2) < 0.5;
      if (aHoriz === bHoriz) return null; // both horizontal or both vertical
      // Make 'h' the horizontal segment, 'v' the vertical
      let hx1, hx2, hy, vx, vy1, vy2;
      if (aHoriz) {
        hy = ay1; hx1 = Math.min(ax1, ax2); hx2 = Math.max(ax1, ax2);
        vx = bx1; vy1 = Math.min(by1, by2); vy2 = Math.max(by1, by2);
      } else {
        hy = by1; hx1 = Math.min(bx1, bx2); hx2 = Math.max(bx1, bx2);
        vx = ax1; vy1 = Math.min(ay1, ay2); vy2 = Math.max(ay1, ay2);
      }
      if (vx > hx1 + 1 && vx < hx2 - 1 && hy > vy1 + 1 && hy < vy2 - 1) {
        return { x: vx, y: hy };
      }
      return null;
    }

    // For each edge, collect arc-jump points (on horizontal segments only)
    edges.forEach((e) => { e.arcJumps = []; });

    for (let i = 0; i < edges.length; i++) {
      for (let j = i + 1; j < edges.length; j++) {
        const ea = edges[i], eb = edges[j];
        // Check all segment pairs between the two edges
        for (let si = 0; si < ea.pts.length - 1; si++) {
          for (let sj = 0; sj < eb.pts.length - 1; sj++) {
            const ix = segIntersect(
              ea.pts[si].x, ea.pts[si].y, ea.pts[si + 1].x, ea.pts[si + 1].y,
              eb.pts[sj].x, eb.pts[sj].y, eb.pts[sj + 1].x, eb.pts[sj + 1].y,
            );
            if (!ix) continue;
            // Add arc jump on the horizontal segment (the one being crossed over)
            const aHoriz = Math.abs(ea.pts[si].y - ea.pts[si + 1].y) < 0.5;
            if (aHoriz) {
              ea.arcJumps.push({ x: ix.x, segIdx: si });
            } else {
              eb.arcJumps.push({ x: ix.x, segIdx: sj });
            }
          }
        }
      }
    }

    // Sort arc jumps along each edge's horizontal segments
    edges.forEach((e) => {
      e.arcJumps.sort((a, b) => {
        if (a.segIdx !== b.segIdx) return a.segIdx - b.segIdx;
        return a.x - b.x;
      });
    });

    /* ════════════════════════════════════════════════════════════
       PHASE 7 — SVG generation
    ════════════════════════════════════════════════════════════ */
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'pipeline-flow-svg');
    svg.setAttribute('aria-hidden', 'true');

    /** Build an SVG path string from waypoints, inserting semicircle
     *  arc-jumps on horizontal segments where this edge crosses another. */
    function ptsWithArcs(pts, arcJumps) {
      pts = dedupePts(pts);
      if (!arcJumps || !arcJumps.length) return ptsToD(pts);

      // Build sub-segments: split horizontal segments at arc-jump points
      const expanded = [];
      for (let i = 0; i < pts.length; i++) {
        const jumpsHere = arcJumps.filter((j) => j.segIdx === i);
        if (!jumpsHere.length || i >= pts.length - 1) {
          expanded.push(pts[i]);
          continue;
        }
        // This is the start of a horizontal segment with crossings
        const p1 = pts[i], p2 = pts[i + 1];
        const dir = p2.x > p1.x ? 1 : -1;
        // Sort jumps in traversal order
        jumpsHere.sort((a, b) => (a.x - b.x) * dir);

        // Adaptive radius per crossing: each hop uses at most HALF the gap to its
        // neighbouring crossings (and the full gap to the segment ends), capped at
        // ARC_R. This keeps the apex centred on the crossing AND stops adjacent
        // hops from overlapping — overlapping fixed-radius hops were what made the
        // arcs look uneven, off-centre and randomly sized in dense fan-ins.
        for (let k = 0; k < jumpsHere.length; k++) {
          const x = jumpsHere[k].x;
          const prevX = k > 0 ? jumpsHere[k - 1].x : p1.x;
          const nextX = k < jumpsHere.length - 1 ? jumpsHere[k + 1].x : p2.x;
          const r = Math.min(ARC_R,
            Math.abs(x - prevX) / (k > 0 ? 2 : 1),
            Math.abs(nextX - x) / (k < jumpsHere.length - 1 ? 2 : 1));
          jumpsHere[k]._r = Math.max(2, r);
        }
        expanded.push(p1);
        for (const j of jumpsHere) {
          // Insert arc-jump marker: approach → arc → continue (radius j._r)
          expanded.push({ x: j.x - dir * j._r, y: p1.y });
          expanded.push({ x: j.x + dir * j._r, y: p1.y, _arcJump: true, _dir: dir, _r: j._r });
        }
        // Don't push p2 here — it will be pushed in the next iteration
      }

      // Now render the expanded points, inserting arcs at markers
      if (expanded.length < 2) return ptsToD(pts);
      let d = `M ${expanded[0].x} ${expanded[0].y}`;
      for (let i = 1; i < expanded.length; i++) {
        const c = expanded[i];
        if (c._arcJump) {
          // Clean ROUND semicircle hop, centred on the crossing (apex at the
          // crossing x; approach/exit are symmetric at ±ARC_R around it).
          const sweep = c._dir > 0 ? 0 : 1;  // bulge upward regardless of direction
          const r = c._r || ARC_R;
          d += ` A ${r} ${r} 0 0 ${sweep} ${c.x} ${c.y}`;
        } else if (i < expanded.length - 1) {
          const p = expanded[i - 1], n = expanded[i + 1];
          const d1x = c.x - p.x, d1y = c.y - p.y;
          const d2x = n.x - c.x, d2y = n.y - c.y;
          const l1 = Math.hypot(d1x, d1y), l2 = Math.hypot(d2x, d2y);
          if (l1 < 1 || l2 < 1) {
            d += ` L ${c.x} ${c.y}`;
          } else {
            const cr = Math.min(CORNER_R, l1 * 0.45, l2 * 0.45);
            d += ` L ${c.x - (d1x / l1) * cr} ${c.y - (d1y / l1) * cr}`
              +  ` Q ${c.x} ${c.y} ${c.x + (d2x / l2) * cr} ${c.y + (d2y / l2) * cr}`;
          }
        } else {
          d += ` L ${c.x} ${c.y}`;
        }
      }
      return d;
    }

    // ── Regular edges ──
    edges.forEach((e) => {
      const color = e.conn.color || 'var(--clr-border)';

      // Path — plain rounded-corner orthogonal polyline. Arc-jump "hops" over
      // crossings were removed: with overlaps/double-crosses already eliminated the
      // few genuine crossings read more cleanly as plain crossings than as little
      // semicircle bumps (which never sat well and fought the corner radius).
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p.setAttribute('d', ptsToD(e.pts));
      p.setAttribute('stroke', color);
      p.setAttribute('stroke-width', '2');
      p.setAttribute('fill', 'none');
      p.setAttribute('stroke-linecap', 'round');
      svg.appendChild(p);

      // Arrowhead
      const last = e.pts[e.pts.length - 1];
      const aw = 5, ah = 7;
      const arr = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      arr.setAttribute('points',
        `${last.x} ${last.y} ${last.x - aw} ${last.y - ah} ${last.x + aw} ${last.y - ah}`);
      arr.setAttribute('fill', color);
      svg.appendChild(arr);

      // Inline op-label badge (if any)
      if (e.conn.opLabel) {
        const mid = e.pts.length >= 4
          ? { x: (e.pts[1].x + e.pts[2].x) / 2, y: e.pts[1].y }
          : { x: (e.pts[0].x + e.pts[1].x) / 2, y: (e.pts[0].y + e.pts[1].y) / 2 };
        const label = e.conn.opLabel;
        const fs = 8, pH = 6, pV = 3;
        const bw = label.length * (fs * 0.62) + pH * 2, bh = fs + pV * 2;
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', mid.x - bw / 2);
        rect.setAttribute('y', mid.y - bh / 2);
        rect.setAttribute('width', bw);
        rect.setAttribute('height', bh);
        rect.setAttribute('rx', bh / 2);
        rect.setAttribute('class', 'pipeline-op-badge-rect');
        rect.setAttribute('stroke', color);
        rect.setAttribute('stroke-width', '1');
        svg.appendChild(rect);
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', mid.x);
        text.setAttribute('y', mid.y);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('dominant-baseline', 'middle');
        text.setAttribute('font-size', fs);
        text.setAttribute('font-family', 'ui-monospace, monospace');
        text.setAttribute('font-weight', '700');
        text.setAttribute('fill', color);
        text.textContent = label;
        svg.appendChild(text);
      }
    });

    // ── Bundle connections ──
    bundleConns.forEach((conn) => {
      const fromR = rr(resolveEl(conn.from));
      const toR   = rr(resolveEl(conn.to));
      if (!fromR || !toR) return;
      const colors = conn.bundle, n = colors.length;
      const SP = 4, totalW = (n - 1) * SP;
      const fY = (fromR.b + toR.t) / 2;
      colors.forEach((color, i) => {
        const off = -totalW / 2 + i * SP;
        const pts = [
          { x: fromR.cx + off, y: fromR.b },
          { x: fromR.cx + off, y: fY },
          { x: toR.cx + off,   y: fY },
          { x: toR.cx + off,   y: toR.t },
        ];
        const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        p.setAttribute('d', ptsToD(pts));
        p.setAttribute('stroke', color);
        p.setAttribute('stroke-width', '1.5');
        p.setAttribute('fill', 'none');
        p.setAttribute('stroke-linecap', 'round');
        svg.appendChild(p);
      });
      const aw = 5, ah = 7;
      const arr = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      arr.setAttribute('points',
        `${toR.cx} ${toR.t} ${toR.cx - aw} ${toR.t - ah} ${toR.cx + aw} ${toR.t - ah}`);
      arr.setAttribute('fill', colors[Math.floor(n / 2)]);
      svg.appendChild(arr);
    });

    wrapper.appendChild(svg);
  }


  function makePipelineStage(key) {
    const el = document.createElement('div');
    el.className = 'pipeline-stage';
    el.dataset.stage = key;
    return el;
  }

  function makePipeOpNode(id, labelKey) {
    const el = document.createElement('div');
    el.id = id;
    el.className = 'pipe-op-node';
    el.dataset.i18nStage = labelKey;
    el.textContent = i18nApi.t('stageLabel')?.[labelKey] || labelKey;
    return el;
  }

  function makePipeCard(cardClass, label, canvasId, sizeClass, isOutput, method, imageData) {
    const wrapper = document.createElement('div');
    wrapper.className = `pipe-card ${cardClass}`;

    const lbl = document.createElement('div');
    lbl.className = 'pipe-card-label';
    lbl.textContent = label;
    wrapper.appendChild(lbl);

    const wrap = document.createElement('div');
    wrap.className = 'pipe-card-wrap';

    const canvas = document.createElement('canvas');
    canvas.id = canvasId;
    canvas.className = `pipe-card-canvas ${sizeClass || ''}`;
    const CANVAS_SIZES = {
      'pipe-card-canvas--sm': [120, 68],
      'pipe-card-canvas--lg': [220, 124],
      'pipe-card-canvas--xl': [320, 180],
    };
    const [cw, ch] = CANVAS_SIZES[sizeClass] || [160, 90];
    canvas.width  = cw;
    canvas.height = ch;
    wrap.appendChild(canvas);

    const expBtn = document.createElement('button');
    expBtn.className = 'pipe-card-expand btn-chart-expand btn-preview-icon shared-icon-btn shared-icon-btn-sm shared-icon-btn-accent';
    expBtn.type = 'button';
    expBtn.setAttribute('aria-label', i18nApi.t('expand') || 'Expand');
    expBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = document.getElementById(canvasId);
      if (id) openModalFromCanvas(id);
    });
    wrap.appendChild(expBtn);

    wrap.style.cursor = 'zoom-in';
    wrap.addEventListener('click', () => {
      const id = document.getElementById(canvasId);
      if (id) openModalFromCanvas(id);
    });

    wrapper.appendChild(wrap);

    if (imageData) {
      drawImageDataToCanvas(canvas, imageData);
      state.pipelineImageData[canvasId] = imageData;
    }

    if (isOutput) {
      // The output is the prominent result → a download icon sits directly on the
      // thumbnail (visible, not hover-only), replacing the old text button below.
      const dl = document.createElement('button');
      dl.className = 'pipe-card-download-vis shared-icon-btn shared-icon-btn-sm';
      dl.type = 'button';
      dl.setAttribute('aria-label', i18nApi.t('download') || 'Download');
      dl.title = i18nApi.t('download') || 'Download';
      dl.innerHTML = '<span class="icon-close"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11M8 11l4 4 4-4M5 20h14"/></svg></span>';
      dl.addEventListener('click', (e) => {
        e.stopPropagation();
        const cv = document.getElementById(canvasId);
        if (cv) downloadCanvas(cv, `fusion_${method}.png`);
      });
      wrap.appendChild(dl);
    }

    return wrapper;
  }

  function drawImageDataToCanvas(canvas, imageData) {
    const oc = new OffscreenCanvas(imageData.width, imageData.height);
    oc.getContext('2d').putImageData(imageData, 0, 0);

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scale = Math.min(canvas.width / imageData.width, canvas.height / imageData.height);
    const dw = imageData.width * scale, dh = imageData.height * scale;
    const dx = (canvas.width - dw) / 2, dy = (canvas.height - dh) / 2;
    ctx.drawImage(oc, dx, dy, dw, dh);
  }

  /* ════════════════════════════════════════════════════════════════
     PIPELINE EMPTY NOTICE
  ════════════════════════════════════════════════════════════════ */

  function showPipelineEmptyNotice(method) {
    const container = dom.pipelinePanels[method];
    container.innerHTML = `
      <div class="pipeline-empty-notice">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <path d="M3 9h18M9 21V9"/>
        </svg>
        <p data-i18n-status="needBoth">${i18nApi.t('statusNeedBoth')}</p>
      </div>`;
  }

  /* ════════════════════════════════════════════════════════════════
     RUN FUSION
  ════════════════════════════════════════════════════════════════ */

  /** Compute (once) and render the JS pipeline diagram for one method from the
   *  captured fusion input. The intermediate cards feed the diagram; the output
   *  card is later replaced by the faithful Python result. Returns true if the
   *  method's diagram is available. Cheap methods recompute fast; results are
   *  cached so revisiting a tab is instant. */
  function ensureMethodComputed(method) {
    if (!state.fusionInput) return false;
    if (state.pipelineResults[method]) return true;
    const { rgbData, lwirData, w, h } = state.fusionInput;
    try {
      const result = COMPUTE_FN[method](rgbData, lwirData, w, h);
      state.pipelineResults[method] = result;
      buildPipelineDiagram(method, result.stages, result.connections);
      return true;
    } catch (err) {
      console.error('[fusion] Error computing method', method, err);
      return false;
    }
  }

  function runFusion() {
    if (!cardRgb.hasImage() || !cardLwir.hasImage()) {
      setStatus(i18nApi.t('statusNeedBoth'));
      return;
    }

    if (state.generating) return;
    state.generating = true;
    dom.btnGenerate.disabled = true;
    setStatus(i18nApi.t('statusRunning'));

    Promise.resolve().then(() => {
      try {
        const bmpRgb  = cardRgb.getBitmap();
        const bmpLwir = cardLwir.getBitmap();
        const { w, h, capped, fullW, fullH } = workingSize(bmpRgb, bmpLwir, state.maxWorkSide);
        state.workCapped = capped ? { w, h, fullW, fullH } : null;

        const rgbData  = bitmapToImageData(bmpRgb,  w, h);
        const lwirData = bitmapToImageData(bmpLwir, w, h);

        // Capture input for on-demand faithful (Python) output; invalidate caches.
        // Diagrams are now computed lazily per method (only when its tab is shown)
        // instead of all methods up front — this avoids computing 4-6 pipelines
        // synchronously on the main thread and bounds retained memory to the
        // methods actually viewed (see ensureMethodComputed / setActiveMethod).
        state.fusionInput = { rgbData, lwirData, w, h };
        state.pyOutputs = {};
        state.pyPending = {};
        ALL_METHODS.forEach((m) => {
          state.pipelineResults[m] = null;
          if (dom.pipelinePanels[m]) dom.pipelinePanels[m].innerHTML = '';
        });

        const methods = state.activeStrategy === 'repr'  ? METHODS_REPR
                      : state.activeStrategy === 'alpha' ? METHODS_ALPHA
                      : METHODS_STATIC;
        const defaultMethod = state.activeStrategy === 'repr'  ? 'wavelet'
                            : state.activeStrategy === 'alpha' ? 'sobel'
                            : 'rgbt';
        // Keep the current method if it belongs to this strategy, else default.
        const activeMethod = methods.includes(state.activeMethod) ? state.activeMethod : defaultMethod;

        ensureMethodComputed(activeMethod);
        document.getElementById(`btn-method-${activeMethod}`)?.click();
        state.activeMethod = activeMethod;

        const activeResult = state.pipelineResults[activeMethod];
        if (activeResult) {
          const doneMsg = i18nApi.t('statusDone', {
            method: state.activeMethod.toUpperCase(), w, h,
          });
          let msg = doneMsg;
          if (state.workCapped) {
            const full = `${fullW}×${fullH}`;
            const capMsg = i18nApi.t('statusCapped', { full }) || `capped from ${full}`;
            msg = `${doneMsg} · ${capMsg}`;
          }
          setStatus(msg, 'success');
          dom.btnDownload.disabled = false;
        }

        // Replace the active method's output with the faithful Python result.
        requestFaithfulOutput(state.activeMethod);
      } catch (err) {
        console.error('[fusion] Error computing pipeline:', err);
        setStatus('Error computing fusion. See console for details.', 'error');
      } finally {
        state.generating = false;
        dom.btnGenerate.disabled = false;
      }
    });
  }

  /* ════════════════════════════════════════════════════════════════
     STRATEGY + TAB MANAGEMENT
  ════════════════════════════════════════════════════════════════ */

  function setActiveStrategy(strategy) {
    state.activeStrategy = strategy;
    const isRepr  = strategy === 'repr';
    const isAlpha = strategy === 'alpha';

    ALL_METHODS.forEach((m) => { dom.pipelinePanels[m]?.classList.add('hidden'); });

    document.getElementById('method-tabs')     ?.classList.toggle('hidden',  isRepr || isAlpha);
    document.getElementById('method-tabs-repr') ?.classList.toggle('hidden', !isRepr);
    document.getElementById('method-tabs-alpha')?.classList.toggle('hidden', !isAlpha);
    document.getElementById('strategy-hint')     ?.classList.toggle('hidden', isRepr || isAlpha);
    document.getElementById('strategy-hint-repr') ?.classList.toggle('hidden', !isRepr);
    document.getElementById('strategy-hint-alpha')?.classList.toggle('hidden', !isAlpha);
    document.getElementById('strategy-methods-static')?.classList.toggle('hidden', isRepr || isAlpha);
    document.getElementById('strategy-methods-repr')  ?.classList.toggle('hidden', !isRepr);
    document.getElementById('strategy-methods-alpha') ?.classList.toggle('hidden', !isAlpha);

    const methods = isRepr ? METHODS_REPR : isAlpha ? METHODS_ALPHA : METHODS_STATIC;
    const hasResults = methods.some((m) => state.pipelineResults[m] !== null);
    if (!hasResults && cardRgb?.hasImage() && cardLwir?.hasImage() && !state.generating) {
      Promise.resolve().then(runFusion);
    } else {
      const defaultId = isRepr ? 'btn-method-wavelet' : isAlpha ? 'btn-method-sobel' : 'btn-method-rgbt';
      document.getElementById(defaultId)?.click();
    }
    syncBottomTabs();
  }

  function setActiveMethod(method) {
    state.activeMethod = method;
    // Lazily compute this method's diagram the first time its tab is shown.
    if (state.fusionInput && ensureMethodComputed(method)) {
      // Switch the output to the faithful (Python) result on demand.
      requestFaithfulOutput(method);
    } else {
      showPipelineEmptyNotice(method);
    }
    syncBottomTabs();
    updateDiagramExplain(method);
    requestAnimationFrame(redrawAllConnections);
  }

  /** Contextual walkthrough shown below the active diagram: explains, stage by
   *  stage, what the boxes and intermediate cards represent for `method`
   *  (reuses the i18n methodWalk text; updated on tab switch + language change). */
  function updateDiagramExplain(method) {
    const el = document.getElementById('diagram-explain');
    if (!el) return;
    const txt = _i18nCopy && _i18nCopy.methodWalk && _i18nCopy.methodWalk[method];
    if (!txt) { el.classList.add('hidden'); el.innerHTML = ''; return; }
    el.classList.remove('hidden');
    el.innerHTML = '';
    const h = document.createElement('strong');
    h.className = 'diagram-explain-title';
    h.textContent = (_i18nCopy && _i18nCopy.diagramWalkTitle) || 'How to read this diagram';
    const p = document.createElement('p');
    p.textContent = txt;
    el.appendChild(h);
    el.appendChild(p);
    // '*' on an intermediate card / mosaic = preview contrast-enhanced for
    // visibility. Explain it once, beside the diagram, for the methods that use it
    // (wavelet/curvelet families + sobel/ssim/superpixel/fa) — it was previously
    // only in code comments / Spanish-only baked captions.
    const STAR_METHODS = new Set(['wavelet', 'waveletmax', 'curvelet', 'curveletmax',
      'sobel', 'ssim', 'superpixel', 'fa']);
    if (STAR_METHODS.has(method) && _i18nCopy && _i18nCopy.starLegend) {
      const legend = document.createElement('p');
      legend.className = 'diagram-explain-legend';
      legend.textContent = _i18nCopy.starLegend;
      el.appendChild(legend);
    }
  }

  /* ════════════════════════════════════════════════════════════════
     BOTTOM TAB BAR (multispectral-specific)
     A clone of the strategy + method tablists pinned to the viewport
     bottom while a tall pipeline is scrolled, so the user can switch
     strategy/method without scrolling back to the top. Clicks delegate
     to the real top buttons (reusing all existing logic); active state
     is mirrored by syncBottomTabs(). Shown only when the top tabs are
     scrolled out of view (IntersectionObserver guard).
  ════════════════════════════════════════════════════════════════ */
  function buildBottomTabs() {
    const methodShell = document.querySelector('.method-shell');
    if (!methodShell || document.getElementById('method-tabs-bottom')) return;
    const innerPanel    = methodShell.querySelector(':scope > .shared-tabbed-panel');
    const strategyShell = methodShell.closest('.strategy-shell');
    const outerPanel    = strategyShell && strategyShell.querySelector(':scope > .shared-tabbed-panel');
    if (!innerPanel || !strategyShell || !outerPanel) return;

    // Clone a top tablist into the BOTTOM edge of an EXISTING shell box. The
    // clone becomes a direct child of that shell, so it REUSES the shell's own
    // .shared-tabbed-shell-strip tab styling (no duplicated CSS); a CSS
    // transform flips it vertically and the labels are counter-flipped upright.
    function cloneBottom(srcId, newId) {
      const src = document.getElementById(srcId);
      if (!src) return null;
      const clone = src.cloneNode(true);
      clone.id = newId;
      clone.removeAttribute('aria-label');
      clone.classList.add('bottom-mirror', 'shared-tablist-joined', 'shared-tablist-fill');
      clone.querySelectorAll('.shared-tab').forEach((b) => {
        if (b.id) b.id = `${b.id}-bottom`;
        b.removeAttribute('aria-controls');   // not SharedTabs-managed; we delegate
        b.setAttribute('tabindex', '-1');
        b.innerHTML = `<span class="mirror-label">${b.innerHTML}</span>`;
      });
      return clone;
    }

    // Method tabs → bottom edge of the (inner) method box; strategy tabs →
    // bottom edge of the (outer) strategy box. This preserves the box hierarchy:
    // each row hangs from the very box it belongs to (inverted vs the top).
    ['method-tabs', 'method-tabs-repr', 'method-tabs-alpha'].forEach((id) => {
      const c = cloneBottom(id, `${id}-bottom`);
      if (c) methodShell.appendChild(c);              // after the inner panel
    });
    const sTabs = cloneBottom('strategy-tabs', 'strategy-tabs-bottom');
    if (sTabs) strategyShell.appendChild(sTabs);       // after the outer panel

    // Delegate clicks to the corresponding TOP button (reuses every handler),
    // but keep the CLICKED bottom tab visually fixed: record its viewport
    // position, switch, then scroll by the delta so the page grows/shrinks
    // UPWARD instead of the tab jumping down. No jumps while comparing methods.
    function anchoredSwitch(anchorEl, triggerFn) {
      const before = anchorEl.getBoundingClientRect().top;
      triggerFn();
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const dy = anchorEl.getBoundingClientRect().top - before;
        if (Math.abs(dy) > 1) window.scrollBy(0, dy);
      }));
    }
    document.querySelectorAll('.bottom-mirror .shared-tab[data-strategy]').forEach((b) => {
      b.addEventListener('click', () => anchoredSwitch(b, () =>
        document.getElementById(`btn-strategy-${b.dataset.strategy}`)?.click()));
    });
    document.querySelectorAll('.bottom-mirror .shared-tab[data-method]').forEach((b) => {
      b.addEventListener('click', () => anchoredSwitch(b, () =>
        document.getElementById(`btn-method-${b.dataset.method}`)?.click()));
    });

    // Guard: reveal the bottom edges only once the top tabs scroll above the
    // viewport. A 1px sentinel just below the top method tabs is the probe; it
    // toggles `show-bottom-tabs` on the outer shell (CSS reveals both rows).
    let sentinel = document.getElementById('top-tabs-sentinel');
    if (!sentinel) {
      sentinel = document.createElement('div');
      sentinel.id = 'top-tabs-sentinel';
      sentinel.setAttribute('aria-hidden', 'true');
      innerPanel.insertBefore(sentinel, innerPanel.firstChild);
    }
    if ('IntersectionObserver' in window) {
      const obs = new IntersectionObserver((entries) => {
        const e = entries[0];
        const off = !e.isIntersecting && e.boundingClientRect.top < 0;
        strategyShell.classList.toggle('show-bottom-tabs', off);
      }, { threshold: 0 });
      obs.observe(sentinel);
    }

    syncBottomTabs();
  }

  /** Mirror the active strategy/method + visible method-tablist onto the
   *  cloned bottom bar. No-op until buildBottomTabs() has run. */
  function syncBottomTabs() {
    if (!document.querySelector('.bottom-mirror')) return;
    const isRepr  = state.activeStrategy === 'repr';
    const isAlpha = state.activeStrategy === 'alpha';
    document.querySelectorAll('.bottom-mirror .shared-tab[data-strategy]').forEach((b) => {
      const active = b.dataset.strategy === state.activeStrategy;
      b.classList.toggle('active', active);
      b.setAttribute('aria-selected', String(active));
    });
    document.getElementById('method-tabs-bottom')      ?.classList.toggle('hidden', isRepr || isAlpha);
    document.getElementById('method-tabs-repr-bottom') ?.classList.toggle('hidden', !isRepr);
    document.getElementById('method-tabs-alpha-bottom')?.classList.toggle('hidden', !isAlpha);
    document.querySelectorAll('.bottom-mirror .shared-tab[data-method]').forEach((b) => {
      const active = b.dataset.method === state.activeMethod;
      b.classList.toggle('active', active);
      b.setAttribute('aria-selected', String(active));
    });
  }

  /** Repaint a method's output canvas (and its cached full-res ImageData, which
   *  feeds the modal, histogram and download) with a new ImageData. */
  function applyFaithfulOutput(method, imageData) {
    const cv = document.getElementById(`ps-${method}-out`);
    if (cv) {
      cv.width = imageData.width;
      cv.height = imageData.height;
      cv.getContext('2d').putImageData(imageData, 0, 0);
      state.pipelineImageData[cv.id] = imageData;
    }
    clearPyFallback(method);   // the faithful Python result is now shown
    // Refresh the output histogram if this is the active method.
    if (state.activeMethod === method) refreshOutputHistogram(method);
  }

  /** Flag that `method`'s displayed output is the JS approximation, NOT the
   *  faithful Python result (Pyodide unavailable or errored). This is a fidelity
   *  safeguard: the JS path is illustrative and may differ from the reference,
   *  so we mark the output card and warn in the status instead of swapping
   *  silently. Cleared by applyFaithfulOutput once the real result arrives. */
  function markPyFallback(method, reason) {
    if (!state.pyFailed) state.pyFailed = {};
    state.pyFailed[method] = true;
    const card = document.getElementById(`ps-${method}-out`)?.closest('.pipe-card');
    if (card) {
      card.classList.add('pipe-card--py-fallback');
      card.setAttribute('title', i18nApi.t('pyFellBack') || '');
    }
    if (state.activeMethod === method) {
      console.warn('[fusion] showing JS approximation for', method, reason || '');
      setStatus(i18nApi.t('pyFellBack') || 'Showing JS approximation (Python engine failed); result may differ from the reference.', 'warn');
    }
  }

  function clearPyFallback(method) {
    if (state.pyFailed) delete state.pyFailed[method];
    const card = document.getElementById(`ps-${method}-out`)?.closest('.pipe-card');
    if (card) { card.classList.remove('pipe-card--py-fallback'); card.removeAttribute('title'); }
  }

  /** Compute the faithful fused output for `method` via Pyodide (cached), then
   *  swap it into the diagram. On error/unavailability the JS output stays but
   *  is flagged (markPyFallback) — never swapped silently. */
  function requestFaithfulOutput(method) {
    if (!state.fusionInput) return;
    if (!window.FusionPy) { markPyFallback(method, 'Pyodide engine not loaded'); return; }
    if (state.pyOutputs && state.pyOutputs[method]) {
      applyFaithfulOutput(method, state.pyOutputs[method]);
      return;
    }
    if (state.pyPending && state.pyPending[method]) return;  // already in flight
    if (!state.pyPending) state.pyPending = {};
    state.pyPending[method] = true;

    const { rgbData, lwirData, w, h } = state.fusionInput;
    const warming = !window.FusionPy.isReady();
    if (warming) setStatus(i18nApi.t('statusPyWarming') || 'Loading Python engine…');

    window.FusionPy.computeOutput(method, rgbData, lwirData, w, h)
      .then((imageData) => {
        if (!state.pyOutputs) state.pyOutputs = {};
        state.pyOutputs[method] = imageData;
        applyFaithfulOutput(method, imageData);
        if (warming && state.activeMethod === method) {
          setStatus(i18nApi.t('statusDone', {
            method: method.toUpperCase(),
            w: imageData.width, h: imageData.height,
          }), 'success');
        }
      })
      .catch((err) => {
        console.error('[fusion] Python output failed for', method, err);
        markPyFallback(method, err);
      })
      .finally(() => {
        if (state.pyPending) delete state.pyPending[method];
      });
  }

  /** Hook for refreshing the fused-output histogram after a faithful swap.
   *  No-op if the tool has no dedicated output histogram. */
  function refreshOutputHistogram(method) {
    if (typeof renderOutputHistogram === 'function') {
      try { renderOutputHistogram(method); } catch (e) { /* ignore */ }
    }
  }

  /* ════════════════════════════════════════════════════════════════
     MODAL
  ════════════════════════════════════════════════════════════════ */

  function openModal(bitmapOrCanvas) {
    const modal  = dom.modalOverlay;
    const canvas = dom.modalCanvas;
    const ctx    = canvas.getContext('2d');
    canvas.width  = bitmapOrCanvas.width;
    canvas.height = bitmapOrCanvas.height;
    ctx.drawImage(bitmapOrCanvas, 0, 0);
    state.modalSrcId = null;   // openModalFromCanvas sets the real id afterwards
    modal.classList.remove('hidden');
    modal.setAttribute('aria-hidden', 'false');
    dom.btnCloseModal.focus();
  }

  function openModalFromCanvas(srcCanvas) {
    if (!srcCanvas || srcCanvas.width === 0) return;
    const imageData = state.pipelineImageData[srcCanvas.id];
    if (imageData) {
      const oc = new OffscreenCanvas(imageData.width, imageData.height);
      oc.getContext('2d').putImageData(imageData, 0, 0);
      openModal(oc);
    } else {
      openModal(srcCanvas);
    }
    state.modalSrcId = srcCanvas.id;   // for the modal download's filename
  }

  /** Download the image currently shown in the expand modal (full-res). */
  function downloadModalImage() {
    if (!dom.modalCanvas || !dom.modalCanvas.width) return;
    const id = state.modalSrcId;
    const name = id
      ? `fusion_${id.replace(/^ps-/, '').replace(/[^a-z0-9]+/gi, '_').replace(/^_|_$/g, '')}.png`
      : 'image.png';
    downloadCanvas(dom.modalCanvas, name);
  }

  function closeModal() {
    dom.modalOverlay.classList.add('hidden');
    dom.modalOverlay.setAttribute('aria-hidden', 'true');
  }

  function closeHistModal() {
    clearHistModalChart();
    dom.histModalOverlay.classList.add('hidden');
    dom.histModalOverlay.setAttribute('aria-hidden', 'true');
  }

  /* ════════════════════════════════════════════════════════════════
     DOWNLOAD
  ════════════════════════════════════════════════════════════════ */

  function downloadCanvas(canvas, filename) {
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = filename || 'fusion.png';
    a.click();
  }

  function downloadActiveOutput() {
    const result = state.pipelineResults[state.activeMethod];
    if (!result) return;
    const canvas = document.getElementById(`ps-${state.activeMethod}-out`);
    if (canvas) {
      downloadCanvas(canvas, `fusion_${state.activeMethod}.png`);
    }
  }

  /* ════════════════════════════════════════════════════════════════
     STATUS
  ════════════════════════════════════════════════════════════════ */

  function setStatus(msg, type) {
    if (!dom.status) return;
    dom.status.textContent = msg;
    dom.status.classList.remove('status--error', 'status--success', 'status--warn');
    if (type === 'error')   dom.status.classList.add('status--error');
    if (type === 'success') dom.status.classList.add('status--success');
    if (type === 'warn')    dom.status.classList.add('status--warn');
  }

  function updateStatus() {
    if (!cardRgb.hasImage() || !cardLwir.hasImage()) {
      setStatus(i18nApi.t('statusReady'));
      dom.btnDownload.disabled = true;
    }
  }

  /* ════════════════════════════════════════════════════════════════
     I18N APPLY
  ════════════════════════════════════════════════════════════════ */

  /** Render a bullet list of each strategy's methods (name + description) under
   *  its intro hint, REUSING the i18n methodDesc strings (same text as the tab
   *  tooltips). Re-run on language change; lists track their hint's visibility. */
  function populateStrategyMethodLists(t) {
    const groups = [
      { listId: 'strategy-methods-static', hintId: 'strategy-hint',       methods: METHODS_STATIC },
      { listId: 'strategy-methods-repr',   hintId: 'strategy-hint-repr',  methods: METHODS_REPR },
      { listId: 'strategy-methods-alpha',  hintId: 'strategy-hint-alpha', methods: METHODS_ALPHA },
    ];
    groups.forEach(({ listId, hintId, methods }) => {
      const hint = document.getElementById(hintId);
      if (!hint) return;
      let ul = document.getElementById(listId);
      if (!ul) {
        ul = document.createElement('ul');
        ul.id = listId;
        ul.className = 'strategy-methods-list hint';
        hint.insertAdjacentElement('afterend', ul);
      }
      ul.classList.toggle('hidden', hint.classList.contains('hidden'));
      ul.innerHTML = '';
      methods.forEach((m) => {
        const desc = t.methodDesc && t.methodDesc[m];
        if (!desc) return;
        const li = document.createElement('li');
        const dash = desc.indexOf('—');     // methodDesc is "Name — explanation"
        if (dash > 0) {
          const strong = document.createElement('strong');
          strong.textContent = desc.slice(0, dash).trim();
          li.appendChild(strong);
          li.appendChild(document.createTextNode(' — ' + desc.slice(dash + 1).trim()));
        } else {
          li.textContent = desc;
        }
        ul.appendChild(li);
      });
    });
  }

  function applyTranslations(lang) {
    state.lang = lang;
    const t = i18nApi.getCopy(lang);
    _i18nCopy = t;
    if (!t) return;

    const set = (id, key) => {
      const el = document.getElementById(id);
      if (el && t[key]) el.textContent = t[key];
    };

    set('subtitle',          'subtitle');
    set('intro-title',       'introTitle');
    set('intro-text',        'introText');
    set('cfg-title',         'cfgTitle');
    set('lbl-examples',      'example');
    set('lbl-resolution',    'resolution');
    const _resOpts = { '512': 'resLight', '768': 'resBalanced', '1024': 'resHigh', '0': 'resFull' };
    Object.keys(_resOpts).forEach((val) => {
      const opt = document.querySelector(`#select-resolution option[value="${val}"]`);
      if (opt && t[_resOpts[val]]) opt.textContent = t[_resOpts[val]];
    });
    set('badge-rgb',         'badgeVisible');
    set('badge-lwir',        'badgeThermal');
    set('src-rgb-title',     'srcRgbTitle');
    set('src-lwir-title',    'srcLwirTitle');
    set('drop-hint-rgb',     'dropHint');
    set('drop-hint-lwir',    'dropHint');
    set('btn-generate',      'runFusion');
    set('btn-download-active','downloadOutput');
    set('strategy-hint',           'strategyHint');
    set('strategy-hint-repr',      'strategyReprHint');
    set('strategy-hint-alpha',     'strategyAlphaHint');
    set('btn-strategy-static-early', 'strategyStaticEarlyLabel');
    set('btn-strategy-repr',       'strategyReprLabel');
    set('btn-strategy-alpha',      'strategyAlphaLabel');
    set('lbl-equalize-rgb',        'equalize');
    set('lbl-equalize-lwir',       'equalize');
    set('lbl-hist-log',            'logScale');
    set('footer-author',           'footerAuthor');

    ALL_METHODS.forEach((m) => {
      const btn = document.getElementById(`btn-method-${m}`);
      if (btn && t.methodDesc?.[m]) btn.title = t.methodDesc[m];
    });

    populateStrategyMethodLists(t);
    updateDiagramExplain(state.activeMethod);

    document.querySelectorAll('[data-i18n-stage]').forEach((el) => {
      const key = el.dataset.i18nStage;
      if (t.stageLabel?.[key]) el.textContent = t.stageLabel[key];
    });

    const customOpt = document.querySelector('#select-example option[value="custom"]');
    if (customOpt && t.custom) customOpt.textContent = t.custom;

    set('image-modal-title', 'preview');
    if (t.close) {
      ['btn-close-image-modal', 'btn-close-hist-modal'].forEach((id) => {
        const btn = document.getElementById(id);
        if (btn) { btn.setAttribute('aria-label', t.close); btn.setAttribute('title', t.close); }
      });
    }

    updateStatus();
  }

  /* ════════════════════════════════════════════════════════════════
     INIT
  ════════════════════════════════════════════════════════════════ */

  function init() {
    initImageCards();

    ALL_METHODS.forEach((m) => showPipelineEmptyNotice(m));

    // Method tabs – SharedTabs handles button state + panel visibility via aria-controls
    window.SharedTabs.bind(document.getElementById('method-tabs'), {
      onSelect: (tabEl) => setActiveMethod(tabEl.dataset.method),
    });
    window.SharedTabs.bind(document.getElementById('method-tabs-repr'), {
      onSelect: (tabEl) => setActiveMethod(tabEl.dataset.method),
    });
    window.SharedTabs.bind(document.getElementById('method-tabs-alpha'), {
      onSelect: (tabEl) => setActiveMethod(tabEl.dataset.method),
    });

    // Strategy tabs – manual binding (panels are not separate DOM elements)
    document.querySelectorAll('#strategy-tabs .shared-tab').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#strategy-tabs .shared-tab').forEach((b) => {
          const active = b === btn;
          b.classList.toggle('active', active);
          b.setAttribute('aria-selected', String(active));
        });
        setActiveStrategy(btn.dataset.strategy);
      });
    });

    // Contextual diagram walkthrough: a panel BESIDE the diagram (right column,
    // under the LWIR histogram, sticky) that explains, stage by stage, what the
    // active method's boxes/cards mean. Sits in the empty space alongside the
    // tall pipeline instead of buried below it.
    const fdw = document.querySelector('.fusion-diagram-wrapper');
    if (fdw && !document.getElementById('diagram-explain')) {
      const ex = document.createElement('div');
      ex.id = 'diagram-explain';
      ex.className = 'diagram-explain hidden';
      const histLwir = document.getElementById('hist-lwir-card');
      if (histLwir && histLwir.parentNode === fdw) {
        const aside = document.createElement('div');
        aside.className = 'diagram-aside';
        fdw.insertBefore(aside, histLwir);
        aside.appendChild(histLwir);   // histogram stays on top of the column
        aside.appendChild(ex);         // explanation below it, beside the pipeline
      } else {
        fdw.insertAdjacentElement('afterend', ex);   // fallback: below the diagram
      }
    }

    // Bottom tab bar: clone of the strategy+method tabs, pinned to the viewport
    // bottom while a tall pipeline is scrolled (shown only when the top tabs are
    // off-screen). Built after the top tabs are wired so the clones reuse them.
    buildBottomTabs();

    // Example selector (combobox)
    dom.selectExample?.addEventListener('change', () => {
      const val = dom.selectExample.value;
      if (val === 'custom') {
        applyMode(true);
        state.activeExample = null;
        if (state.customRgbBitmap) {
          cardRgb.setImage(state.customRgbBitmap, state.customRgbLabel);
          state.rawRgb = state.customRgbBitmap;
          state.rawRgbLabel = state.customRgbLabel;
          updateCardEqualization('rgb');
        } else {
          state.rawRgb = null; state.rawRgbLabel = '';
          cardRgb.clear();
        }
        if (state.customLwirBitmap) {
          cardLwir.setImage(state.customLwirBitmap, state.customLwirLabel);
          state.rawLwir = state.customLwirBitmap;
          state.rawLwirLabel = state.customLwirLabel;
          updateCardEqualization('lwir');
        } else {
          state.rawLwir = null; state.rawLwirLabel = '';
          cardLwir.clear();
        }
        if (state.customRgbBitmap || state.customLwirBitmap) onImageChange();
      } else {
        loadExample(Number(val));
      }
    });

    // Resolution cap: friendly default (512) bounds Pyodide/WASM + JS-compute memory;
    // the user can raise it if they have RAM. If a fusion already ran, re-apply now.
    dom.selectResolution?.addEventListener('change', () => {
      const v = parseInt(dom.selectResolution.value, 10);
      state.maxWorkSide = Number.isFinite(v) ? v : 0;
      if (state.fusionInput && !state.generating) dom.btnGenerate?.click();
    });

    // Per-image equalization toggles
    dom.chkEqualizeRgb?.addEventListener('change',  () => setEqualization('rgb',  dom.chkEqualizeRgb.checked));
    dom.chkEqualizeLwir?.addEventListener('change', () => setEqualization('lwir', dom.chkEqualizeLwir.checked));

    // Histogram expand buttons
    dom.btnExpandHistRgb?.addEventListener('click',  () => openHistogramModal('rgb'));
    dom.btnExpandHistLwir?.addEventListener('click', () => openHistogramModal('lwir'));

    // Histogram log/linear scale toggle (shared by thumbnails + modal)
    if (dom.histLogScale) dom.histLogScale.checked = state.histLog;  // reflect default
    dom.histLogScale?.addEventListener('change', function () {
      state.histLog = this.checked;
      const disp = (type) => {
        const raw = type === 'rgb' ? state.rawRgb : state.rawLwir;
        if (!raw) return null;
        const eq = type === 'rgb' ? state.equalizeRgb : state.equalizeLwir;
        return eq ? equalizeToOffscreenCanvas(raw, type) : raw;
      };
      updateHistogramChart('rgb', disp('rgb'));
      updateHistogramChart('lwir', disp('lwir'));
      if (state.histModalType && dom.histModalOverlay && !dom.histModalOverlay.classList.contains('hidden')) {
        openHistogramModal(state.histModalType);
      }
    });

    // Inline histogram thumbnails are hand-drawn on demand (updateHistogramChart);
    // here we just wire click-to-expand.
    [['rgb', dom.histRgb], ['lwir', dom.histLwir]].forEach(([t, cv]) => {
      if (!cv) return;
      cv.style.cursor = 'zoom-in';
      cv.addEventListener('click', () => openHistogramModal(t));
    });

    // Generate / Download buttons
    dom.btnGenerate?.addEventListener('click', runFusion);
    dom.btnDownload?.addEventListener('click', downloadActiveOutput);

    // Image modal close
    dom.btnCloseModal?.addEventListener('click', closeModal);
    dom.btnDownloadModal?.addEventListener('click', downloadModalImage);
    dom.modalOverlay?.addEventListener('click', (e) => {
      // Close on a click anywhere that is NOT the image itself or a control
      // button — so the dialog padding (the large non-image area inside the
      // lightbox stage) closes too, not only the dark overlay outside the box.
      if (e.target.closest('#image-modal-canvas') || e.target.closest('.image-modal-controls')) return;
      closeModal();
    });

    // Histogram modal close
    dom.btnCloseHistModal?.addEventListener('click', closeHistModal);
    dom.histModalOverlay?.addEventListener('click', (e) => {
      if (e.target === dom.histModalOverlay) closeHistModal();
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { closeModal(); closeHistModal(); }
    });

    // Integrate with page-shell-core (handles theme toggle, lang switcher,
    // and removes the `i18n-pending` class so the page becomes visible).
    window.SharedToolPageShell.initToolPage({
      i18nApi,
      onApplyLanguage: (_copy, lang) => {
        applyTranslations(lang);
      },
      relatedWork: {
        toolId:    'multispectral-fusion-tool',
        sourceUrl: window.FUSION_RELATED_WORK_URL || '/assets/shared/related-work.json',
      },
    });

    setActiveMethod('rgbt');
    updateStatus();

    requestAnimationFrame(drawInputConnections);
    window.addEventListener('resize', () => {
      requestAnimationFrame(drawInputConnections);
      requestAnimationFrame(redrawAllConnections);
    });

    // Preload LLVIP example by default
    loadExample(1);

    // Warm up the Python (Pyodide) fusion engine in the background while the
    // user reads, so the first faithful output is ready quickly. Deferred to
    // idle time to avoid competing with initial page render.
    if (window.FusionPy) {
      const warm = () => window.FusionPy.ready().catch((e) => {
        console.warn('[fusion] Pyodide preload failed (will retry on demand):', e);
      });
      if ('requestIdleCallback' in window) requestIdleCallback(warm, { timeout: 3000 });
      else setTimeout(warm, 1200);
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
