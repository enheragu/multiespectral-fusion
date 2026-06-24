// js/app-pyfusion.js — faithful fusion via the real Python reference in Pyodide.
// Uses the shared PyodideRuntime (tool-pages-theme/assets/js/core/pyodide-runtime.js).
// Runs the actual Dataset.fusion_methods code (vendored under py/) so outputs match
// the original pipeline (skimage SSIM/SLIC, sklearn PCA/FA, pywt, curvelets, cv2).
//
// Exposes window.FusionPy:
//   ready()    -> Promise (resolves when Pyodide + packages + code are loaded)
//   isReady()  -> bool
//   computeOutput(method, rgbData, lwirData, w, h) -> Promise<ImageData>
(function () {
  'use strict';

  var PKG_BASE = 'py/';                 // relative to the page URL (jekyll serves it)
  var FUSION_FILES = [
    'fusion_methods/normalization.py',
    'fusion_methods/static_image_compression.py',
    'fusion_methods/local_filter_fusion.py',
    'fusion_methods/wavelets_mdmr_compression.py',
    'fusion_methods/pca_fa_compression.py',
    'th_equalization.py',
  ];

  // Minimal stubs replacing the original repo's helper modules (logging / IO
  // decorators) which the vendored fusion files import but don't need here.
  var STUB_UTILS = [
    'class _B:',
    '    pass',
    'bcolors = _B()',
    "for _n in ('OKGREEN','WARNING','FAIL','ENDC','OKBLUE','BOLD','HEADER','OKCYAN','UNDERLINE'):",
    "    setattr(bcolors, _n, '')",
    'def log(*a, **k):',
    '    pass',
  ].join('\n');

  var STUB_DECORATORS = [
    'def time_execution_measure(f):',
    '    return f',
    'def save_image_if_path(f):',
    '    return f',
    'def save_npmat_if_path(f):',
    '    return f',
  ].join('\n');

  var _runtime = null;
  var _ready = null;

  function _resolveBase() {
    // resolve PKG_BASE against the document URL so fetch works under /multiespectral_fusion_web/
    return new URL(PKG_BASE, document.baseURI).href;
  }

  function _fetchText(url) {
    return fetch(url, { cache: 'force-cache' }).then(function (r) {
      if (!r.ok) throw new Error('fetch ' + url + ' -> ' + r.status);
      return r.text();
    });
  }

  function _buildRuntime() {
    if (_runtime) return _runtime;
    var base = _resolveBase();

    // Fetch the entry module + vendored fusion files, assemble the Pyodide FS map.
    var ready = Promise.all([
      _fetchText(base + 'worker_entry.py'),
      Promise.all(FUSION_FILES.map(function (f) { return _fetchText(base + f); })),
    ]).then(function (res) {
      var entry = res[0];
      var texts = res[1];
      var files = {
        'utils.py': STUB_UTILS,
        'Dataset/__init__.py': '',
        'Dataset/decorators.py': STUB_DECORATORS,
        'Dataset/fusion_methods/__init__.py': '',
      };
      FUSION_FILES.forEach(function (rel, i) {
        // normalization/static/... live under Dataset/fusion_methods/;
        // th_equalization.py lives directly under Dataset/.
        var dest = rel.indexOf('fusion_methods/') === 0
          ? 'Dataset/' + rel
          : 'Dataset/' + rel;
        files[dest] = texts[i];
      });

      _runtime = window.PyodideRuntime.create({
        id: 'multispectral-fusion',
        version: '0.27.0',
        packages: ['numpy', 'scipy', 'scikit-image', 'scikit-learn', 'pywavelets'],
        // opencv is needed by every method (the fusion modules `import cv2` at
        // load time), so it must be present before pythonCode runs. curvelets is
        // installed lazily (see CURVELET_METHODS below) because only 2 methods
        // use it and it must never block the other 11 if it fails to install.
        install: ['opencv-python'],
        files: files,
        pythonCode: entry,
        prewarm: false,   // we call ready() explicitly below
      });
      return _runtime.ready();
    });

    return ready;
  }

  function ready() {
    if (!_ready) _ready = _buildRuntime();
    return _ready;
  }

  function isReady() {
    return !!_runtime && !!_ready;
  }

  // Methods that need the (optional) curvelets package. Installed on first use so
  // a failure here can never block the other 11 methods.
  var CURVELET_METHODS = { curvelet: 1, curveletmax: 1 };
  var CURVELET_PKG = 'curvelets==0.0.6a0';   // exposes SimpleUDCT(shape,nscales,nbands_per_direction,alpha,winthresh); 0.1+/1.x renamed it to UDCT

  function _ensureMethodDeps(method) {
    if (CURVELET_METHODS[method]) return _runtime.install([CURVELET_PKG]);
    return Promise.resolve();
  }

  // rgbData / lwirData are ImageData (RGBA). Returns a Promise<ImageData> of the
  // faithful fused output at the same w*h.
  function computeOutput(method, rgbData, lwirData, w, h) {
    return ready().then(function () {
      return _ensureMethodDeps(method);
    }).then(function () {
      // copy buffers so we can transfer without detaching the caller's ImageData
      var rgb = new Uint8Array(rgbData.data.length);
      rgb.set(rgbData.data);
      var lwir = new Uint8Array(lwirData.data.length);
      lwir.set(lwirData.data);
      var payload = { method: method, w: w, h: h, rgb: rgb, lwir: lwir };
      return _runtime.call('fuse', payload, { transfer: [rgb.buffer, lwir.buffer] });
    }).then(function (res) {
      var arr = res.rgba instanceof Uint8Array ? res.rgba : new Uint8Array(res.rgba);
      return new ImageData(new Uint8ClampedArray(arr.buffer, 0, arr.length), res.w, res.h);
    });
  }

  window.FusionPy = { ready: ready, isReady: isReady, computeOutput: computeOutput };
})();
