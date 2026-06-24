# _worker_entry.py — Pyodide entry point for multispectral fusion.
# Runs the *real* reference Dataset.fusion_methods code (vendored under py/).
# Defines fuse(payload) which the JS runtime calls via _dispatch.
#
# Boundary: the web sends canvas ImageData (RGBA, row-major). OpenCV uses BGR.
#   visible RGBA -> BGR ndarray (drop alpha, swap R/B)
#   thermal RGBA -> single-channel grayscale (take R)
# Output: reference combine_* return BGR uint8 -> convert back to RGBA for canvas.

import os
import sys
# Vendored modules are written to the Pyodide FS relative to the CWD; make sure
# that directory is importable.
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '/home/pyodide')

# IMPORTANT: import skimage submodules BEFORE the fusion modules. A skimage
# lazy-loading quirk otherwise makes ssim_v2 and superpixel collapse to the
# same wrong output. Verified against hand-inlined algorithms.
import numpy as np
from skimage.metrics import structural_similarity as _warm_ssim   # noqa: F401
from skimage.segmentation import slic as _warm_slic               # noqa: F401
from skimage import filters as _warm_filters                      # noqa: F401
from skimage.util import img_as_float as _warm_iaf                 # noqa: F401

from Dataset.fusion_methods import static_image_compression as _S
from Dataset.fusion_methods import local_filter_fusion as _L
from Dataset.fusion_methods import wavelets_mdmr_compression as _W
from Dataset.fusion_methods import pca_fa_compression as _P


def _first(x):
    return x[0] if isinstance(x, tuple) else x


# web method name -> callable(vis_bgr, th_gray) -> fused image (BGR uint8 or float)
_METHODS = {
    'rgbt':        lambda v, t: _S.combine_rgbt_v2(v, t),
    'hsvt':        lambda v, t: _S.combine_hsvt(v, t),
    'vths':        lambda v, t: _S.combine_vths_v2(v, t),
    'vt':          lambda v, t: _S.combine_vt(v, t),
    'wavelet':     lambda v, t: _first(_W.combine_rgbt_wavelet(v, t)),
    'waveletmax':  lambda v, t: _W.combine_rgbt_wavelet_max(v, t),
    'curvelet':    lambda v, t: _first(_W.combine_rgbt_curvelet(v, t)),
    'curveletmax': lambda v, t: _W.combine_rgbt_curvelet_max(v, t),
    'pca':         lambda v, t: _P.combine_rgbt_pca_to3ch(v, t),
    'fa':          lambda v, t: _P.combine_rgbt_fa_to3ch(v, t),
    'sobel':       lambda v, t: _L.combine_rgbt_sobel_weighted(v, t),
    'ssim':        lambda v, t: _L.combine_rgbt_ssim_v2(v, t),
    'superpixel':  lambda v, t: _L.combine_rgbt_superpixel(v, t),
}


def fuse(payload):
    p = payload.to_py() if hasattr(payload, 'to_py') else payload
    w = int(p['w'])
    h = int(p['h'])
    method = p['method']

    rgba = np.asarray(p['rgb'], dtype=np.uint8).reshape(h, w, 4)
    vis_bgr = rgba[:, :, [2, 1, 0]].copy()              # R,G,B -> B,G,R
    th = np.asarray(p['lwir'], dtype=np.uint8).reshape(h, w, 4)[:, :, 0].copy()

    fn = _METHODS.get(method)
    if fn is None:
        raise ValueError('unknown method: ' + str(method))

    out = np.asarray(fn(vis_bgr, th))
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim == 2:                                   # single-channel -> replicate
        out = np.dstack([out, out, out])

    out_rgba = np.empty((h, w, 4), dtype=np.uint8)
    out_rgba[:, :, 0] = out[:, :, 2]                    # B -> R
    out_rgba[:, :, 1] = out[:, :, 1]                    # G
    out_rgba[:, :, 2] = out[:, :, 0]                    # R -> B
    out_rgba[:, :, 3] = 255
    result = {'rgba': out_rgba.tobytes(), 'w': w, 'h': h}
    # Free the large working arrays and force a GC pass NOW. The WASM heap only ever
    # grows (it never shrinks back to the OS), so we must hand the space back to the
    # Python allocator before the next fuse() runs — otherwise each method switch
    # ratchets the heap up to a new high-water mark and the tab balloons.
    del rgba, vis_bgr, th, out, out_rgba
    import gc
    gc.collect()
    return result
