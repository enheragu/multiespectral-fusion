# MultiespectralFusion

MultiespectralFusion is a static web application that demonstrates multispectral image fusion between visible (RGB) and long-wave infrared (LWIR) thermal images, with a step-by-step visualization of each method's pipeline.

It lets you:

- load a visible and a thermal image (or use the bundled LLVIP / KAIST examples),
- explore three fusion strategies: static early fusion, early reprojection, and adaptive α-fusion,
- compare 13 fusion methods (RGBT, HSVT, VTHS, VT, wavelet avg/max, curvelet avg/max, PCA, FA, sobel-weighted, SSIM, superpixel),
- follow the step-by-step pipeline diagram of each method,
- get faithful outputs computed with the original Python reference, run in the browser via [Pyodide](https://pyodide.org/),
- inspect per-channel histograms and download the fused result,
- choose the working resolution to keep memory in check on any machine.

Web version:

- https://enheragu.github.io/multiespectral-fusion/

Built on the shared [tool-pages-theme](https://github.com/enheragu/tool-pages-theme).

---

Author: [Enrique Heredia-Aguado](https://enheragu.github.io/)
