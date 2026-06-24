// js/app-math.js — Pure image-processing and math utilities for MultiespectralFusion.
// No DOM access, no state. All exports assigned to window.FusionMath.
(function () {
  'use strict';

  /* ════════════════════════════════════════════════════════════════
     COLOUR MATH
  ════════════════════════════════════════════════════════════════ */

  /** Convert RGB (0-255 each) to HSV (h∈[0,360), s∈[0,1], v∈[0,1]). */
  function rgbToHsv(r, g, b) {
    const rn = r / 255, gn = g / 255, bn = b / 255;
    const max = Math.max(rn, gn, bn);
    const min = Math.min(rn, gn, bn);
    const d = max - min;
    let h = 0;
    if (d !== 0) {
      if (max === rn)      h = ((gn - bn) / d) % 6;
      else if (max === gn) h = (bn - rn) / d + 2;
      else                 h = (rn - gn) / d + 4;
      h = ((h * 60) + 360) % 360;
    }
    return { h, s: max === 0 ? 0 : d / max, v: max };
  }

  /** Convert HSV (h∈[0,360), s∈[0,1], v∈[0,1]) to RGB (0-255 each). */
  function hsvToRgb(h, s, v) {
    const c = v * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = v - c;
    let r1 = 0, g1 = 0, b1 = 0;
    if (h < 60)       { r1 = c; g1 = x; }
    else if (h < 120) { r1 = x; g1 = c; }
    else if (h < 180) { g1 = c; b1 = x; }
    else if (h < 240) { g1 = x; b1 = c; }
    else if (h < 300) { r1 = x; b1 = c; }
    else              { r1 = c; b1 = x; }
    return [
      Math.round((r1 + m) * 255),
      Math.round((g1 + m) * 255),
      Math.round((b1 + m) * 255),
    ];
  }

  /* ════════════════════════════════════════════════════════════════
     OFFSCREEN CANVAS HELPERS
  ════════════════════════════════════════════════════════════════ */

  /**
   * Render a bitmap into an offscreen canvas at a given size and return
   * the ImageData.
   */
  function bitmapToImageData(bitmap, w, h) {
    const oc = new OffscreenCanvas(w, h);
    const ctx = oc.getContext('2d');
    ctx.drawImage(bitmap, 0, 0, w, h);
    return ctx.getImageData(0, 0, w, h);
  }

  // Cap the longer working side. Fusion runs the full scientific Python stack in
  // Pyodide/WASM (plus heavy JS transforms); memory and time scale with pixel
  // count, so an uncapped full-res image (e.g. 1280×1024) makes curvelet alone peak
  // ~5 GB and the windowed SSIM crawl. A reduced default keeps the tool friendly on
  // any machine; the user can raise it from the Resolution selector if they have
  // RAM to spare. NOTE: this cap was silently lost in a refactor — its absence is
  // what made every method run at full resolution and the machine run out of RAM.
  const MAX_WORK_SIDE = 512;

  /**
   * Choose the working resolution: minimum of both bitmaps (never up-scale), then
   * downscale proportionally if the longer side exceeds the cap. `maxSide` overrides
   * the default cap (0/falsy = no cap, i.e. full resolution). Returns the capped
   * {w,h} plus {capped, fullW, fullH} so the caller can report "capped from".
   */
  function workingSize(bmpA, bmpB, maxSide) {
    const cap = maxSide === undefined ? MAX_WORK_SIDE : maxSide;
    const fullW = Math.min(bmpA.width, bmpB.width);
    const fullH = Math.min(bmpA.height, bmpB.height);
    const longSide = Math.max(fullW, fullH);
    if (!cap || longSide <= cap) return { w: fullW, h: fullH, capped: false, fullW, fullH };
    const s = cap / longSide;
    const w = Math.max(1, Math.round(fullW * s));
    const h = Math.max(1, Math.round(fullH * s));
    return { w, h, capped: true, fullW, fullH };
  }

  /** Build a greyscale ImageData from a single-channel Uint8Array. */
  function greyImageData(channel, w, h) {
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      const v = channel[i];
      data[i * 4]     = v;
      data[i * 4 + 1] = v;
      data[i * 4 + 2] = v;
      data[i * 4 + 3] = 255;
    }
    return new ImageData(data, w, h);
  }

  /** Build a false-colour ImageData from three channel Uint8Arrays. */
  function falseColourImageData(chR, chG, chB, w, h) {
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      data[i * 4]     = chR[i];
      data[i * 4 + 1] = chG[i];
      data[i * 4 + 2] = chB[i];
      data[i * 4 + 3] = 255;
    }
    return new ImageData(data, w, h);
  }

  /* ════════════════════════════════════════════════════════════════
     CLAHE EQUALIZATION
     Matches Python reference: th_equalization (clipLimit=6.0, tileGridSize=(6,6))
     and rgb_equalization (CLAHE on Y channel of YCbCr).
  ════════════════════════════════════════════════════════════════ */

  /**
   * Apply CLAHE to a single grayscale channel (Uint8Array).
   * Matches cv.createCLAHE(clipLimit=6.0, tileGridSize=(6,6)).apply()
   */
  function applyCLAHE(gray, width, height, clipLimit, tileRows, tileCols) {
    const BINS = 256;
    const tileH = Math.ceil(height / tileRows);
    const tileW = Math.ceil(width  / tileCols);
    const luts  = new Array(tileRows * tileCols);

    for (let tr = 0; tr < tileRows; tr++) {
      for (let tc = 0; tc < tileCols; tc++) {
        const y0 = tr * tileH, y1 = Math.min(y0 + tileH, height);
        const x0 = tc * tileW, x1 = Math.min(x0 + tileW, width);
        const area = (y1 - y0) * (x1 - x0);
        const clipThr = Math.max(1, Math.round(clipLimit * area / BINS));

        const hist = new Int32Array(BINS);
        for (let y = y0; y < y1; y++)
          for (let x = x0; x < x1; x++)
            hist[gray[y * width + x]]++;

        let excess = 0;
        for (let i = 0; i < BINS; i++) {
          if (hist[i] > clipThr) { excess += hist[i] - clipThr; hist[i] = clipThr; }
        }
        const perBin = Math.floor(excess / BINS);
        let rem = excess % BINS;
        for (let i = 0; i < BINS; i++) { hist[i] += perBin; if (rem-- > 0) hist[i]++; }

        const lut = new Uint8Array(BINS);
        let cdf = 0;
        const scale = 255 / area;
        for (let i = 0; i < BINS; i++) { cdf += hist[i]; lut[i] = Math.min(255, Math.round(cdf * scale)); }
        luts[tr * tileCols + tc] = lut;
      }
    }

    const out = new Uint8Array(gray.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const val = gray[y * width + x];
        const ty = (y + 0.5) / tileH - 0.5;
        const tx = (x + 0.5) / tileW - 0.5;
        const tr0 = Math.max(0, Math.min(tileRows - 1, Math.floor(ty)));
        const tc0 = Math.max(0, Math.min(tileCols - 1, Math.floor(tx)));
        const tr1 = Math.min(tileRows - 1, tr0 + 1);
        const tc1 = Math.min(tileCols - 1, tc0 + 1);
        const wr = Math.max(0, Math.min(1, ty - tr0));
        const wc = Math.max(0, Math.min(1, tx - tc0));
        out[y * width + x] = Math.round(
          (1 - wr) * ((1 - wc) * luts[tr0 * tileCols + tc0][val] + wc * luts[tr0 * tileCols + tc1][val]) +
          wr       * ((1 - wc) * luts[tr1 * tileCols + tc0][val] + wc * luts[tr1 * tileCols + tc1][val])
        );
      }
    }
    return out;
  }

  /**
   * Equalize an RGBA ImageData for a visible RGB image.
   * Converts to YCbCr, applies CLAHE to Y, converts back to RGB.
   */
  function equalizeRgbImageData(imgData) {
    const { data, width, height } = imgData;
    const n = width * height;
    const yArr = new Uint8Array(n), cbArr = new Uint8Array(n), crArr = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
      yArr[i]  = Math.max(0, Math.min(255, Math.round( 0.299 * r + 0.587 * g + 0.114 * b)));
      cbArr[i] = Math.max(0, Math.min(255, Math.round(-0.169 * r - 0.331 * g + 0.500 * b + 128)));
      crArr[i] = Math.max(0, Math.min(255, Math.round( 0.500 * r - 0.419 * g - 0.081 * b + 128)));
    }
    const yEq = applyCLAHE(yArr, width, height, 6.0, 6, 6);
    const out = new Uint8ClampedArray(data.length);
    for (let i = 0; i < n; i++) {
      const y = yEq[i], cb = cbArr[i] - 128, cr = crArr[i] - 128;
      out[i * 4]     = Math.max(0, Math.min(255, Math.round(y + 1.403 * cr)));
      out[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(y - 0.344 * cb - 0.714 * cr)));
      out[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(y + 1.773 * cb)));
      out[i * 4 + 3] = data[i * 4 + 3];
    }
    return new ImageData(out, width, height);
  }

  /**
   * Equalize an RGBA ImageData for a grayscale LWIR thermal image.
   * Applies CLAHE directly to the grayscale channel.
   */
  function equalizeLwirImageData(imgData) {
    const { data, width, height } = imgData;
    const n = width * height;
    const gray = new Uint8Array(n);
    for (let i = 0; i < n; i++) gray[i] = data[i * 4];
    const eq = applyCLAHE(gray, width, height, 6.0, 6, 6);
    const out = new Uint8ClampedArray(data.length);
    for (let i = 0; i < n; i++) {
      out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = eq[i];
      out[i * 4 + 3] = data[i * 4 + 3];
    }
    return new ImageData(out, width, height);
  }

  /**
   * Equalize a bitmap and return an OffscreenCanvas with the result.
   * The returned OffscreenCanvas is drawable (accepted by ctx.drawImage and SharedImageCard.setImage).
   */
  function equalizeToOffscreenCanvas(bitmap, type) {
    const oc  = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = oc.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    const eqData = type === 'rgb'
      ? equalizeRgbImageData(ctx.getImageData(0, 0, bitmap.width, bitmap.height))
      : equalizeLwirImageData(ctx.getImageData(0, 0, bitmap.width, bitmap.height));
    ctx.putImageData(eqData, 0, 0);
    return oc;
  }

  /* ════════════════════════════════════════════════════════════════
     REPROJECTION MATH HELPERS
  ════════════════════════════════════════════════════════════════ */

  /** Scale any typed array to [0, 255] Uint8Array. */
  function normalizeToUint8(arr) {
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] < mn) mn = arr[i];
      if (arr[i] > mx) mx = arr[i];
    }
    const out = new Uint8Array(arr.length);
    const range = mx - mn;
    if (range < 1e-9) return out;
    for (let i = 0; i < arr.length; i++) {
      out[i] = Math.round((arr[i] - mn) / range * 255);
    }
    return out;
  }

  /**
   * 1-level 2D Haar DWT (unnormalized).
   * cA = (TL+TR+BL+BR)/4, cH = (TL-TR+BL-BR)/4
   * cV = (TL+TR-BL-BR)/4, cD = (TL-TR-BL+BR)/4
   */
  function haarDWT2(ch, w, h) {
    const hw = Math.floor(w / 2);
    const hh = Math.floor(h / 2);
    const n  = hw * hh;
    const cA = new Float32Array(n), cH = new Float32Array(n);
    const cV = new Float32Array(n), cD = new Float32Array(n);
    for (let row = 0; row < hh; row++) {
      for (let col = 0; col < hw; col++) {
        const r0 = 2 * row, c0 = 2 * col;
        const a = ch[r0 * w + c0],         b = ch[r0 * w + c0 + 1];
        const c = ch[(r0 + 1) * w + c0],   d = ch[(r0 + 1) * w + c0 + 1];
        const idx = row * hw + col;
        cA[idx] = (a + b + c + d) / 4;
        cH[idx] = (a - b + c - d) / 4;
        cV[idx] = (a + b - c - d) / 4;
        cD[idx] = (a - b - c + d) / 4;
      }
    }
    return { cA, cH, cV, cD, hw, hh };
  }

  /** Perfect-reconstruction IDWT matching haarDWT2. */
  function haarIDWT2(cA, cH, cV, cD, hw, hh) {
    const w = hw * 2, h = hh * 2;
    const out = new Float32Array(w * h);
    for (let row = 0; row < hh; row++) {
      for (let col = 0; col < hw; col++) {
        const idx = row * hw + col;
        const a = cA[idx], hv = cH[idx], v = cV[idx], d = cD[idx];
        const r0 = 2 * row, c0 = 2 * col;
        out[r0 * w + c0]               = a + hv + v + d;
        out[r0 * w + c0 + 1]           = a - hv + v - d;
        out[(r0 + 1) * w + c0]         = a + hv - v - d;
        out[(r0 + 1) * w + c0 + 1]     = a - hv - v + d;
      }
    }
    return { data: out, w, h };
  }

  /** Build a normalised 1D Gaussian kernel (always odd length). */
  function gaussianKernel1D(sigma) {
    const k = Math.max(3, Math.round(sigma * 4) | 1);  // always odd
    const half = (k - 1) / 2;
    const kernel = new Float32Array(k);
    let sum = 0;
    for (let i = 0; i < k; i++) {
      const x = i - half;
      kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
      sum += kernel[i];
    }
    for (let i = 0; i < k; i++) kernel[i] /= sum;
    return kernel;
  }

  /** Separable 2D convolution of a Float32 channel with a 1D kernel. */
  function separableConvolve(ch, w, h, kernel) {
    const k = kernel.length;
    const half = (k - 1) / 2;
    const tmp = new Float32Array(w * h);
    const out = new Float32Array(w * h);
    // Horizontal pass
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        let acc = 0;
        for (let ki = 0; ki < k; ki++) {
          const c = Math.max(0, Math.min(w - 1, col + ki - half));
          acc += ch[row * w + c] * kernel[ki];
        }
        tmp[row * w + col] = acc;
      }
    }
    // Vertical pass
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        let acc = 0;
        for (let ki = 0; ki < k; ki++) {
          const r = Math.max(0, Math.min(h - 1, row + ki - half));
          acc += tmp[r * w + col] * kernel[ki];
        }
        out[row * w + col] = acc;
      }
    }
    return out;
  }

  /**
   * Iterative Jacobi eigendecomposition for a symmetric 4×4 matrix M
   * (flat Float32Array[16], row-major).
   * Returns { values, vectors } sorted by descending eigenvalue.
   * vectors[k] is the k-th eigenvector as Float32Array(4).
   */
  function jacobiEigen4(M) {
    const n = 4;
    const a = Float32Array.from(M);
    const V = new Float32Array(n * n);
    for (let i = 0; i < n; i++) V[i * n + i] = 1;

    for (let iter = 0; iter < 200; iter++) {
      let p = 0, q = 1, maxA = 0;
      for (let i = 0; i < n - 1; i++) {
        for (let j = i + 1; j < n; j++) {
          const v = Math.abs(a[i * n + j]);
          if (v > maxA) { maxA = v; p = i; q = j; }
        }
      }
      if (maxA < 1e-10) break;

      const app = a[p * n + p], aqq = a[q * n + q], apq = a[p * n + q];
      const tau = (aqq - app) / (2 * apq);
      const t = tau >= 0
        ? 1 / (tau + Math.sqrt(1 + tau * tau))
        : 1 / (tau - Math.sqrt(1 + tau * tau));
      const c = 1 / Math.sqrt(1 + t * t);
      const s = t * c;

      a[p * n + p] = app - t * apq;
      a[q * n + q] = aqq + t * apq;
      a[p * n + q] = 0;
      a[q * n + p] = 0;

      for (let r = 0; r < n; r++) {
        if (r === p || r === q) continue;
        const arp = a[r * n + p], arq = a[r * n + q];
        a[r * n + p] = c * arp - s * arq;
        a[p * n + r] = c * arp - s * arq;
        a[r * n + q] = s * arp + c * arq;
        a[q * n + r] = s * arp + c * arq;
      }
      for (let r = 0; r < n; r++) {
        const vrp = V[r * n + p], vrq = V[r * n + q];
        V[r * n + p] = c * vrp - s * vrq;
        V[r * n + q] = s * vrp + c * vrq;
      }
    }

    const pairs = [];
    for (let i = 0; i < n; i++) {
      const vec = new Float32Array(n);
      for (let j = 0; j < n; j++) vec[j] = V[j * n + i];
      pairs.push({ val: a[i * n + i], vec });
    }
    pairs.sort((x, y) => y.val - x.val);
    return { values: pairs.map((p) => p.val), vectors: pairs.map((p) => p.vec) };
  }

  /* ════════════════════════════════════════════════════════════════
     ALPHA-FUSION MATH HELPERS
  ════════════════════════════════════════════════════════════════ */

  /**
   * Compute Sobel gradient magnitude for a greyscale Float32 channel.
   * Returns Float32Array of per-pixel magnitudes.
   */
  function sobelMagnitude(grey, w, h) {
    const out = new Float32Array(w * h);
    for (let row = 1; row < h - 1; row++) {
      for (let col = 1; col < w - 1; col++) {
        const r0 = (row - 1) * w, r1 = row * w, r2 = (row + 1) * w;
        const gx = -grey[r0 + col - 1] + grey[r0 + col + 1]
                  - 2 * grey[r1 + col - 1] + 2 * grey[r1 + col + 1]
                  - grey[r2 + col - 1] + grey[r2 + col + 1];
        const gy = -grey[r0 + col - 1] - 2 * grey[r0 + col] - grey[r0 + col + 1]
                  + grey[r2 + col - 1] + 2 * grey[r2 + col] + grey[r2 + col + 1];
        out[r1 + col] = Math.sqrt(gx * gx + gy * gy);
      }
    }
    return out;
  }

  /**
   * Compute local variance in a (2*radius+1)² neighbourhood for each pixel.
   * Returns Float32Array.
   */
  function localVariance(ch, w, h, radius) {
    const n    = w * h;
    const out  = new Float32Array(n);
    const diam = 2 * radius + 1;
    const kernel = new Float32Array(diam).fill(1 / diam);
    const mean = separableConvolve(ch, w, h, kernel);
    const sq = new Float32Array(n);
    for (let i = 0; i < n; i++) sq[i] = ch[i] * ch[i];
    const meanSq = separableConvolve(sq, w, h, kernel);
    for (let i = 0; i < n; i++) {
      out[i] = Math.max(0, meanSq[i] - mean[i] * mean[i]);
    }
    return out;
  }

  /**
   * Build per-pixel α = a / (a + b + ε) from two magnitude arrays.
   * Returns Float32Array in [0, 1].
   */
  function buildAlphaMap(a, b) {
    const EPS = 1e-6;
    const alpha = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) alpha[i] = a[i] / (a[i] + b[i] + EPS);
    return alpha;
  }

  /**
   * α-blend: each output channel = α·src + (1-α)·T, then normalize to [0, 255].
   */
  function alphaBlend(alpha, src, thermal) {
    const n = src.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = alpha[i] * src[i] + (1 - alpha[i]) * thermal[i];
    return normalizeToUint8(out);
  }

  /**
   * Build a false-colour heat-map (α=0→blue, α=0.5→green, α=1→red) for visualization.
   */
  function alphaHeatmap(alpha, w, h) {
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      const a = alpha[i];
      let r = 0, g = 0, b = 0;
      if (a <= 0.5) {
        const t = a * 2;
        r = 0;
        g = Math.round(t * 255);
        b = Math.round((1 - t) * 255);
      } else {
        const t = (a - 0.5) * 2;
        r = Math.round(t * 255);
        g = Math.round((1 - t) * 255);
        b = 0;
      }
      data[i * 4]     = r;
      data[i * 4 + 1] = g;
      data[i * 4 + 2] = b;
      data[i * 4 + 3] = 255;
    }
    return new ImageData(data, w, h);
  }

  /* ════════════════════════════════════════════════════════════════
     EXPORT
  ════════════════════════════════════════════════════════════════ */

  window.FusionMath = {
    // Colour
    rgbToHsv, hsvToRgb,
    // Canvas helpers
    bitmapToImageData, workingSize, greyImageData, falseColourImageData,
    // CLAHE
    applyCLAHE, equalizeRgbImageData, equalizeLwirImageData, equalizeToOffscreenCanvas,
    // Reprojection math
    normalizeToUint8, haarDWT2, haarIDWT2, gaussianKernel1D, separableConvolve, jacobiEigen4,
    // Alpha-fusion math
    sobelMagnitude, localVariance, buildAlphaMap, alphaBlend, alphaHeatmap,
  };
})();
