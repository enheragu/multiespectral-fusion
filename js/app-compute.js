// js/app-compute.js — Fusion compute functions for MultiespectralFusion.
// Depends on window.FusionMath (app-math.js must load first).
// All exports assigned to window.FusionCompute.
(function () {
  'use strict';

  const {
    rgbToHsv, hsvToRgb,
    greyImageData, falseColourImageData,
    normalizeToUint8,
    haarDWT2, haarIDWT2,
    gaussianKernel1D, separableConvolve,
    jacobiEigen4,
    sobelMagnitude, localVariance, buildAlphaMap, alphaBlend, alphaHeatmap,
  } = window.FusionMath;

  /* ════════════════════════════════════════════════════════════════
     STATIC EARLY FUSION ALGORITHMS
  ════════════════════════════════════════════════════════════════ */

  function computeRGBT(rgbData, lwirData, w, h) {
    const n = w * h;
    const rgb  = rgbData.data;
    const lwir = lwirData.data;

    const chR = new Uint8Array(n), chG = new Uint8Array(n), chB = new Uint8Array(n);
    const chT = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      chR[i] = rgb[i * 4];
      chG[i] = rgb[i * 4 + 1];
      chB[i] = rgb[i * 4 + 2];
      chT[i] = lwir[i * 4];
    }

    const chRT = new Uint8Array(n), chGT = new Uint8Array(n), chBT = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      chRT[i] = Math.round(chR[i] * chT[i] / 255);
      chGT[i] = Math.round(chG[i] * chT[i] / 255);
      chBT[i] = Math.round(chB[i] * chT[i] / 255);
    }

    const imgChR  = falseColourImageData(chR,  new Uint8Array(n), new Uint8Array(n), w, h);
    const imgChG  = falseColourImageData(new Uint8Array(n), chG, new Uint8Array(n), w, h);
    const imgChB  = falseColourImageData(new Uint8Array(n), new Uint8Array(n), chB, w, h);
    const imgChT  = greyImageData(chT, w, h);
    const imgChRT = falseColourImageData(chRT, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgChGT = falseColourImageData(new Uint8Array(n), chGT, new Uint8Array(n), w, h);
    const imgChBT = falseColourImageData(new Uint8Array(n), new Uint8Array(n), chBT, w, h);
    const output  = falseColourImageData(chRT, chGT, chBT, w, h);

    return {
      stages: [
        {
          key:    'channels',
          opNode: { id: 'op-split', labelKey: 'splitRgb' },
          cards: [
            { id: 'ch-r', imageData: imgChR, cardClass: 'pipe-card--r', desc: 'R' },
            { id: 'ch-g', imageData: imgChG, cardClass: 'pipe-card--g', desc: 'G' },
            { id: 'ch-b', imageData: imgChB, cardClass: 'pipe-card--b', desc: 'B' },
            { id: 'ch-t', imageData: imgChT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          key:    'product',
          opNode: null,
          cards: [
            { id: 'rt', imageData: imgChRT, cardClass: 'pipe-card--r', desc: 'R×T' },
            { id: 'gt', imageData: imgChGT, cardClass: 'pipe-card--g', desc: 'G×T' },
            { id: 'bt', imageData: imgChBT, cardClass: 'pipe-card--b', desc: 'B×T' },
          ],
        },
        {
          key:    'merge',
          opNode: { id: 'op-merge', labelKey: 'mergeRgb' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out', desc: 'Output', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',  to: 'op-split', color: 'var(--clr-border)' },
        { from: 'src-lwir', to: 'ch-t',     color: 'var(--clr-ch-t)' },
        { from: 'op-split', to: 'ch-r', color: 'var(--clr-ch-r)' },
        { from: 'op-split', to: 'ch-g', color: 'var(--clr-ch-g)' },
        { from: 'op-split', to: 'ch-b', color: 'var(--clr-ch-b)' },
        { from: 'ch-r', to: 'rt',       color: 'var(--clr-ch-r)' },
        { from: 'ch-g', to: 'gt',       color: 'var(--clr-ch-g)' },
        { from: 'ch-b', to: 'bt',       color: 'var(--clr-ch-b)' },
        { from: 'ch-t', to: 'rt',       color: 'var(--clr-ch-t)' },
        { from: 'ch-t', to: 'gt',       color: 'var(--clr-ch-t)' },
        { from: 'ch-t', to: 'bt',       color: 'var(--clr-ch-t)' },
        { from: 'rt',   to: 'op-merge', color: 'var(--clr-ch-r)' },
        { from: 'gt',   to: 'op-merge', color: 'var(--clr-ch-g)' },
        { from: 'bt',   to: 'op-merge', color: 'var(--clr-ch-b)' },
        { from: 'op-merge', to: 'out',  color: 'var(--clr-border)' },
      ],
      output,
    };
  }

  function computeHSVT(rgbData, lwirData, w, h) {
    const n = w * h;
    const rgb  = rgbData.data;
    const lwir = lwirData.data;

    const chH   = new Uint8Array(n), chS   = new Uint8Array(n);
    const chV   = new Uint8Array(n), chT   = new Uint8Array(n);
    const chVT  = new Uint8Array(n);
    const outR  = new Uint8Array(n), outG  = new Uint8Array(n), outB  = new Uint8Array(n);

    for (let i = 0; i < n; i++) {
      const r = rgb[i * 4], g = rgb[i * 4 + 1], b = rgb[i * 4 + 2];
      const t = lwir[i * 4];
      const { h, s, v } = rgbToHsv(r, g, b);

      chH[i]  = Math.round((h / 360) * 255);
      chS[i]  = Math.round(s * 255);
      chV[i]  = Math.round(v * 255);
      chT[i]  = t;

      const vt = (v + t / 255) / 2;
      chVT[i] = Math.round(vt * 255);

      const [ro, go, bo] = hsvToRgb(h, s, vt);
      outR[i] = ro; outG[i] = go; outB[i] = bo;
    }

    const imgH = new ImageData(w * h * 4 > 0 ? (() => {
      const d = new Uint8ClampedArray(n * 4);
      for (let i = 0; i < n; i++) {
        const h360 = (chH[i] / 255) * 360;
        const [r, g, b] = hsvToRgb(h360, 1, 1);
        d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
      }
      return d;
    })() : new Uint8ClampedArray(n * 4), w, h);

    const imgS   = greyImageData(chS, w, h);
    const imgV   = greyImageData(chV, w, h);
    const imgT   = greyImageData(chT, w, h);
    const imgVT  = greyImageData(chVT, w, h);
    const output = falseColourImageData(outR, outG, outB, w, h);

    return {
      stages: [
        {
          key:    'hsv-split',
          opNode: { id: 'op-split', labelKey: 'splitHsv' },
          cards: [
            { id: 'h', imageData: imgH, cardClass: 'pipe-card--h', desc: 'H (Hue)' },
            { id: 's', imageData: imgS, cardClass: 'pipe-card--s', desc: 'S (Sat.)' },
            { id: 'v', imageData: imgV, cardClass: 'pipe-card--v', desc: 'V (Value)' },
            { id: 't', imageData: imgT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          key:    'blend',
          opNode: null,
          cards: [
            { id: 'vt', imageData: imgVT, cardClass: 'pipe-card--v', desc: 'avg(V,T)' },
          ],
        },
        {
          key:    'hsv-merge',
          opNode: { id: 'op-merge', labelKey: 'hsvBack' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out', desc: 'Output', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',  to: 'op-split', color: 'var(--clr-border)' },
        { from: 'src-lwir', to: 't',        color: 'var(--clr-ch-t)' },
        { from: 'op-split', to: 'h', color: 'var(--clr-ch-h)' },
        { from: 'op-split', to: 's', color: 'var(--clr-ch-s)' },
        { from: 'op-split', to: 'v', color: 'var(--clr-ch-v)' },
        { from: 'v',  to: 'vt',      color: 'var(--clr-ch-v)' },
        { from: 't',  to: 'vt',      color: 'var(--clr-ch-t)' },
        { from: 'h', to: 'op-merge', color: 'var(--clr-ch-h)' },
        { from: 's', to: 'op-merge', color: 'var(--clr-ch-s)' },
        { from: 'vt', to: 'op-merge', color: 'var(--clr-ch-v)' },
        { from: 'op-merge', to: 'out', color: 'var(--clr-border)' },
      ],
      output,
    };
  }

  function computeVTHS(rgbData, lwirData, w, h) {
    const n = w * h;
    const rgb  = rgbData.data;
    const lwir = lwirData.data;

    const chH  = new Uint8Array(n), chS  = new Uint8Array(n);
    const chV  = new Uint8Array(n), chT  = new Uint8Array(n), chHS = new Uint8Array(n);

    for (let i = 0; i < n; i++) {
      const r = rgb[i * 4], g = rgb[i * 4 + 1], b = rgb[i * 4 + 2];
      const t = lwir[i * 4];
      const { h, s, v } = rgbToHsv(r, g, b);

      const hByte = Math.round((h / 360) * 255);
      const sByte = Math.round(s * 255);
      const vByte = Math.round(v * 255);

      chH[i]  = hByte;
      chS[i]  = sByte;
      chV[i]  = vByte;
      chT[i]  = t;
      chHS[i] = (hByte & 0xF0) | (sByte >> 4);
    }

    const imgH = new ImageData((() => {
      const d = new Uint8ClampedArray(n * 4);
      for (let i = 0; i < n; i++) {
        const h360 = (chH[i] / 255) * 360;
        const [r, g, b] = hsvToRgb(h360, 1, 1);
        d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
      }
      return d;
    })(), w, h);

    const imgS   = greyImageData(chS, w, h);
    const imgV   = greyImageData(chV, w, h);
    const imgT   = greyImageData(chT, w, h);
    const imgHS  = greyImageData(chHS, w, h);
    // R/B match the real output: Python cv.merge([v,th,hs]) is BGR, the worker
    // reverses to displayed RGB = (hs, th, v) — so the preview must be (HS,T,V).
    const output = falseColourImageData(chHS, chT, chV, w, h);

    return {
      stages: [
        {
          key:    'vths-split',
          opNode: { id: 'op-split', labelKey: 'splitHsv' },
          cards: [
            { id: 'h', imageData: imgH, cardClass: 'pipe-card--h', desc: 'H (Hue)' },
            { id: 's', imageData: imgS, cardClass: 'pipe-card--s', desc: 'S (Sat.)' },
            { id: 'v', imageData: imgV, cardClass: 'pipe-card--v', desc: 'V (Value)' },
            { id: 't', imageData: imgT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          key:    'vths-pack',
          opNode: null,
          cards: [
            { id: 'hs', imageData: imgHS, cardClass: 'pipe-card--hs', desc: 'HS (packed)' },
          ],
        },
        {
          key:    'vths-merge',
          opNode: { id: 'op-merge', labelKey: 'mergeOut' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out', desc: 'Output (R=HS, G=T, B=V)', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',  to: 'op-split', color: 'var(--clr-border)' },
        { from: 'src-lwir', to: 't',        color: 'var(--clr-ch-t)' },
        { from: 'op-split', to: 'h', color: 'var(--clr-ch-h)' },
        { from: 'op-split', to: 's', color: 'var(--clr-ch-s)' },
        { from: 'op-split', to: 'v', color: 'var(--clr-ch-v)' },
        { from: 'h',  to: 'hs', color: 'var(--clr-ch-h)' },
        { from: 's',  to: 'hs', color: 'var(--clr-ch-s)' },
        { from: 'v',  to: 'op-merge', color: 'var(--clr-ch-v)' },
        { from: 't',  to: 'op-merge', color: 'var(--clr-ch-t)' },
        { from: 'hs', to: 'op-merge', color: 'var(--clr-ch-hs)' },
        { from: 'op-merge', to: 'out', color: 'var(--clr-border)' },
      ],
      output,
    };
  }

  function computeVT(rgbData, lwirData, w, h) {
    const n = w * h;
    const rgb  = rgbData.data;
    const lwir = lwirData.data;

    const chH  = new Uint8Array(n), chS  = new Uint8Array(n);
    const chV  = new Uint8Array(n), chT  = new Uint8Array(n), chAvg = new Uint8Array(n);

    for (let i = 0; i < n; i++) {
      const r = rgb[i * 4], g = rgb[i * 4 + 1], b = rgb[i * 4 + 2];
      const t = lwir[i * 4];
      const { h, s, v } = rgbToHsv(r, g, b);
      const vByte = Math.round(v * 255);

      chH[i]   = Math.round((h / 360) * 255);
      chS[i]   = Math.round(s * 255);
      chV[i]   = vByte;
      chT[i]   = t;
      chAvg[i] = Math.round((vByte + t) / 2);
    }

    const imgH = new ImageData((() => {
      const d = new Uint8ClampedArray(n * 4);
      for (let i = 0; i < n; i++) {
        const h360 = (chH[i] / 255) * 360;
        const [r, g, b] = hsvToRgb(h360, 1, 1);
        d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
      }
      return d;
    })(), w, h);

    const imgS   = greyImageData(chS, w, h);
    const imgV   = greyImageData(chV, w, h);
    const imgT   = greyImageData(chT, w, h);
    const imgAvg = greyImageData(chAvg, w, h);
    // R/B match the real output: Python cv.merge([v,th,both]) is BGR, the worker
    // reverses to displayed RGB = (avg, th, v) — so the preview must be (avg,T,V).
    const output = falseColourImageData(chAvg, chT, chV, w, h);

    return {
      stages: [
        {
          key:    'vt-split',
          opNode: { id: 'op-split', labelKey: 'splitHsv' },
          cards: [
            { id: 'h', imageData: imgH, cardClass: 'pipe-card--h', desc: 'H (Hue)' },
            { id: 's', imageData: imgS, cardClass: 'pipe-card--s', desc: 'S (Sat.)' },
            { id: 'v', imageData: imgV, cardClass: 'pipe-card--v', desc: 'V (Value)' },
            { id: 't', imageData: imgT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          key:    'vt-blend',
          opNode: null,
          cards: [
            { id: 'avg', imageData: imgAvg, cardClass: 'pipe-card--v', desc: 'avg(V,T)' },
          ],
        },
        {
          key:    'vt-merge',
          opNode: { id: 'op-merge', labelKey: 'mergeOut' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out', desc: 'Output (R=avg, G=T, B=V)', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',  to: 'op-split', color: 'var(--clr-border)' },
        { from: 'src-lwir', to: 't',        color: 'var(--clr-ch-t)' },
        { from: 'op-split', to: 'h', color: 'var(--clr-ch-h)' },
        { from: 'op-split', to: 's', color: 'var(--clr-ch-s)' },
        { from: 'op-split', to: 'v', color: 'var(--clr-ch-v)' },
        { from: 'v',   to: 'avg', color: 'var(--clr-ch-v)' },
        { from: 't',   to: 'avg', color: 'var(--clr-ch-t)' },
        { from: 'v',  to: 'op-merge', color: 'var(--clr-ch-v)' },
        { from: 'avg', to: 'op-merge', color: 'var(--clr-ch-v)' },
        { from: 't',  to: 'op-merge', color: 'var(--clr-ch-t)' },
        { from: 'op-merge', to: 'out', color: 'var(--clr-border)' },
      ],
      output,
    };
  }

  /* ════════════════════════════════════════════════════════════════
     REPROJECTION FUSION ALGORITHMS
  ════════════════════════════════════════════════════════════════ */

  // Assemble the canonical 2x2 wavelet decomposition view from the four subbands:
  //   ┌─────────┬─────────┐
  //   │  cA     │  cH      │   (approximation | horizontal detail)
  //   ├─────────┼─────────┤
  //   │  cV     │  cD      │   (vertical detail | diagonal detail)
  //   └─────────┴─────────┘
  // Each quadrant is normalised independently so the (small) detail subbands are
  // visible. `tint` false-colours per channel ('r'|'g'|'b') or renders grey.
  // Enhance a (small, signed) detail subband for visibility: take magnitude and
  // apply a gamma<1 lift so the faint edge coefficients become perceptible — they
  // carry real structure that is otherwise near-black. cA (approximation) is shown
  // as-is. The detail quadrants are labelled with ↑ to announce they are amplified.
  function _enhanceDetail(arr) {
    let mx = 0;
    for (let i = 0; i < arr.length; i++) { const a = Math.abs(arr[i]); if (a > mx) mx = a; }
    const out = new Uint8Array(arr.length);
    const inv = mx > 0 ? 1 / mx : 0;
    for (let i = 0; i < arr.length; i++) out[i] = Math.round(255 * Math.pow(Math.abs(arr[i]) * inv, 0.6));
    return out;
  }

  function subbandMosaic(d, hw, hh, tint) {
    const W = hw * 2, H = hh * 2;
    const buf = new Uint8Array(W * H);
    const place = (q, ox, oy) => {
      for (let y = 0; y < hh; y++)
        for (let x = 0; x < hw; x++) buf[(oy + y) * W + (ox + x)] = q[y * hw + x];
    };
    place(normalizeToUint8(d.cA), 0, 0);
    place(_enhanceDetail(d.cH), hw, 0);
    place(_enhanceDetail(d.cV), 0, hh);
    place(_enhanceDetail(d.cD), hw, hh);

    const z = new Uint8Array(W * H);
    const img = tint === 'r' ? falseColourImageData(buf, z, z, W, H)
              : tint === 'g' ? falseColourImageData(z, buf, z, W, H)
              : tint === 'b' ? falseColourImageData(z, z, buf, W, H)
              :                greyImageData(buf, W, H);

    // Bake quadrant dividers + labels into the image (so they travel as plain ImageData).
    const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    ctx.putImageData(img, 0, 0);
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = Math.max(1, Math.round(W / 220));
    ctx.beginPath(); ctx.moveTo(hw, 0); ctx.lineTo(hw, H); ctx.moveTo(0, hh); ctx.lineTo(W, hh); ctx.stroke();
    const fs = Math.max(10, Math.round(W * 0.026));
    ctx.font = `bold ${fs}px system-ui, sans-serif`;
    ctx.textBaseline = 'top';
    const lab = (t, x, y) => {
      const tw = ctx.measureText(t).width;
      ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(x + 3, y + 3, tw + 6, fs + 4);
      ctx.fillStyle = 'rgba(255,255,255,0.95)'; ctx.fillText(t, x + 5, y + 4);
    };
    // cA = approximation (as-is); cH/cV/cD = detail subbands, amplified for visibility (*).
    lab('cA', 0, 0); lab('cH*', hw, 0); lab('cV*', 0, hh); lab('cD*', hw, hh);
    // The '*' on cH*/cV*/cD* (detail amplified for visibility) is explained once,
    // bilingually, by the shared starLegend beside the diagram — no baked caption
    // (it was Spanish-only, mismatching the English band labels).
    return ctx.getImageData(0, 0, W, H);
  }

  function _computeWaveletFusion(rgbData, lwirData, w, h, mode) {
    const n   = w * h;
    const rgb = rgbData.data, lwir = lwirData.data;

    const chR = new Uint8Array(n), chG = new Uint8Array(n);
    const chB = new Uint8Array(n), chT = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      chR[i] = rgb[i * 4]; chG[i] = rgb[i * 4 + 1];
      chB[i] = rgb[i * 4 + 2]; chT[i] = lwir[i * 4];
    }
    const fR = new Float32Array(chR), fG = new Float32Array(chG);
    const fB = new Float32Array(chB), fT = new Float32Array(chT);

    const dR = haarDWT2(fR, w, h), dG = haarDWT2(fG, w, h);
    const dB = haarDWT2(fB, w, h), dT = haarDWT2(fT, w, h);
    const { hw, hh } = dR;

    const fuse = mode === 'max'
      ? (a, b) => (Math.abs(a) >= Math.abs(b) ? a : b)
      : (a, b) => (a + b) / 2;

    function fuseCoeffs(ca, cb) {
      const out = new Float32Array(ca.length);
      for (let i = 0; i < ca.length; i++) out[i] = fuse(ca[i], cb[i]);
      return out;
    }

    // The APPROXIMATION band (cA) follows a DIFFERENT rule than the detail bands:
    //   • wavelet (avg)  → cA averaged, like the detail bands.
    //   • waveletmax     → detail is abs-maxed, but cA is a THERMAL-WEIGHTED BLEND
    //     (Python processWaveletCoeffsMax: w = T/T.max; cA = w·cA_T + (1−w)·cA_rgb)
    //     — NOT abs-maxed. Abs-maxing cA (the old behaviour) misrepresented both
    //     the coeff mosaic and the reconstructed preview channels.
    let wA = null;
    if (mode === 'max') {
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < dT.cA.length; i++) { const v = dT.cA[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
      const rng = (mx - mn) || 1;
      wA = new Float32Array(dT.cA.length);
      for (let i = 0; i < dT.cA.length; i++) wA[i] = (dT.cA[i] - mn) / rng; // ≈ T/T.max at cA scale
    }
    function fuseCA(crgb, ct) {
      if (!wA) return fuseCoeffs(crgb, ct);                 // avg mode: average
      const out = new Float32Array(crgb.length);
      for (let i = 0; i < crgb.length; i++) out[i] = wA[i] * ct[i] + (1 - wA[i]) * crgb[i];
      return out;
    }

    // Hoist the fused coefficients per channel so we can both reconstruct AND
    // visualise them — fusion happens HERE, in the coefficient domain (each RGB
    // channel's subbands fused against the SHARED thermal subbands), then ONE
    // inverse DWT per channel.
    const fR_cA = fuseCA(dR.cA, dT.cA), fR_cH = fuseCoeffs(dR.cH, dT.cH),
          fR_cV = fuseCoeffs(dR.cV, dT.cV), fR_cD = fuseCoeffs(dR.cD, dT.cD);
    const fG_cA = fuseCA(dG.cA, dT.cA), fG_cH = fuseCoeffs(dG.cH, dT.cH),
          fG_cV = fuseCoeffs(dG.cV, dT.cV), fG_cD = fuseCoeffs(dG.cD, dT.cD);
    const fB_cA = fuseCA(dB.cA, dT.cA), fB_cH = fuseCoeffs(dB.cH, dT.cH),
          fB_cV = fuseCoeffs(dB.cV, dT.cV), fB_cD = fuseCoeffs(dB.cD, dT.cD);

    const recR = haarIDWT2(fR_cA, fR_cH, fR_cV, fR_cD, hw, hh);
    const recG = haarIDWT2(fG_cA, fG_cH, fG_cV, fG_cD, hw, hh);
    const recB = haarIDWT2(fB_cA, fB_cH, fB_cV, fB_cD, hw, hh);

    const rw = recR.w, rh = recR.h;
    const outR = normalizeToUint8(recR.data);
    const outG = normalizeToUint8(recG.data);
    const outB = normalizeToUint8(recB.data);

    const imgChR = falseColourImageData(chR, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgChG = falseColourImageData(new Uint8Array(n), chG, new Uint8Array(n), w, h);
    const imgChB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), chB, w, h);
    const imgChT = greyImageData(chT, w, h);
    // Canonical wavelet decomposition mosaics (cA | cH / cV | cD) per channel,
    // so the approximation AND the detail subbands are both visible.
    const imgDwtR = subbandMosaic(dR, hw, hh, 'r');
    const imgDwtG = subbandMosaic(dG, hw, hh, 'g');
    const imgDwtB = subbandMosaic(dB, hw, hh, 'b');
    const imgDwtT = subbandMosaic(dT, hw, hh, 'grey');
    // Fused-coefficient mosaics (each channel's subbands after fusion vs thermal).
    const imgFusR = subbandMosaic({ cA: fR_cA, cH: fR_cH, cV: fR_cV, cD: fR_cD }, hw, hh, 'r');
    const imgFusG = subbandMosaic({ cA: fG_cA, cH: fG_cH, cV: fG_cV, cD: fG_cD }, hw, hh, 'g');
    const imgFusB = subbandMosaic({ cA: fB_cA, cH: fB_cH, cV: fB_cV, cD: fB_cD }, hw, hh, 'b');
    const output = falseColourImageData(outR, outG, outB, rw, rh);

    // waveletmax uses a MIXED rule (cA thermal-weighted blend, detail abs-max);
    // wavelet averages every band. Reflect that honestly in the card label.
    const fz = mode === 'max' ? 'cA blend·detail maxabs' : 'avg coefs';
    return {
      stages: [
        {
          key: 'w-channels',
          opNode: { id: 'op-split', labelKey: 'splitRgb' },
          cards: [
            { id: 'ch-r', imageData: imgChR, cardClass: 'pipe-card--r', desc: 'R' },
            { id: 'ch-g', imageData: imgChG, cardClass: 'pipe-card--g', desc: 'G' },
            { id: 'ch-b', imageData: imgChB, cardClass: 'pipe-card--b', desc: 'B' },
            { id: 'ch-t', imageData: imgChT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          key: 'w-dwt',
          opNode: { id: 'op-dwt', labelKey: 'dwtOp' },
          cards: [
            { id: 'dwt-r', imageData: imgDwtR, cardClass: 'pipe-card--r', desc: 'DWT(R)' },
            { id: 'dwt-g', imageData: imgDwtG, cardClass: 'pipe-card--g', desc: 'DWT(G)' },
            { id: 'dwt-b', imageData: imgDwtB, cardClass: 'pipe-card--b', desc: 'DWT(B)' },
            { id: 'dwt-t', imageData: imgDwtT, cardClass: 'pipe-card--t', desc: 'DWT(T)', separateBefore: true },
          ],
        },
        {
          // No single op-node: each channel's subbands are fused with the SHARED
          // thermal subbands independently (like RGBT's R×T/G×T/B×T), so the thermal
          // fans into every fused card rather than everything funnelling through one node.
          key: 'w-fuse',
          opNode: null,
          cards: [
            { id: 'fus-r', imageData: imgFusR, cardClass: 'pipe-card--r', desc: `${fz} (R,T)` },
            { id: 'fus-g', imageData: imgFusG, cardClass: 'pipe-card--g', desc: `${fz} (G,T)` },
            { id: 'fus-b', imageData: imgFusB, cardClass: 'pipe-card--b', desc: `${fz} (B,T)` },
          ],
        },
        {
          key: 'w-out',
          opNode: { id: 'op-idwt', labelKey: 'idwtOp' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out', desc: 'Output', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',  to: 'op-split', color: 'var(--clr-border)' },
        { from: 'src-lwir', to: 'ch-t',     color: 'var(--clr-ch-t)' },
        { from: 'op-split', to: 'ch-r',     color: 'var(--clr-ch-r)' },
        { from: 'op-split', to: 'ch-g',     color: 'var(--clr-ch-g)' },
        { from: 'op-split', to: 'ch-b',     color: 'var(--clr-ch-b)' },
        { from: 'ch-r', to: 'op-dwt', color: 'var(--clr-ch-r)' },
        { from: 'ch-g', to: 'op-dwt', color: 'var(--clr-ch-g)' },
        { from: 'ch-b', to: 'op-dwt', color: 'var(--clr-ch-b)' },
        { from: 'ch-t', to: 'op-dwt', color: 'var(--clr-ch-t)' },
        { from: 'op-dwt', to: 'dwt-r', color: 'var(--clr-ch-r)' },
        { from: 'op-dwt', to: 'dwt-g', color: 'var(--clr-ch-g)' },
        { from: 'op-dwt', to: 'dwt-b', color: 'var(--clr-ch-b)' },
        { from: 'op-dwt', to: 'dwt-t', color: 'var(--clr-ch-t)' },
        { from: 'dwt-r', to: 'fus-r', color: 'var(--clr-ch-r)' },
        { from: 'dwt-g', to: 'fus-g', color: 'var(--clr-ch-g)' },
        { from: 'dwt-b', to: 'fus-b', color: 'var(--clr-ch-b)' },
        { from: 'dwt-t', to: 'fus-r', color: 'var(--clr-ch-t)' },
        { from: 'dwt-t', to: 'fus-g', color: 'var(--clr-ch-t)' },
        { from: 'dwt-t', to: 'fus-b', color: 'var(--clr-ch-t)' },
        { from: 'fus-r', to: 'op-idwt', color: 'var(--clr-ch-r)' },
        { from: 'fus-g', to: 'op-idwt', color: 'var(--clr-ch-g)' },
        { from: 'fus-b', to: 'op-idwt', color: 'var(--clr-ch-b)' },
        { from: 'op-idwt', to: 'out', color: 'var(--clr-border)' },
      ],
      output,
    };
  }

  // Curvelet decomposition PREVIEW mosaic: coarse (low-pass) | detail (high-pass)
  // placed side by side at FULL resolution (each panel keeps the image aspect, so
  // the split is clearly a visualization and NOT a deformation by the method). The
  // mosaic is 2w×h (wider than tall) and fills the wide thumbnail naturally. The
  // detail panel is amplified for visibility (*); the caption marks it a preview —
  // the real directional UDCT bands drive the faithful Pyodide OUTPUT.
  // Multi-scale 2×2 mosaic for the curvelet preview: [coarse | det¹ ; det² | det³]
  // — nscales=4 like the real UDCT (one coarse band + three detail scales). Each
  // band is box-downsampled to half-size to fit the grid; detail bands are
  // contrast-lifted (*) for visibility (the '*' is explained by the shared legend).
  // NOTE: the real UDCT additionally splits each DETAIL scale into oriented
  // directional sub-bands — those exist only in the faithful Pyodide output, not
  // in this fast multi-scale preview.
  function _curveletPyramidMosaic(p, w, h, tint) {
    const hw = w >> 1, hh = h >> 1, W = hw * 2, H = hh * 2;
    const ds = (src) => {                       // box-downsample 2× to hw×hh
      const o = new Float32Array(hw * hh);
      for (let y = 0; y < hh; y++) for (let x = 0; x < hw; x++) {
        const sx = x * 2, sy = y * 2;
        o[y * hw + x] = (src[sy * w + sx] + src[sy * w + sx + 1]
                       + src[(sy + 1) * w + sx] + src[(sy + 1) * w + sx + 1]) / 4;
      }
      return o;
    };
    const cC = normalizeToUint8(ds(p.coarse));
    const c1 = _enhanceDetail(ds(p.d1)), c2 = _enhanceDetail(ds(p.d2)), c3 = _enhanceDetail(ds(p.d3));
    const buf = new Uint8Array(W * H);
    for (let y = 0; y < hh; y++) for (let x = 0; x < hw; x++) {
      buf[y * W + x]             = cC[y * hw + x];        // top-left  : coarse
      buf[y * W + hw + x]        = c1[y * hw + x];        // top-right : detail scale 1 (finest)
      buf[(hh + y) * W + x]      = c2[y * hw + x];        // bot-left  : detail scale 2
      buf[(hh + y) * W + hw + x] = c3[y * hw + x];        // bot-right : detail scale 3 (coarsest)
    }
    const z = new Uint8Array(W * H);
    const img = tint === 'r' ? falseColourImageData(buf, z, z, W, H)
              : tint === 'g' ? falseColourImageData(z, buf, z, W, H)
              : tint === 'b' ? falseColourImageData(z, z, buf, W, H)
              :                greyImageData(buf, W, H);
    const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d'); ctx.putImageData(img, 0, 0);
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = Math.max(1, Math.round(W / 220));
    ctx.beginPath(); ctx.moveTo(hw, 0); ctx.lineTo(hw, H); ctx.moveTo(0, hh); ctx.lineTo(W, hh); ctx.stroke();
    const fs = Math.max(10, Math.round(W * 0.026));
    ctx.font = `bold ${fs}px system-ui, sans-serif`; ctx.textBaseline = 'top';
    const lab = (t, x, y) => {
      const tw = ctx.measureText(t).width;
      ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(x + 3, y + 3, tw + 6, fs + 4);
      ctx.fillStyle = 'rgba(255,255,255,0.95)'; ctx.fillText(t, x + 5, y + 4);
    };
    lab('coarse', 0, 0); lab('det¹*', hw, 0); lab('det²*', 0, hh); lab('det³*', hw, hh);
    return ctx.getImageData(0, 0, W, H);
  }

  // Curvelet fusion diagram. Structure mirrors the real algorithm (and the
  // wavelet diagram): forward UDCT per channel → coarse + detail bands → fuse
  // per channel against the thermal bands → inverse UDCT → normalize. Faithful
  // to _fuse_curvelet_scales: the COARSE band is ALWAYS averaged; only DETAIL
  // bands use the mode (avg or abs-max). The OUTPUT shown is swapped for the
  // real Pyodide curvelet; the band cards are an instant preview (see mosaic).
  function _computeCurveletFusion(rgbData, lwirData, w, h, mode) {
    const n = w * h;
    const rgb = rgbData.data, lwir = lwirData.data;
    const chR = new Uint8Array(n), chG = new Uint8Array(n), chB = new Uint8Array(n), chT = new Uint8Array(n);
    for (let i = 0; i < n; i++) { chR[i] = rgb[i*4]; chG[i] = rgb[i*4+1]; chB[i] = rgb[i*4+2]; chT[i] = lwir[i*4]; }
    const fR = new Float32Array(chR), fG = new Float32Array(chG), fB = new Float32Array(chB), fT = new Float32Array(chT);

    // Multi-scale à trous Laplacian pyramid — nscales=4 like the real UDCT: one
    // coarse band (g3) + three detail scales (d1 finest … d3 coarsest) from
    // successive Gaussian blurs. (The real UDCT additionally splits each detail
    // scale into oriented directional bands — only the Pyodide output has those.)
    const k1 = gaussianKernel1D(1.0), k2 = gaussianKernel1D(2.0), k4 = gaussianKernel1D(4.0);
    const pyramid = (f) => {
      const g1 = separableConvolve(f, w, h, k1);
      const g2 = separableConvolve(g1, w, h, k2);
      const g3 = separableConvolve(g2, w, h, k4);
      const d1 = new Float32Array(n), d2 = new Float32Array(n), d3 = new Float32Array(n);
      for (let i = 0; i < n; i++) { d1[i] = f[i] - g1[i]; d2[i] = g1[i] - g2[i]; d3[i] = g2[i] - g3[i]; }
      return { coarse: g3, d1, d2, d3 };                  // coarse + d1 + d2 + d3 === f
    };
    const pR = pyramid(fR), pG = pyramid(fG), pB = pyramid(fB), pT = pyramid(fT);

    const fuseCoarse = (a, b) => (a + b) / 2;                                  // coarse (j==0): always avg
    const fuseDetail = mode === 'max'
      ? (a, b) => (Math.abs(a) >= Math.abs(b) ? a : b)                          // detail: abs-max
      : (a, b) => (a + b) / 2;                                                  // detail: avg
    const fuseBand = (a, b, isCoarse) => {
      const o = new Float32Array(n), fn = isCoarse ? fuseCoarse : fuseDetail;
      for (let i = 0; i < n; i++) o[i] = fn(a[i], b[i]);
      return o;
    };
    const fusePyr = (p) => ({
      coarse: fuseBand(p.coarse, pT.coarse, true),
      d1: fuseBand(p.d1, pT.d1, false), d2: fuseBand(p.d2, pT.d2, false), d3: fuseBand(p.d3, pT.d3, false),
    });
    const fpR = fusePyr(pR), fpG = fusePyr(pG), fpB = fusePyr(pB);
    const recon = (fp) => {
      const o = new Float32Array(n);
      for (let i = 0; i < n; i++) o[i] = fp.coarse[i] + fp.d1[i] + fp.d2[i] + fp.d3[i];
      return normalizeToUint8(o);
    };
    const output = falseColourImageData(recon(fpR), recon(fpG), recon(fpB), w, h);

    const imgChR = falseColourImageData(chR, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgChG = falseColourImageData(new Uint8Array(n), chG, new Uint8Array(n), w, h);
    const imgChB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), chB, w, h);
    const imgChT = greyImageData(chT, w, h);
    const cvR = _curveletPyramidMosaic(pR, w, h, 'r'), cvG = _curveletPyramidMosaic(pG, w, h, 'g'),
          cvB = _curveletPyramidMosaic(pB, w, h, 'b'), cvT = _curveletPyramidMosaic(pT, w, h, 't');
    const fuR = _curveletPyramidMosaic(fpR, w, h, 'r'), fuG = _curveletPyramidMosaic(fpG, w, h, 'g'), fuB = _curveletPyramidMosaic(fpB, w, h, 'b');

    const fz = mode === 'max' ? 'coarse avg · detail max' : 'avg coefs';
    return {
      stages: [
        {
          key: 'c-channels',
          opNode: { id: 'op-split', labelKey: 'splitRgb' },
          cards: [
            { id: 'ch-r', imageData: imgChR, cardClass: 'pipe-card--r', desc: 'R' },
            { id: 'ch-g', imageData: imgChG, cardClass: 'pipe-card--g', desc: 'G' },
            { id: 'ch-b', imageData: imgChB, cardClass: 'pipe-card--b', desc: 'B' },
            { id: 'ch-t', imageData: imgChT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          key: 'c-decomp',
          opNode: { id: 'op-curv-fwd', labelKey: 'curvOp' },
          cards: [
            { id: 'cv-r', imageData: cvR, cardClass: 'pipe-card--r', desc: 'UDCT(R)' },
            { id: 'cv-g', imageData: cvG, cardClass: 'pipe-card--g', desc: 'UDCT(G)' },
            { id: 'cv-b', imageData: cvB, cardClass: 'pipe-card--b', desc: 'UDCT(B)' },
            { id: 'cv-t', imageData: cvT, cardClass: 'pipe-card--t', desc: 'UDCT(T)', separateBefore: true },
          ],
        },
        {
          key: 'c-fuse',
          opNode: null,
          cards: [
            { id: 'fus-r', imageData: fuR, cardClass: 'pipe-card--r', desc: `${fz} (R,T)` },
            { id: 'fus-g', imageData: fuG, cardClass: 'pipe-card--g', desc: `${fz} (G,T)` },
            { id: 'fus-b', imageData: fuB, cardClass: 'pipe-card--b', desc: `${fz} (B,T)` },
          ],
        },
        {
          key: 'c-out',
          opNode: { id: 'op-curv-bwd', labelKey: 'icurvOp' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out', desc: 'Output', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',      to: 'op-split',     color: 'var(--clr-border)' },
        { from: 'src-lwir',     to: 'ch-t',         color: 'var(--clr-ch-t)' },
        { from: 'op-split',     to: 'ch-r',         color: 'var(--clr-ch-r)' },
        { from: 'op-split',     to: 'ch-g',         color: 'var(--clr-ch-g)' },
        { from: 'op-split',     to: 'ch-b',         color: 'var(--clr-ch-b)' },
        { from: 'ch-r',         to: 'op-curv-fwd',  color: 'var(--clr-ch-r)' },
        { from: 'ch-g',         to: 'op-curv-fwd',  color: 'var(--clr-ch-g)' },
        { from: 'ch-b',         to: 'op-curv-fwd',  color: 'var(--clr-ch-b)' },
        { from: 'ch-t',         to: 'op-curv-fwd',  color: 'var(--clr-ch-t)' },
        { from: 'op-curv-fwd',  to: 'cv-r',         color: 'var(--clr-ch-r)' },
        { from: 'op-curv-fwd',  to: 'cv-g',         color: 'var(--clr-ch-g)' },
        { from: 'op-curv-fwd',  to: 'cv-b',         color: 'var(--clr-ch-b)' },
        { from: 'op-curv-fwd',  to: 'cv-t',         color: 'var(--clr-ch-t)' },
        { from: 'cv-r',         to: 'fus-r',        color: 'var(--clr-ch-r)' },
        { from: 'cv-g',         to: 'fus-g',        color: 'var(--clr-ch-g)' },
        { from: 'cv-b',         to: 'fus-b',        color: 'var(--clr-ch-b)' },
        { from: 'cv-t',         to: 'fus-r',        color: 'var(--clr-ch-t)' },
        { from: 'cv-t',         to: 'fus-g',        color: 'var(--clr-ch-t)' },
        { from: 'cv-t',         to: 'fus-b',        color: 'var(--clr-ch-t)' },
        { from: 'fus-r',        to: 'op-curv-bwd',  color: 'var(--clr-ch-r)' },
        { from: 'fus-g',        to: 'op-curv-bwd',  color: 'var(--clr-ch-g)' },
        { from: 'fus-b',        to: 'op-curv-bwd',  color: 'var(--clr-ch-b)' },
        { from: 'op-curv-bwd',  to: 'out',          color: 'var(--clr-border)' },
      ],
      output,
    };
  }

  function _computeDecompositionFusion(rgbData, lwirData, w, h, kind) {
    const n   = w * h;
    const rgb = rgbData.data, lwir = lwirData.data;

    const chR = new Uint8Array(n), chG = new Uint8Array(n);
    const chB = new Uint8Array(n), chT = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      chR[i] = rgb[i * 4]; chG[i] = rgb[i * 4 + 1];
      chB[i] = rgb[i * 4 + 2]; chT[i] = lwir[i * 4];
    }

    function meanAndStd(ch) {
      let m = 0;
      for (let i = 0; i < n; i++) m += ch[i];
      m /= n;
      let s = 0;
      for (let i = 0; i < n; i++) s += (ch[i] - m) ** 2;
      return { m, s: Math.sqrt(s / n) || 1 };
    }

    // Channel order matches Python's OpenCV BGR split: [B, G, R, T]
    // (cv.split on a BGR image yields b,g,r; data_vector = [b,g,r,th])
    const channels = [chB, chG, chR, chT];
    const centred  = channels.map((ch) => {
      const { m, s } = meanAndStd(ch);
      const f = new Float32Array(n);
      const denom = kind === 'fa' ? s : 1;
      for (let i = 0; i < n; i++) f[i] = (ch[i] - m) / denom;
      return f;
    });

    const cov = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = i; j < 4; j++) {
        let s = 0;
        const ci = centred[i], cj = centred[j];
        for (let k = 0; k < n; k++) s += ci[k] * cj[k];
        cov[i * 4 + j] = cov[j * 4 + i] = s / (n - 1);
      }
    }

    const { vectors } = jacobiEigen4(cov);

    const proj = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
    for (let k = 0; k < 3; k++) {
      const ev = vectors[k];
      for (let i = 0; i < n; i++) {
        proj[k][i] = centred[0][i] * ev[0] + centred[1][i] * ev[1]
                   + centred[2][i] * ev[2] + centred[3][i] * ev[3];
      }
    }

    // Deterministic sign (PCA only): sklearn PCA applies svd_flip — for each
    // component it flips the sign so the score with the largest magnitude is
    // positive. Without it the JS preview can appear light/dark-inverted vs the
    // real Pyodide output. (FA's FactorAnalysis sign convention differs → skip.)
    if (kind === 'pca') {
      for (let k = 0; k < 3; k++) {
        let mi = 0, mv = 0;
        for (let i = 0; i < n; i++) { const a = Math.abs(proj[k][i]); if (a > mv) { mv = a; mi = i; } }
        if (proj[k][mi] < 0) for (let i = 0; i < n; i++) proj[k][i] = -proj[k][i];
      }
    }

    // Per-component scaling for the individual component/factor CARDS (each shown
    // on its own, so its own min/max reads best).
    const [c1, c2, c3] = proj.map(normalizeToUint8);
    const img1   = greyImageData(c1, w, h);
    const img2   = greyImageData(c2, w, h);
    const img3   = greyImageData(c3, w, h);
    // OUTPUT colour balance: PCA's Python normalize() scales the whole 3-channel
    // composite by ONE global min/max (normalization.py) — per-component scaling
    // would distort the false-colour balance. FA's reference differs, so leave it
    // per-component there.
    let [oc1, oc2, oc3] = [c1, c2, c3];
    if (kind === 'pca') {
      let mn = Infinity, mx = -Infinity;
      for (const a of proj) for (let i = 0; i < a.length; i++) { const v = a[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
      const rng = (mx - mn) || 1;
      [oc1, oc2, oc3] = proj.map((a) => {
        const o = new Uint8Array(a.length);
        for (let i = 0; i < a.length; i++) o[i] = Math.max(0, Math.min(255, Math.floor((a[i] - mn) / rng * 255)));
        return o;
      });
    }
    const output = falseColourImageData(oc1, oc2, oc3, w, h);

    const imgChR = falseColourImageData(chR, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgChG = falseColourImageData(new Uint8Array(n), chG, new Uint8Array(n), w, h);
    const imgChB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), chB, w, h);
    const imgChT = greyImageData(chT, w, h);

    const isPca   = kind === 'pca';
    const opLabel = isPca ? 'pcaOp' : 'faOp';
    const opId    = isPca ? 'op-pca' : 'op-fa';
    const labels  = isPca ? ['PC1', 'PC2', 'PC3'] : ['F1', 'F2', 'F3'];
    const ids     = isPca ? ['pc1', 'pc2', 'pc3'] : ['f1', 'f2', 'f3'];
    const classes = isPca
      ? ['pipe-card--pc1', 'pipe-card--pc2', 'pipe-card--pc3']
      : ['pipe-card--f1',  'pipe-card--f2',  'pipe-card--f3'];
    const clrVars = isPca
      ? ['var(--clr-ch-pc1)', 'var(--clr-ch-pc2)', 'var(--clr-ch-pc3)']
      : ['var(--clr-ch-f1)',  'var(--clr-ch-f2)',  'var(--clr-ch-f3)'];
    const imgs    = [img1, img2, img3];

    return {
      stages: [
        {
          key: `${kind}-channels`,
          opNode: { id: 'op-split', labelKey: 'splitRgb' },
          cards: [
            { id: 'ch-r', imageData: imgChR, cardClass: 'pipe-card--r', desc: 'R' },
            { id: 'ch-g', imageData: imgChG, cardClass: 'pipe-card--g', desc: 'G' },
            { id: 'ch-b', imageData: imgChB, cardClass: 'pipe-card--b', desc: 'B' },
            { id: 'ch-t', imageData: imgChT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          key: `${kind}-proj`,
          opNode: { id: opId, labelKey: opLabel },
          cards: ids.map((id, k) => ({
            id, imageData: imgs[k], cardClass: classes[k], desc: labels[k],
          })),
        },
        {
          key: `${kind}-out`,
          opNode: { id: 'op-merge', labelKey: 'mergeOut' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out',
              desc: 'Output', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',  to: 'op-split', color: 'var(--clr-border)' },
        { from: 'src-lwir', to: 'ch-t',     color: 'var(--clr-ch-t)' },
        { from: 'op-split', to: 'ch-r',     color: 'var(--clr-ch-r)' },
        { from: 'op-split', to: 'ch-g',     color: 'var(--clr-ch-g)' },
        { from: 'op-split', to: 'ch-b',     color: 'var(--clr-ch-b)' },
        { from: 'ch-r',     to: opId,       color: 'var(--clr-ch-r)' },
        { from: 'ch-g',     to: opId,       color: 'var(--clr-ch-g)' },
        { from: 'ch-b',     to: opId,       color: 'var(--clr-ch-b)' },
        { from: 'ch-t',     to: opId,       color: 'var(--clr-ch-t)' },
        ...ids.map((id, k) => ({ from: opId,    to: id,         color: clrVars[k] })),
        ...ids.map((id, k) => ({ from: id,      to: 'op-merge', color: clrVars[k] })),
        { from: 'op-merge', to: 'out',      color: 'var(--clr-border)' },
      ],
      output,
    };
  }

  /* ════════════════════════════════════════════════════════════════
     ALPHA FUSION ALGORITHMS
  ════════════════════════════════════════════════════════════════ */

  function _computeAlphaFusion(rgbData, lwirData, w, h, mode) {
    const n   = w * h;
    const rgb = rgbData.data, lwir = lwirData.data;

    const chR = new Uint8Array(n), chG = new Uint8Array(n);
    const chB = new Uint8Array(n), chT = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      chR[i] = rgb[i * 4]; chG[i] = rgb[i * 4 + 1];
      chB[i] = rgb[i * 4 + 2]; chT[i] = lwir[i * 4];
    }

    const fGrey = new Float32Array(n);
    for (let i = 0; i < n; i++) fGrey[i] = 0.299 * chR[i] + 0.587 * chG[i] + 0.114 * chB[i];
    const fT    = new Float32Array(chT);

    let alpha;
    let opLabelKey;

    if (mode === 'sobel') {
      opLabelKey = 'sobelOp';
      alpha = buildAlphaMap(sobelMagnitude(fGrey, w, h), sobelMagnitude(fT, w, h));
    } else if (mode === 'ssim') {
      opLabelKey = 'localVarOp';
      alpha = buildAlphaMap(localVariance(fGrey, w, h, 2), localVariance(fT, w, h, 2));
    } else {
      // superpixel: 16×16 patches
      opLabelKey = 'patchStdOp';
      const PATCH = 16;
      alpha = new Float32Array(n);
      for (let py = 0; py < h; py += PATCH) {
        for (let px = 0; px < w; px += PATCH) {
          const pw = Math.min(PATCH, w - px), ph = Math.min(PATCH, h - py);
          const pn = pw * ph;
          let sumG = 0, sumT = 0;
          for (let dy = 0; dy < ph; dy++) {
            for (let dx = 0; dx < pw; dx++) {
              const idx = (py + dy) * w + (px + dx);
              sumG += fGrey[idx]; sumT += fT[idx];
            }
          }
          const mG = sumG / pn, mT = sumT / pn;
          let vG = 0, vT = 0;
          for (let dy = 0; dy < ph; dy++) {
            for (let dx = 0; dx < pw; dx++) {
              const idx = (py + dy) * w + (px + dx);
              vG += (fGrey[idx] - mG) ** 2;
              vT += (fT[idx] - mT) ** 2;
            }
          }
          const stdG = Math.sqrt(vG / pn), stdT = Math.sqrt(vT / pn);
          const a = stdG / (stdG + stdT + 1e-6);
          for (let dy = 0; dy < ph; dy++) {
            for (let dx = 0; dx < pw; dx++) {
              alpha[(py + dy) * w + (px + dx)] = a;
            }
          }
        }
      }
    }

    const imgAlpha = alphaHeatmap(alpha, w, h);
    const outR = alphaBlend(alpha, new Float32Array(chR), fT);
    const outG = alphaBlend(alpha, new Float32Array(chG), fT);
    const outB = alphaBlend(alpha, new Float32Array(chB), fT);
    const output = falseColourImageData(outR, outG, outB, w, h);

    const imgChR = falseColourImageData(chR, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgChG = falseColourImageData(new Uint8Array(n), chG, new Uint8Array(n), w, h);
    const imgChB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), chB, w, h);
    const imgChT = greyImageData(chT, w, h);

    return {
      stages: [
        {
          key: `${mode}-channels`,
          opNode: { id: 'op-split', labelKey: 'splitRgb' },
          cards: [
            { id: 'ch-r', imageData: imgChR, cardClass: 'pipe-card--r', desc: 'R' },
            { id: 'ch-g', imageData: imgChG, cardClass: 'pipe-card--g', desc: 'G' },
            { id: 'ch-b', imageData: imgChB, cardClass: 'pipe-card--b', desc: 'B' },
            { id: 'ch-t', imageData: imgChT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          key: `${mode}-alpha`,
          opNode: { id: 'op-alpha', labelKey: opLabelKey },
          cards: [
            { id: 'alpha', imageData: imgAlpha, cardClass: 'pipe-card--alpha', desc: 'α map' },
          ],
        },
        {
          key: `${mode}-out`,
          opNode: { id: 'op-blend', labelKey: 'alphaBlend' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out', desc: 'Output', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',  to: 'op-split', color: 'var(--clr-border)' },
        { from: 'src-lwir', to: 'ch-t',     color: 'var(--clr-ch-t)' },
        { from: 'op-split', to: 'ch-r',     color: 'var(--clr-ch-r)' },
        { from: 'op-split', to: 'ch-g',     color: 'var(--clr-ch-g)' },
        { from: 'op-split', to: 'ch-b',     color: 'var(--clr-ch-b)' },
        { from: 'ch-r',     to: 'op-alpha', color: 'var(--clr-ch-r)' },
        { from: 'ch-g',     to: 'op-alpha', color: 'var(--clr-ch-g)' },
        { from: 'ch-b',     to: 'op-alpha', color: 'var(--clr-ch-b)' },
        { from: 'ch-t',     to: 'op-alpha', color: 'var(--clr-ch-t)' },
        { from: 'op-alpha', to: 'alpha',    color: 'var(--clr-ch-alpha)' },
        { from: 'alpha',    to: 'op-blend', color: 'var(--clr-ch-alpha)' },
        { from: 'ch-r',     to: 'op-blend', color: 'var(--clr-ch-r)' },
        { from: 'ch-g',     to: 'op-blend', color: 'var(--clr-ch-g)' },
        { from: 'ch-b',     to: 'op-blend', color: 'var(--clr-ch-b)' },
        { from: 'ch-t',     to: 'op-blend', color: 'var(--clr-ch-t)' },
        { from: 'op-blend', to: 'out',      color: 'var(--clr-border)' },
      ],
      output,
    };
  }

  /* ════════════════════════════════════════════════════════════════
     PUBLIC WRAPPERS
  ════════════════════════════════════════════════════════════════ */

  function computeWavelet(rgbData, lwirData, w, h)    { return _computeWaveletFusion(rgbData, lwirData, w, h, 'avg'); }
  function computeWaveletMax(rgbData, lwirData, w, h) { return _computeWaveletFusion(rgbData, lwirData, w, h, 'max'); }
  function computeCurvelet(rgbData, lwirData, w, h)   { return _computeCurveletFusion(rgbData, lwirData, w, h, 'avg'); }
  function computeCurveletMax(rgbData, lwirData, w, h){ return _computeCurveletFusion(rgbData, lwirData, w, h, 'max'); }
  // Sobel-weighted fusion — faithful to py/fusion_methods/local_filter_fusion.py
  // combine_rgbt_sobel_weighted(visible, thermal, alpha=0.5):
  //   grad_thermal      = filters.sobel(thermal)            # Sobel on THERMAL only
  //   grad_thermal_norm = (g - g.min()) / (g.ptp() + 1e-8)  # min-max → [0,1] = g
  //   wT = alpha * grad_thermal_norm                        # thermal weight, alpha=0.5
  //   fused[...,c] = (1 - wT) * visible[...,c] + wT * thermal
  //   fused = normalize(fused)
  // The thermal edge map drives a SHARED weight wT that fans into every per-channel
  // blend (R,G,B) — exactly like RGBT's R×T/G×T/B×T and the wavelet per-channel fuse.
  function _computeSobelFusion(rgbData, lwirData, w, h) {
    const n   = w * h;
    const rgb = rgbData.data, lwir = lwirData.data;
    const ALPHA = 0.5;

    const chR = new Uint8Array(n), chG = new Uint8Array(n);
    const chB = new Uint8Array(n), chT = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      chR[i] = rgb[i * 4]; chG[i] = rgb[i * 4 + 1];
      chB[i] = rgb[i * 4 + 2]; chT[i] = lwir[i * 4];
    }
    const fT = new Float32Array(chT);

    // Sobel gradient magnitude of the THERMAL channel only (skimage filters.sobel(thermal)).
    const gradT = sobelMagnitude(fT, w, h);

    // min-max normalize the thermal gradient to [0,1] = g  (grad_thermal_norm).
    let gmin = Infinity, gmax = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = gradT[i];
      if (v < gmin) gmin = v;
      if (v > gmax) gmax = v;
    }
    const grange = (gmax - gmin) + 1e-8;
    const g  = new Float32Array(n);   // g  ∈ [0,1]
    const wT = new Float32Array(n);   // wT = alpha * g  ∈ [0, alpha]  (thermal weight)
    for (let i = 0; i < n; i++) {
      g[i]  = (gradT[i] - gmin) / grange;
      wT[i] = ALPHA * g[i];
    }

    // Per-channel weighted blend: fused_c = (1 - wT)*RGB_c + wT*T.
    // alphaBlend(a, src, T) = a*src + (1-a)*T, so pass a = (1 - wT) → (1-wT)*RGB + wT*T.
    const aRgb = new Float32Array(n);
    for (let i = 0; i < n; i++) aRgb[i] = 1 - wT[i];
    const outR = alphaBlend(aRgb, new Float32Array(chR), fT);
    const outG = alphaBlend(aRgb, new Float32Array(chG), fT);
    const outB = alphaBlend(aRgb, new Float32Array(chB), fT);
    const output = falseColourImageData(outR, outG, outB, w, h);

    // ── intermediate images ──────────────────────────────────────────
    const imgChR = falseColourImageData(chR, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgChG = falseColourImageData(new Uint8Array(n), chG, new Uint8Array(n), w, h);
    const imgChB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), chB, w, h);
    const imgChT = greyImageData(chT, w, h);

    // Thermal edge map |∇T|: the raw Sobel magnitude is faint, so amplify it for
    // visibility (magnitude + gamma<1, same treatment as wavelet detail subbands)
    // and announce it with a '*'. grey-on-thermal.
    const imgEdge = greyImageData(_enhanceDetail(gradT), w, h);
    // Normalized weight g ∈ [0,1] (the actual grad_thermal_norm), shown grey as-is.
    const imgG = greyImageData(normalizeToUint8(g), w, h);
    // Thermal weight map wT = alpha·g via heat-map: high (red) = "more thermal".
    const imgW = alphaHeatmap(wT, w, h);

    // Per-channel fused cards (single-channel false colour) — the blend each RGB
    // channel undergoes against the shared thermal weight.
    const imgFusR = falseColourImageData(outR, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgFusG = falseColourImageData(new Uint8Array(n), outG, new Uint8Array(n), w, h);
    const imgFusB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), outB, w, h);

    return {
      stages: [
        {
          key: 'sobel-channels',
          opNode: { id: 'op-split', labelKey: 'splitRgb' },
          cards: [
            { id: 'ch-r', imageData: imgChR, cardClass: 'pipe-card--r', desc: 'R' },
            { id: 'ch-g', imageData: imgChG, cardClass: 'pipe-card--g', desc: 'G' },
            { id: 'ch-b', imageData: imgChB, cardClass: 'pipe-card--b', desc: 'B' },
            { id: 'ch-t', imageData: imgChT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          // Sobel runs on the THERMAL channel ONLY → edge magnitude |∇T|, then
          // min-max normalize → g. Both fed exclusively from T.
          key: 'sobel-edge',
          opNode: { id: 'op-sobel', labelKey: 'sobelEdgeOp' },
          cards: [
            { id: 'edge', imageData: imgEdge, cardClass: 'pipe-card--t', desc: '|∇T|*' },
            { id: 'gnorm', imageData: imgG,   cardClass: 'pipe-card--t', desc: 'g = norm|∇T|' },
          ],
        },
        {
          // Thermal weight wT = α·g (α=0.5). Heat-map: red = more thermal.
          key: 'sobel-weight',
          opNode: null,
          cards: [
            { id: 'wt', imageData: imgW, cardClass: 'pipe-card--alpha', desc: 'wₜ = α·g' },
          ],
        },
        {
          // No single op-node: every RGB channel is blended against the SHARED
          // thermal weight wT independently (like RGBT R×T/G×T/B×T), so wT and T
          // fan into each fused card rather than funnelling through one node.
          key: 'sobel-fuse',
          opNode: null,
          cards: [
            { id: 'fus-r', imageData: imgFusR, cardClass: 'pipe-card--r', desc: '(1-wₜ)R + wₜT' },
            { id: 'fus-g', imageData: imgFusG, cardClass: 'pipe-card--g', desc: '(1-wₜ)G + wₜT' },
            { id: 'fus-b', imageData: imgFusB, cardClass: 'pipe-card--b', desc: '(1-wₜ)B + wₜT' },
          ],
        },
        {
          key: 'sobel-out',
          opNode: { id: 'op-merge', labelKey: 'mergeRgb' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out', desc: 'Output', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',  to: 'op-split', color: 'var(--clr-border)' },
        { from: 'src-lwir', to: 'ch-t',     color: 'var(--clr-ch-t)' },
        { from: 'op-split', to: 'ch-r',     color: 'var(--clr-ch-r)' },
        { from: 'op-split', to: 'ch-g',     color: 'var(--clr-ch-g)' },
        { from: 'op-split', to: 'ch-b',     color: 'var(--clr-ch-b)' },
        // Sobel is computed from the thermal channel only.
        { from: 'ch-t',     to: 'op-sobel', color: 'var(--clr-ch-t)' },
        { from: 'op-sobel', to: 'edge',     color: 'var(--clr-ch-t)' },
        { from: 'op-sobel', to: 'gnorm',    color: 'var(--clr-ch-t)' },
        // Normalized gradient g → thermal weight wT.
        { from: 'gnorm',    to: 'wt',       color: 'var(--clr-ch-t)' },
        // Each RGB channel + the shared weight wT + thermal fan into each fused card.
        { from: 'ch-r',     to: 'fus-r',    color: 'var(--clr-ch-r)' },
        { from: 'ch-g',     to: 'fus-g',    color: 'var(--clr-ch-g)' },
        { from: 'ch-b',     to: 'fus-b',    color: 'var(--clr-ch-b)' },
        { from: 'wt',       to: 'fus-r',    color: 'var(--clr-ch-alpha)' },
        { from: 'wt',       to: 'fus-g',    color: 'var(--clr-ch-alpha)' },
        { from: 'wt',       to: 'fus-b',    color: 'var(--clr-ch-alpha)' },
        { from: 'ch-t',     to: 'fus-r',    color: 'var(--clr-ch-t)' },
        { from: 'ch-t',     to: 'fus-g',    color: 'var(--clr-ch-t)' },
        { from: 'ch-t',     to: 'fus-b',    color: 'var(--clr-ch-t)' },
        // Merge fused channels → output.
        { from: 'fus-r',    to: 'op-merge', color: 'var(--clr-ch-r)' },
        { from: 'fus-g',    to: 'op-merge', color: 'var(--clr-ch-g)' },
        { from: 'fus-b',    to: 'op-merge', color: 'var(--clr-ch-b)' },
        { from: 'op-merge', to: 'out',      color: 'var(--clr-border)' },
      ],
      output,
    };
  }

  function _computeSsimFusion(rgbData, lwirData, w, h) {
    const n   = w * h;
    const rgb = rgbData.data, lwir = lwirData.data;

    // ── Split RGB + thermal (thermal shared across all three channels) ──
    const chR = new Uint8Array(n), chG = new Uint8Array(n);
    const chB = new Uint8Array(n), chT = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      chR[i] = rgb[i * 4]; chG[i] = rgb[i * 4 + 1];
      chB[i] = rgb[i * 4 + 2]; chT[i] = lwir[i * 4];
    }
    const fR = new Float32Array(chR), fG = new Float32Array(chG);
    const fB = new Float32Array(chB), fT = new Float32Array(chT);

    // ── Windowed SSIM map of one RGB channel vs the SAME thermal ──
    // Mirrors skimage.metrics.structural_similarity(full=True, data_range=255):
    // local means/variances/covariance over a uniform window (box filter), then
    //   ssim = ((2·μx·μy + C1)(2·σxy + C2)) / ((μx² + μy² + C1)(σx² + σy² + C2))
    // with C1=(0.01·255)², C2=(0.03·255)². win_size=11 (radius 5), clamped odd≥3.
    const L  = 255;
    const C1 = (0.01 * L) * (0.01 * L);
    const C2 = (0.03 * L) * (0.03 * L);
    let win = Math.min(11, w, h);
    if (win % 2 === 0) win -= 1;
    win = Math.max(win, 3);
    const radius = (win - 1) / 2;
    const boxKernel = new Float32Array(win).fill(1 / win); // separable uniform window

    // Precompute thermal local stats (shared denominator term across channels).
    const muT  = separableConvolve(fT, w, h, boxKernel);
    const fTsq = new Float32Array(n);
    for (let i = 0; i < n; i++) fTsq[i] = fT[i] * fT[i];
    const muTsq = separableConvolve(fTsq, w, h, boxKernel);

    function ssimMapVsThermal(fX) {
      const muX  = separableConvolve(fX, w, h, boxKernel);
      const fXsq = new Float32Array(n);
      const fXT  = new Float32Array(n);
      for (let i = 0; i < n; i++) { fXsq[i] = fX[i] * fX[i]; fXT[i] = fX[i] * fT[i]; }
      const muXsq = separableConvolve(fXsq, w, h, boxKernel);
      const muXT  = separableConvolve(fXT,  w, h, boxKernel);
      const map = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const mx = muX[i], my = muT[i];
        const vx  = Math.max(0, muXsq[i] - mx * mx);
        const vy  = Math.max(0, muTsq[i] - my * my);
        const cxy = muXT[i] - mx * my;
        const num = (2 * mx * my + C1) * (2 * cxy + C2);
        const den = (mx * mx + my * my + C1) * (vx + vy + C2);
        map[i] = num / den;
      }
      return map;
    }

    // ── Smooth (gaussian σ=2) + min-max normalize → per-channel weight in [0,1] ──
    const gk = gaussianKernel1D(2.0);
    function smoothNorm(map) {
      const sm = separableConvolve(map, w, h, gk);
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < n; i++) { if (sm[i] < mn) mn = sm[i]; if (sm[i] > mx) mx = sm[i]; }
      const range = (mx - mn) + 1e-8;
      const wmap = new Float32Array(n);
      for (let i = 0; i < n; i++) wmap[i] = (sm[i] - mn) / range;
      return wmap;
    }

    const ssimR = ssimMapVsThermal(fR);
    const ssimG = ssimMapVsThermal(fG);
    const ssimB = ssimMapVsThermal(fB);

    const wR = smoothNorm(ssimR);
    const wG = smoothNorm(ssimG);
    const wB = smoothNorm(ssimB);

    // ── v2 blend: fused_i = w_i·RGB_i + (1−w_i)·T (high SSIM ⇒ keep RGB) ──
    const fusedR = new Float32Array(n), fusedG = new Float32Array(n), fusedB = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      fusedR[i] = wR[i] * fR[i] + (1 - wR[i]) * fT[i];
      fusedG[i] = wG[i] * fG[i] + (1 - wG[i]) * fT[i];
      fusedB[i] = wB[i] * fB[i] + (1 - wB[i]) * fT[i];
    }
    const outR = normalizeToUint8(fusedR);
    const outG = normalizeToUint8(fusedG);
    const outB = normalizeToUint8(fusedB);
    const output = falseColourImageData(outR, outG, outB, w, h);

    // ── Visualisations ──
    // SSIM maps are faint/centred near 1: realise via |·| + gamma<1 lift so the
    // structure becomes perceptible (announced with '*'), like the wavelet mosaics.
    function enhanceMap(map) {
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < n; i++) { if (map[i] < mn) mn = map[i]; if (map[i] > mx) mx = map[i]; }
      const range = (mx - mn) + 1e-8;
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) out[i] = Math.round(255 * Math.pow((map[i] - mn) / range, 0.6));
      return out;
    }
    function weightToByte(wmap) { // already normalized to [0,1]
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) out[i] = Math.round(255 * wmap[i]);
      return out;
    }

    const imgChR = falseColourImageData(chR, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgChG = falseColourImageData(new Uint8Array(n), chG, new Uint8Array(n), w, h);
    const imgChB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), chB, w, h);
    const imgChT = greyImageData(chT, w, h);

    // SSIM maps tinted per channel (faint → enhanced, announced with '*').
    const imgSsimR = falseColourImageData(enhanceMap(ssimR), new Uint8Array(n), new Uint8Array(n), w, h);
    const imgSsimG = falseColourImageData(new Uint8Array(n), enhanceMap(ssimG), new Uint8Array(n), w, h);
    const imgSsimB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), enhanceMap(ssimB), w, h);

    // Smoothed + normalized weight maps as α-style heatmaps (0=blue→1=red).
    const imgWR = alphaHeatmap(wR, w, h);
    const imgWG = alphaHeatmap(wG, w, h);
    const imgWB = alphaHeatmap(wB, w, h);

    // Fused per-channel previews (each = w·RGB + (1−w)·T), tinted per channel.
    const imgFusR = falseColourImageData(outR, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgFusG = falseColourImageData(new Uint8Array(n), outG, new Uint8Array(n), w, h);
    const imgFusB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), outB, w, h);

    return {
      stages: [
        {
          key: 'ssim-channels',
          opNode: { id: 'op-split', labelKey: 'splitRgb' },
          cards: [
            { id: 'ch-r', imageData: imgChR, cardClass: 'pipe-card--r', desc: 'R' },
            { id: 'ch-g', imageData: imgChG, cardClass: 'pipe-card--g', desc: 'G' },
            { id: 'ch-b', imageData: imgChB, cardClass: 'pipe-card--b', desc: 'B' },
            { id: 'ch-t', imageData: imgChT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          // Each RGB channel's windowed SSIM is computed against the SAME thermal,
          // so thermal fans into every SSIM card (no single funnel node).
          key: 'ssim-maps',
          opNode: { id: 'op-ssim', labelKey: 'ssimOp' },
          cards: [
            { id: 'ssim-r', imageData: imgSsimR, cardClass: 'pipe-card--r', desc: 'SSIM(R,T)*' },
            { id: 'ssim-g', imageData: imgSsimG, cardClass: 'pipe-card--g', desc: 'SSIM(G,T)*' },
            { id: 'ssim-b', imageData: imgSsimB, cardClass: 'pipe-card--b', desc: 'SSIM(B,T)*' },
          ],
        },
        {
          key: 'ssim-weights',
          opNode: { id: 'op-ssim-norm', labelKey: 'ssimNormOp' },
          cards: [
            { id: 'w-r', imageData: imgWR, cardClass: 'pipe-card--r', desc: 'wR' },
            { id: 'w-g', imageData: imgWG, cardClass: 'pipe-card--g', desc: 'wG' },
            { id: 'w-b', imageData: imgWB, cardClass: 'pipe-card--b', desc: 'wB' },
          ],
        },
        {
          // Explicit per-channel blend (like RGBT's R×T/G×T/B×T): each weight + the
          // SHARED thermal fan into their fused card; no single op-node emits all 3.
          key: 'ssim-blend',
          opNode: null,
          cards: [
            { id: 'fus-r', imageData: imgFusR, cardClass: 'pipe-card--r', desc: 'wR·R+(1−wR)·T' },
            { id: 'fus-g', imageData: imgFusG, cardClass: 'pipe-card--g', desc: 'wG·G+(1−wG)·T' },
            { id: 'fus-b', imageData: imgFusB, cardClass: 'pipe-card--b', desc: 'wB·B+(1−wB)·T' },
          ],
        },
        {
          key: 'ssim-out',
          opNode: { id: 'op-merge', labelKey: 'mergeRgb' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out', desc: 'Output', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',  to: 'op-split', color: 'var(--clr-border)' },
        { from: 'src-lwir', to: 'ch-t',     color: 'var(--clr-ch-t)' },
        { from: 'op-split', to: 'ch-r',     color: 'var(--clr-ch-r)' },
        { from: 'op-split', to: 'ch-g',     color: 'var(--clr-ch-g)' },
        { from: 'op-split', to: 'ch-b',     color: 'var(--clr-ch-b)' },
        // RGB channels + shared thermal → windowed SSIM op.
        { from: 'ch-r', to: 'op-ssim', color: 'var(--clr-ch-r)' },
        { from: 'ch-g', to: 'op-ssim', color: 'var(--clr-ch-g)' },
        { from: 'ch-b', to: 'op-ssim', color: 'var(--clr-ch-b)' },
        { from: 'ch-t', to: 'op-ssim', color: 'var(--clr-ch-t)' },
        { from: 'op-ssim', to: 'ssim-r', color: 'var(--clr-ch-r)' },
        { from: 'op-ssim', to: 'ssim-g', color: 'var(--clr-ch-g)' },
        { from: 'op-ssim', to: 'ssim-b', color: 'var(--clr-ch-b)' },
        // SSIM maps → gaussian-smooth + min-max normalize → weight maps.
        { from: 'ssim-r', to: 'op-ssim-norm', color: 'var(--clr-ch-r)' },
        { from: 'ssim-g', to: 'op-ssim-norm', color: 'var(--clr-ch-g)' },
        { from: 'ssim-b', to: 'op-ssim-norm', color: 'var(--clr-ch-b)' },
        { from: 'op-ssim-norm', to: 'w-r', color: 'var(--clr-ch-r)' },
        { from: 'op-ssim-norm', to: 'w-g', color: 'var(--clr-ch-g)' },
        { from: 'op-ssim-norm', to: 'w-b', color: 'var(--clr-ch-b)' },
        // Per-channel blend: weight + original channel + SHARED thermal → fused card.
        { from: 'w-r',  to: 'fus-r', color: 'var(--clr-ch-r)' },
        { from: 'w-g',  to: 'fus-g', color: 'var(--clr-ch-g)' },
        { from: 'w-b',  to: 'fus-b', color: 'var(--clr-ch-b)' },
        { from: 'ch-r', to: 'fus-r', color: 'var(--clr-ch-r)' },
        { from: 'ch-g', to: 'fus-g', color: 'var(--clr-ch-g)' },
        { from: 'ch-b', to: 'fus-b', color: 'var(--clr-ch-b)' },
        { from: 'ch-t', to: 'fus-r', color: 'var(--clr-ch-t)' },
        { from: 'ch-t', to: 'fus-g', color: 'var(--clr-ch-t)' },
        { from: 'ch-t', to: 'fus-b', color: 'var(--clr-ch-t)' },
        { from: 'fus-r', to: 'op-merge', color: 'var(--clr-ch-r)' },
        { from: 'fus-g', to: 'op-merge', color: 'var(--clr-ch-g)' },
        { from: 'fus-b', to: 'op-merge', color: 'var(--clr-ch-b)' },
        { from: 'op-merge', to: 'out', color: 'var(--clr-border)' },
      ],
      output,
    };
  }

  /* ════════════════════════════════════════════════════════════════
     SUPERPIXEL FUSION  (combine_rgbt_superpixel)
     SLIC-like superpixels on the 4-ch RGB+T stack → per-segment std of
     visible (mean over R,G,B) and of thermal → piecewise-constant maps →
     weight = gaussian_blur(minmaxnorm(sigmoid(σ_vis − σ_th))) →
     per channel: weight·visible + (1−weight)·thermal → normalize.
  ════════════════════════════════════════════════════════════════ */

  // SLIC-like content-adaptive segmentation. A true skimage SLIC is too heavy
  // for plain JS, so we run a lightweight grid-seeded k-means in 6-D (R,G,B,T +
  // x,y) — the SAME model SLIC uses — with a few iterations and a local 2·step
  // search window (SLIC's standard optimisation). Produces compact, content-
  // adaptive regions that follow image structure (NOT a fixed 16×16 grid).
  function _slicSuperpixels(chR, chG, chB, chT, w, h, nSegments, compactness, iters) {
    const n = w * h;
    const step = Math.max(2, Math.round(Math.sqrt((w * h) / nSegments)));
    const cols = Math.max(1, Math.round(w / step));
    const rows = Math.max(1, Math.round(h / step));
    const K = cols * rows;

    // Cluster centres: [R,G,B,T,x,y]; seeded on a regular grid.
    const cx = new Float32Array(K), cy = new Float32Array(K);
    const cR = new Float32Array(K), cG = new Float32Array(K);
    const cB = new Float32Array(K), cT = new Float32Array(K);
    for (let ry = 0; ry < rows; ry++) {
      for (let rx = 0; rx < cols; rx++) {
        const k = ry * cols + rx;
        const px = Math.min(w - 1, Math.round((rx + 0.5) * w / cols));
        const py = Math.min(h - 1, Math.round((ry + 0.5) * h / rows));
        const idx = py * w + px;
        cx[k] = px; cy[k] = py;
        cR[k] = chR[idx]; cG[k] = chG[idx]; cB[k] = chB[idx]; cT[k] = chT[idx];
      }
    }

    const labels = new Int32Array(n).fill(-1);
    // SLIC distance: colour/thermal distance + (compactness/step)²·spatial dist.
    const invS2 = (compactness * compactness) / (step * step);
    const dist = new Float32Array(n);

    for (let it = 0; it < iters; it++) {
      dist.fill(Infinity);
      for (let k = 0; k < K; k++) {
        const x0 = Math.max(0, Math.round(cx[k]) - step), x1 = Math.min(w - 1, Math.round(cx[k]) + step);
        const y0 = Math.max(0, Math.round(cy[k]) - step), y1 = Math.min(h - 1, Math.round(cy[k]) + step);
        for (let y = y0; y <= y1; y++) {
          for (let x = x0; x <= x1; x++) {
            const idx = y * w + x;
            const dr = chR[idx] - cR[k], dg = chG[idx] - cG[k];
            const db = chB[idx] - cB[k], dt = chT[idx] - cT[k];
            const dc = dr * dr + dg * dg + db * db + dt * dt;
            const ddx = x - cx[k], ddy = y - cy[k];
            const ds = ddx * ddx + ddy * ddy;
            const D = dc + invS2 * ds;
            if (D < dist[idx]) { dist[idx] = D; labels[idx] = k; }
          }
        }
      }
      // Recompute centres as the mean of each cluster's members.
      const sumR = new Float64Array(K), sumG = new Float64Array(K), sumB = new Float64Array(K);
      const sumT = new Float64Array(K), sumX = new Float64Array(K), sumY = new Float64Array(K);
      const cnt = new Int32Array(K);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = y * w + x; const k = labels[idx];
          if (k < 0) continue;
          sumR[k] += chR[idx]; sumG[k] += chG[idx]; sumB[k] += chB[idx]; sumT[k] += chT[idx];
          sumX[k] += x; sumY[k] += y; cnt[k]++;
        }
      }
      for (let k = 0; k < K; k++) {
        if (!cnt[k]) continue;
        cR[k] = sumR[k] / cnt[k]; cG[k] = sumG[k] / cnt[k];
        cB[k] = sumB[k] / cnt[k]; cT[k] = sumT[k] / cnt[k];
        cx[k] = sumX[k] / cnt[k]; cy[k] = sumY[k] / cnt[k];
      }
    }
    // Pixels never claimed by any window → nearest grid cell (guarantees a label).
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (labels[idx] < 0) {
          const rx = Math.min(cols - 1, Math.floor(x * cols / w));
          const ry = Math.min(rows - 1, Math.floor(y * rows / h));
          labels[idx] = ry * cols + rx;
        }
      }
    }
    return { labels, K };
  }

  // Segmentation view: faded base image with bright superpixel boundaries
  // (a pixel sits on a boundary when its right/down neighbour has a different label).
  function _superpixelOverlay(chR, chG, chB, labels, w, h) {
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        const lab = labels[idx];
        const right = x + 1 < w ? labels[idx + 1] : lab;
        const down  = y + 1 < h ? labels[idx + w] : lab;
        if (right !== lab || down !== lab) {
          data[idx * 4] = 255; data[idx * 4 + 1] = 255; data[idx * 4 + 2] = 0; // yellow edges
        } else {
          data[idx * 4]     = Math.round(chR[idx] * 0.55);
          data[idx * 4 + 1] = Math.round(chG[idx] * 0.55);
          data[idx * 4 + 2] = Math.round(chB[idx] * 0.55);
        }
        data[idx * 4 + 3] = 255;
      }
    }
    return new ImageData(data, w, h);
  }

  // Visibility-enhance a faint (per-segment std) map: scale to its own max and
  // apply gamma<1 so low-std regions stay perceptible. Announced with '*' on the
  // card label (like subbandMosaic's amplified detail subbands). tint picks the
  // channel-family colour: 'vis' → yellow (intensity), 't' → thermal-red.
  function _enhanceStdMap(arr, tint, w, h) {
    let mx = 0;
    for (let i = 0; i < arr.length; i++) if (arr[i] > mx) mx = arr[i];
    const inv = mx > 0 ? 1 / mx : 0;
    const u = new Uint8Array(arr.length);
    for (let i = 0; i < arr.length; i++) u[i] = Math.round(255 * Math.pow(arr[i] * inv, 0.6));
    const z = new Uint8Array(arr.length);
    return tint === 'vis' ? falseColourImageData(u, u, z, w, h)
         : tint === 't'   ? falseColourImageData(u, z, z, w, h)
         :                  greyImageData(u, w, h);
  }

  function _computeSuperpixelFusion(rgbData, lwirData, w, h) {
    const n   = w * h;
    const rgb = rgbData.data, lwir = lwirData.data;

    const chR = new Uint8Array(n), chG = new Uint8Array(n);
    const chB = new Uint8Array(n), chT = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      chR[i] = rgb[i * 4]; chG[i] = rgb[i * 4 + 1];
      chB[i] = rgb[i * 4 + 2]; chT[i] = lwir[i * 4];
    }

    // 1. SLIC-like superpixels on the 4-channel RGB+T stack (~300 segments).
    const { labels, K } = _slicSuperpixels(chR, chG, chB, chT, w, h, 300, 10, 4);

    // 2. Per-segment std → piecewise-constant deviation maps.
    //    σ_visible = mean over R,G,B of the per-channel std inside the segment;
    //    σ_thermal = std of thermal inside the segment.
    const cnt = new Int32Array(K);
    const sR = new Float64Array(K), sG = new Float64Array(K), sB = new Float64Array(K), sT = new Float64Array(K);
    const sR2 = new Float64Array(K), sG2 = new Float64Array(K), sB2 = new Float64Array(K), sT2 = new Float64Array(K);
    for (let i = 0; i < n; i++) {
      const k = labels[i];
      const r = chR[i], g = chG[i], b = chB[i], t = chT[i];
      cnt[k]++;
      sR[k] += r; sG[k] += g; sB[k] += b; sT[k] += t;
      sR2[k] += r * r; sG2[k] += g * g; sB2[k] += b * b; sT2[k] += t * t;
    }
    const stdVisSeg = new Float32Array(K), stdThSeg = new Float32Array(K);
    const sd = (sum, sum2, c) => Math.sqrt(Math.max(0, sum2 / c - (sum / c) ** 2));
    for (let k = 0; k < K; k++) {
      const c = cnt[k] || 1;
      stdVisSeg[k] = (sd(sR[k], sR2[k], c) + sd(sG[k], sG2[k], c) + sd(sB[k], sB2[k], c)) / 3;
      stdThSeg[k]  = sd(sT[k], sT2[k], c);
    }

    const devVis = new Float32Array(n), devTh = new Float32Array(n);
    for (let i = 0; i < n; i++) { devVis[i] = stdVisSeg[labels[i]]; devTh[i] = stdThSeg[labels[i]]; }

    // 3. Weight = gaussian_blur( minmaxnorm( sigmoid(σ_vis − σ_th) ) ), σ=2.
    const raw = new Float32Array(n);
    for (let i = 0; i < n; i++) raw[i] = 1 / (1 + Math.exp(-(devVis[i] - devTh[i])));
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < n; i++) { if (raw[i] < mn) mn = raw[i]; if (raw[i] > mx) mx = raw[i]; }
    const span = mx - mn + 1e-8;
    for (let i = 0; i < n; i++) raw[i] = (raw[i] - mn) / span;
    const mask = separableConvolve(raw, w, h, gaussianKernel1D(2.0));

    // 4. Per-channel blend: fused[c] = weight·visible[c] + (1−weight)·thermal,
    //    then normalize each channel to [0,255].
    const fR = new Float32Array(n), fG = new Float32Array(n), fB = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const m = mask[i], im = 1 - m, t = chT[i];
      fR[i] = m * chR[i] + im * t;
      fG[i] = m * chG[i] + im * t;
      fB[i] = m * chB[i] + im * t;
    }
    const outR = normalizeToUint8(fR), outG = normalizeToUint8(fG), outB = normalizeToUint8(fB);

    // ── intermediate images the algorithm actually uses ──
    const imgChR = falseColourImageData(chR, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgChG = falseColourImageData(new Uint8Array(n), chG, new Uint8Array(n), w, h);
    const imgChB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), chB, w, h);
    const imgChT = greyImageData(chT, w, h);

    const imgSeg    = _superpixelOverlay(chR, chG, chB, labels, w, h);
    const imgDevVis = _enhanceStdMap(devVis, 'vis', w, h);   // faint → realzado (*)
    const imgDevTh  = _enhanceStdMap(devTh, 't', w, h);      // faint → realzado (*)
    const imgWeight = alphaHeatmap(mask, w, h);              // weight∈[0,1] heat-map

    const imgFusR = falseColourImageData(outR, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgFusG = falseColourImageData(new Uint8Array(n), outG, new Uint8Array(n), w, h);
    const imgFusB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), outB, w, h);

    const output = falseColourImageData(outR, outG, outB, w, h);

    return {
      stages: [
        {
          key: 'sp-channels',
          opNode: { id: 'op-split', labelKey: 'splitRgb' },
          cards: [
            { id: 'ch-r', imageData: imgChR, cardClass: 'pipe-card--r', desc: 'R' },
            { id: 'ch-g', imageData: imgChG, cardClass: 'pipe-card--g', desc: 'G' },
            { id: 'ch-b', imageData: imgChB, cardClass: 'pipe-card--b', desc: 'B' },
            { id: 'ch-t', imageData: imgChT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          // SLIC runs on the joint 4-channel RGB+T stack → all four channels feed it.
          key: 'sp-seg',
          opNode: { id: 'op-slic', labelKey: 'slicOp' },
          cards: [
            { id: 'seg', imageData: imgSeg, cardClass: 'pipe-card--out', desc: 'Superpixels' },
          ],
        },
        {
          // Per-segment std of the visible (mean R,G,B) and of the thermal.
          key: 'sp-dev',
          opNode: { id: 'op-regstd', labelKey: 'regionStdOp' },
          cards: [
            { id: 'dev-vis', imageData: imgDevVis, cardClass: 'pipe-card--v', desc: 'σ visible *' },
            { id: 'dev-th',  imageData: imgDevTh,  cardClass: 'pipe-card--t', desc: 'σ thermal *', separateBefore: true },
          ],
        },
        {
          key: 'sp-weight',
          opNode: { id: 'op-weight', labelKey: 'spWeightOp' },
          cards: [
            { id: 'weight', imageData: imgWeight, cardClass: 'pipe-card--alpha', desc: 'weight map' },
          ],
        },
        {
          // No single op-node: each channel is blended with the weight map and the
          // SHARED thermal independently (like RGBT's R×T/G×T/B×T and wavelet's
          // per-channel fusion), so weight + thermal fan into every fused card.
          key: 'sp-blend',
          opNode: null,
          cards: [
            { id: 'fus-r', imageData: imgFusR, cardClass: 'pipe-card--r', desc: 'w·R+(1−w)·T' },
            { id: 'fus-g', imageData: imgFusG, cardClass: 'pipe-card--g', desc: 'w·G+(1−w)·T' },
            { id: 'fus-b', imageData: imgFusB, cardClass: 'pipe-card--b', desc: 'w·B+(1−w)·T' },
          ],
        },
        {
          key: 'sp-out',
          opNode: { id: 'op-merge', labelKey: 'mergeRgb' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out', desc: 'Output', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',  to: 'op-split', color: 'var(--clr-border)' },
        { from: 'src-lwir', to: 'ch-t',     color: 'var(--clr-ch-t)' },
        { from: 'op-split', to: 'ch-r',     color: 'var(--clr-ch-r)' },
        { from: 'op-split', to: 'ch-g',     color: 'var(--clr-ch-g)' },
        { from: 'op-split', to: 'ch-b',     color: 'var(--clr-ch-b)' },
        { from: 'ch-r', to: 'op-slic', color: 'var(--clr-ch-r)' },
        { from: 'ch-g', to: 'op-slic', color: 'var(--clr-ch-g)' },
        { from: 'ch-b', to: 'op-slic', color: 'var(--clr-ch-b)' },
        { from: 'ch-t', to: 'op-slic', color: 'var(--clr-ch-t)' },
        { from: 'op-slic', to: 'seg', color: 'var(--clr-border)' },
        { from: 'seg', to: 'op-regstd', color: 'var(--clr-border)' },
        { from: 'ch-r', to: 'op-regstd', color: 'var(--clr-ch-r)' },
        { from: 'ch-g', to: 'op-regstd', color: 'var(--clr-ch-g)' },
        { from: 'ch-b', to: 'op-regstd', color: 'var(--clr-ch-b)' },
        { from: 'ch-t', to: 'op-regstd', color: 'var(--clr-ch-t)' },
        { from: 'op-regstd', to: 'dev-vis', color: 'var(--clr-ch-v)' },
        { from: 'op-regstd', to: 'dev-th',  color: 'var(--clr-ch-t)' },
        { from: 'dev-vis', to: 'op-weight', color: 'var(--clr-ch-v)' },
        { from: 'dev-th',  to: 'op-weight', color: 'var(--clr-ch-t)' },
        { from: 'op-weight', to: 'weight', color: 'var(--clr-ch-alpha)' },
        { from: 'weight', to: 'fus-r', color: 'var(--clr-ch-alpha)' },
        { from: 'weight', to: 'fus-g', color: 'var(--clr-ch-alpha)' },
        { from: 'weight', to: 'fus-b', color: 'var(--clr-ch-alpha)' },
        { from: 'ch-r', to: 'fus-r', color: 'var(--clr-ch-r)' },
        { from: 'ch-g', to: 'fus-g', color: 'var(--clr-ch-g)' },
        { from: 'ch-b', to: 'fus-b', color: 'var(--clr-ch-b)' },
        { from: 'ch-t', to: 'fus-r', color: 'var(--clr-ch-t)' },
        { from: 'ch-t', to: 'fus-g', color: 'var(--clr-ch-t)' },
        { from: 'ch-t', to: 'fus-b', color: 'var(--clr-ch-t)' },
        { from: 'fus-r', to: 'op-merge', color: 'var(--clr-ch-r)' },
        { from: 'fus-g', to: 'op-merge', color: 'var(--clr-ch-g)' },
        { from: 'fus-b', to: 'op-merge', color: 'var(--clr-ch-b)' },
        { from: 'op-merge', to: 'out', color: 'var(--clr-border)' },
      ],
      output,
    };
  }

  // FA fusion pipeline diagram. Faithful (in structure) to the vendored Python
  // combine_rgbt_fa_to3ch: stack [B,G,R,T] -> FactorAnalysis(n_components=3)
  // -> 3 latent factor score maps -> normalize -> false-colour output. The
  // generative FA model is  X ≈ Z·Wᵀ + μ + ε  (ε = per-feature Gaussian noise),
  // so on top of the 3 FACTOR images we ALSO surface, per factor, the residual
  // the factor leaves unexplained across the 4 input channels — the estimated
  // error/noise map the user asked for. (The final OUTPUT image is replaced by
  // Pyodide; the JS only needs to be faithful in STRUCTURE + qualitatively right.)
  function _computeFaFusion(rgbData, lwirData, w, h) {
    const n   = w * h;
    const rgb = rgbData.data, lwir = lwirData.data;

    const chR = new Uint8Array(n), chG = new Uint8Array(n);
    const chB = new Uint8Array(n), chT = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      chR[i] = rgb[i * 4]; chG[i] = rgb[i * 4 + 1];
      chB[i] = rgb[i * 4 + 2]; chT[i] = lwir[i * 4];
    }

    function meanAndStd(ch) {
      let m = 0;
      for (let i = 0; i < n; i++) m += ch[i];
      m /= n;
      let s = 0;
      for (let i = 0; i < n; i++) s += (ch[i] - m) ** 2;
      return { m, s: Math.sqrt(s / n) || 1 };
    }

    // Channel order matches Python's OpenCV BGR split: cv.merge([b,g,r,th])
    // -> data_vector columns are [B, G, R, T].
    const channels = [chB, chG, chR, chT];
    const stats    = channels.map(meanAndStd);
    // FactorAnalysis standardises features internally; standardise here so the
    // correlation structure FA factorises is reproduced.
    const centred  = channels.map((ch, c) => {
      const { m, s } = stats[c];
      const f = new Float32Array(n);
      for (let i = 0; i < n; i++) f[i] = (ch[i] - m) / s;
      return f;
    });

    // Correlation matrix of the 4 standardised channels (FA factorises the
    // shared/common variance of this matrix).
    const cov = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = i; j < 4; j++) {
        let s = 0;
        const ci = centred[i], cj = centred[j];
        for (let k = 0; k < n; k++) s += ci[k] * cj[k];
        cov[i * 4 + j] = cov[j * 4 + i] = s / (n - 1);
      }
    }

    // Eigendecomposition gives the loading directions; the top-3 eigenvectors
    // approximate FA's 3 latent factors (loadings w_k scaled by sqrt(eigenvalue)).
    const { values, vectors } = jacobiEigen4(cov);

    // Per-factor latent scores z_k = X_std · w_k  (the 3 FACTOR maps), plus the
    // loading vector w_k = sqrt(λ_k)·eigvec_k so we can reconstruct each factor's
    // contribution to the 4 input channels and measure what it leaves unexplained.
    const proj    = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
    const loading = [new Float32Array(4), new Float32Array(4), new Float32Array(4)];
    for (let k = 0; k < 3; k++) {
      const ev = vectors[k];
      const sc = Math.sqrt(Math.max(0, values[k]));
      for (let c = 0; c < 4; c++) loading[k][c] = ev[c] * sc;
      for (let i = 0; i < n; i++) {
        proj[k][i] = centred[0][i] * ev[0] + centred[1][i] * ev[1]
                   + centred[2][i] * ev[2] + centred[3][i] * ev[3];
      }
    }

    // Per-FACTOR estimated error / noise (ε in X ≈ Z·Wᵀ + ε):
    // reconstruct the 4 standardised input channels from factor k ALONE
    // (z_k·w_kᵀ) and take the RMS residual across the 4 channels. High residual
    // = signal this single factor cannot represent → the factor's noise map.
    // Estimated FA noise ε — ONE map, computed AFTER all 3 factors. The model is
    // X ≈ Z·Wᵀ + μ + ε, so ε is what the full factorization leaves unexplained:
    // reconstruct each standardised channel from ALL 3 factors (Σ_k z_k·w_kᵀ) and
    // take the RMS residual per pixel. It is deliberately NOT per-factor: a
    // single-factor residual still contains the OTHER factors' real structure, so
    // it would look more informative than the factor itself (an artefact, not
    // noise). Only the all-factor residual is the genuine noise term ε.
    const resid = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let c = 0; c < 4; c++) {
        let recon = 0;
        for (let k = 0; k < 3; k++) recon += proj[k][i] * loading[k][c];
        const r = centred[c][i] - recon;
        acc += r * r;
      }
      resid[i] = Math.sqrt(acc / 4);
    }

    const [c1, c2, c3] = proj.map(normalizeToUint8);
    const output = falseColourImageData(c1, c2, c3, w, h);

    // Factor score maps as GREYSCALE images (F1/F2/F3). The factors are abstract
    // latent variables, not R/G/B colour channels, so they are shown grey (matches
    // PCA's components); only the final OUTPUT is the false-colour composite that
    // maps the 3 factors onto the 3 output channels.
    const facImgs = [c1, c2, c3].map((c) => greyImageData(c, w, h));

    // Estimated noise ε as a single heatmap (alphaHeatmap: blue=low → red=high
    // unexplained signal). Faint by nature, so gamma-lifted and announced with 'ε*'
    // (mirrors subbandMosaic's '*' detail annotation).
    function residHeatmap(e) {
      let mx = 0;
      for (let i = 0; i < n; i++) if (e[i] > mx) mx = e[i];
      const inv = mx > 0 ? 1 / mx : 0;
      const a = new Float32Array(n);
      // gamma<1 lift so the faint noise structure becomes perceptible (*).
      for (let i = 0; i < n; i++) a[i] = Math.pow(e[i] * inv, 0.6);
      const img = alphaHeatmap(a, w, h);
      const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
      const ctx = cv.getContext('2d');
      ctx.putImageData(img, 0, 0);
      const fs = Math.max(10, Math.round(w * 0.05));
      ctx.font = `bold ${fs}px system-ui, sans-serif`;
      ctx.textBaseline = 'top';
      const t = 'ε*';
      const tw = ctx.measureText(t).width;
      ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(4, 4, tw + 8, fs + 6);
      ctx.fillStyle = 'rgba(255,255,255,0.95)'; ctx.fillText(t, 8, 6);
      // '*' on ε* (residual amplified for visibility) is explained once by the
      // shared starLegend beside the diagram — no Spanish-only baked caption.
      return ctx.getImageData(0, 0, w, h);
    }
    const residImg = residHeatmap(resid);

    const imgChR = falseColourImageData(chR, new Uint8Array(n), new Uint8Array(n), w, h);
    const imgChG = falseColourImageData(new Uint8Array(n), chG, new Uint8Array(n), w, h);
    const imgChB = falseColourImageData(new Uint8Array(n), new Uint8Array(n), chB, w, h);
    const imgChT = greyImageData(chT, w, h);

    const facIds = ['f1', 'f2', 'f3'];
    const facCls  = ['pipe-card--f1', 'pipe-card--f2', 'pipe-card--f3'];
    const facClr  = ['var(--clr-ch-f1)', 'var(--clr-ch-f2)', 'var(--clr-ch-f3)'];

    return {
      stages: [
        {
          key: 'fa-channels',
          opNode: { id: 'op-split', labelKey: 'splitRgb' },
          cards: [
            { id: 'ch-r', imageData: imgChR, cardClass: 'pipe-card--r', desc: 'R' },
            { id: 'ch-g', imageData: imgChG, cardClass: 'pipe-card--g', desc: 'G' },
            { id: 'ch-b', imageData: imgChB, cardClass: 'pipe-card--b', desc: 'B' },
            { id: 'ch-t', imageData: imgChT, cardClass: 'pipe-card--t', desc: 'T', separateBefore: true },
          ],
        },
        {
          // FA is a SINGLE joint operation on the 4-channel stack (applied once,
          // not per-channel): one FA node consumes all 4 channels at once and emits
          // the 3 latent factors AND the estimated noise ε together — siblings of
          // the same estimation, so factors and error branch out IN PARALLEL from
          // the node (not noise-out-of-a-factor in series).
          key: 'fa-factors',
          opNode: { id: 'op-fa', labelKey: 'faOp' },
          cards: [
            ...facIds.map((id, k) => ({
              id, imageData: facImgs[k], cardClass: facCls[k], desc: `F${k + 1}`,
            })),
            { id: 'resid', imageData: residImg, cardClass: 'pipe-card--alpha',
              desc: 'ε (residual)', separateBefore: true },
          ],
        },
        {
          key: 'fa-out',
          opNode: { id: 'op-merge', labelKey: 'mergeOut' },
          cards: [
            { id: 'out', imageData: output, cardClass: 'pipe-card--out',
              desc: 'Output', isOutput: true },
          ],
        },
      ],
      connections: [
        { from: 'src-rgb',  to: 'op-split', color: 'var(--clr-border)' },
        { from: 'src-lwir', to: 'ch-t',     color: 'var(--clr-ch-t)' },
        { from: 'op-split', to: 'ch-r',     color: 'var(--clr-ch-r)' },
        { from: 'op-split', to: 'ch-g',     color: 'var(--clr-ch-g)' },
        { from: 'op-split', to: 'ch-b',     color: 'var(--clr-ch-b)' },
        // All 4 channels jointly feed the single FA factorization.
        { from: 'ch-r', to: 'op-fa', color: 'var(--clr-ch-r)' },
        { from: 'ch-g', to: 'op-fa', color: 'var(--clr-ch-g)' },
        { from: 'ch-b', to: 'op-fa', color: 'var(--clr-ch-b)' },
        { from: 'ch-t', to: 'op-fa', color: 'var(--clr-ch-t)' },
        // The FA node emits the 3 factors and the noise ε in parallel.
        ...facIds.map((id, k) => ({ from: 'op-fa', to: id, color: facClr[k] })),
        { from: 'op-fa', to: 'resid', color: 'var(--clr-ch-alpha)' },
        // Only the 3 factors merge into the false-colour output (ε is diagnostic).
        ...facIds.map((id, k) => ({ from: id, to: 'op-merge', color: facClr[k] })),
        { from: 'op-merge', to: 'out', color: 'var(--clr-border)' },
      ],
      output,
    };
  }


  function computePCA(rgbData, lwirData, w, h)        { return _computeDecompositionFusion(rgbData, lwirData, w, h, 'pca'); }
  function computeFA(rgbData, lwirData, w, h)         { return _computeFaFusion(rgbData, lwirData, w, h); }
  function computeSobel(rgbData, lwirData, w, h)      { return _computeSobelFusion(rgbData, lwirData, w, h); }
  function computeSSIM(rgbData, lwirData, w, h)       { return _computeSsimFusion(rgbData, lwirData, w, h); }
  function computeSuperpixel(rgbData, lwirData, w, h) { return _computeSuperpixelFusion(rgbData, lwirData, w, h); }

  /* ════════════════════════════════════════════════════════════════
     EXPORT
  ════════════════════════════════════════════════════════════════ */

  window.FusionCompute = {
    COMPUTE_FN: {
      rgbt: computeRGBT, hsvt: computeHSVT, vths: computeVTHS, vt: computeVT,
      wavelet: computeWavelet, waveletmax: computeWaveletMax,
      curvelet: computeCurvelet, curveletmax: computeCurveletMax,
      pca: computePCA, fa: computeFA,
      sobel: computeSobel, ssim: computeSSIM, superpixel: computeSuperpixel,
    },
  };
})();
