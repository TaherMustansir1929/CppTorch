export interface BenchmarkItem {
  id: string;
  operation: string;
  singleThreadMs: number;
  parallelMs: number;
  unit: 'ms' | 's';
  category: 'operation' | 'epoch';
}

export const BENCHMARK_ITEMS: BenchmarkItem[] = [
  {
    id: 'dot',
    operation: 'Matrix dot (1024×1024)',
    singleThreadMs: 420,
    parallelMs: 28,
    unit: 'ms',
    category: 'operation',
  },
  {
    id: 'relu',
    operation: 'Element-wise ReLU (1M elements)',
    singleThreadMs: 95,
    parallelMs: 7,
    unit: 'ms',
    category: 'operation',
  },
  {
    id: 'epoch',
    operation: 'Full training epoch (spiral, 100 epochs)',
    singleThreadMs: 12400,
    parallelMs: 830,
    unit: 's',
    category: 'epoch',
  },
];

export function clampSlideIndex(index: number, totalSlides: number): number {
  if (totalSlides <= 0) return 0;
  return Math.max(0, Math.min(totalSlides - 1, Math.trunc(index)));
}

export function nextSlideIndex(current: number, totalSlides: number): number {
  return clampSlideIndex(current + 1, totalSlides);
}

export function prevSlideIndex(current: number, totalSlides: number): number {
  return clampSlideIndex(current - 1, totalSlides);
}

export function slideProgressPercent(current: number, totalSlides: number): number {
  if (totalSlides <= 1) return 100;
  const clamped = clampSlideIndex(current, totalSlides);
  return Math.round(((clamped + 1) / totalSlides) * 100);
}

export function multiplyMatrices(a: number[][], b: number[][]): number[][] {
  if (a.length === 0 || b.length === 0 || a[0].length !== b.length) {
    throw new Error('Incompatible matrix dimensions for dot product');
  }
  const rows = a.length;
  const cols = b[0].length;
  const inner = b.length;
  const out: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      let sum = 0;
      for (let k = 0; k < inner; k++) {
        sum += a[i][k] * b[k][j];
      }
      out[i][j] = Number(sum.toFixed(4));
    }
  }
  return out;
}

export function getMatrixHighlightIndices(
  row: number,
  col: number,
  size: number
): { rowA: number[]; colB: number[]; cellC: [number, number] } {
  const safeRow = Math.max(0, Math.min(size - 1, row));
  const safeCol = Math.max(0, Math.min(size - 1, col));
  const indices = Array.from({ length: size }, (_, idx) => idx);
  return {
    rowA: indices.map((k) => safeRow * size + k),
    colB: indices.map((k) => k * size + safeCol),
    cellC: [safeRow, safeCol],
  };
}

export function computeSpeedupRatio(singleMs: number, parallelMs: number): number {
  if (parallelMs <= 0) return 0;
  return Number((singleMs / parallelMs).toFixed(1));
}

export function evaluateLeakyRelu(x: number, alpha = 0): number {
  if (x >= 0) return x;
  return alpha === 0 ? 0 : alpha * x;
}

export function evaluateMsePoint(yPred: number, yTrue: number): number {
  const diff = yPred - yTrue;
  return Number((diff * diff).toFixed(4));
}

export interface AdamState {
  theta: number;
  m: number;
  v: number;
  t: number;
}

export function computeAdamStep(
  state: AdamState,
  gradient: number,
  lr = 0.05,
  beta1 = 0.9,
  beta2 = 0.999,
  eps = 1e-8
): AdamState {
  const t = state.t + 1;
  const m = beta1 * state.m + (1 - beta1) * gradient;
  const v = beta2 * state.v + (1 - beta2) * gradient * gradient;
  const mHat = m / (1 - Math.pow(beta1, t));
  const vHat = v / (1 - Math.pow(beta2, t));
  const theta = state.theta - (lr * mHat) / (Math.sqrt(vHat) + eps);
  return { theta, m, v, t };
}

export type OptimizerName = 'SGD' | 'Adagrad' | 'RMSprop' | 'Adam';

export function generateOptimizerTrajectory(
  optimizer: OptimizerName,
  steps = 28
): Array<{ x: number; y: number; z: number }> {
  const pts: Array<{ x: number; y: number; z: number }> = [];
  for (let i = 0; i < steps; i++) {
    const t = i / Math.max(1, steps - 1);
    if (optimizer === 'SGD') {
      pts.push({
        x: (t - 0.5) * 1.4,
        y: Math.sin(t * Math.PI) * 0.15,
        z: (t - 0.5) * 0.4,
      });
    } else if (optimizer === 'Adagrad') {
      const r = 0.72 * (1 - t * 0.78);
      const angle = t * Math.PI * 4.5;
      pts.push({
        x: Math.cos(angle) * r,
        y: (t - 0.5) * 0.5,
        z: Math.sin(angle) * r,
      });
    } else if (optimizer === 'RMSprop') {
      const amp = 0.6 * (1 - t * 0.65);
      pts.push({
        x: (t - 0.5) * 1.4,
        y: Math.sin(t * Math.PI * 6) * amp,
        z: Math.cos(t * Math.PI * 3) * amp * 0.5,
      });
    } else {
      const r = 0.68 * (1 - t * 0.85);
      const angle = t * Math.PI * 5;
      pts.push({
        x: Math.cos(angle) * r,
        y: Math.sin(t * Math.PI * 4) * 0.25 * (1 - t * 0.6) + (t - 0.5) * 0.35,
        z: Math.sin(angle) * r,
      });
    }
  }
  return pts;
}

export function fitTransformStandardScaler(values: number[]): {
  mean: number;
  std: number;
  scaled: number[];
} {
  if (values.length === 0) {
    return { mean: 0, std: 1, scaled: [] };
  }
  const mean = values.reduce((acc, v) => acc + v, 0) / values.length;
  const variance =
    values.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / values.length;
  const std = Math.sqrt(variance) || 1;
  const scaled = values.map((v) => Number(((v - mean) / std).toFixed(4)));
  return {
    mean: Number(mean.toFixed(4)),
    std: Number(std.toFixed(4)),
    scaled,
  };
}

export function classifySvmPoint(
  x: number,
  y: number,
  planeOffset = 0,
  planeSlope = 0.85
): 1 | -1 {
  const score = y - (planeSlope * x + planeOffset);
  return score >= 0 ? 1 : -1;
}

export function sampleNavierStokesVorticity(
  x: number,
  y: number,
  t: number
): number {
  const v1 = Math.sin(x * 1.6 + t * 0.9) * Math.cos(y * 1.6 - t * 0.7);
  const v2 = 0.5 * Math.cos((x - y) * 2.1 + t * 1.1);
  return Number((v1 + v2).toFixed(4));
}

export const NODE_INFLATION_CONFIG = {
  step: 0.24,
  maxInflation: 0.72,
  deflateRatePerSec: 0.22,
} as const;

export function applyNodeInflationClick(
  current: number,
  step = NODE_INFLATION_CONFIG.step,
  maxInflation = NODE_INFLATION_CONFIG.maxInflation
): number {
  const safeCurrent = Math.max(0, current);
  return Math.min(maxInflation, Number((safeCurrent + step).toFixed(4)));
}

export function decayNodeInflation(
  current: number,
  deltaSec: number,
  deflateRatePerSec = NODE_INFLATION_CONFIG.deflateRatePerSec
): number {
  if (current <= 0 || deltaSec <= 0) return Math.max(0, current);
  return Math.max(0, Number((current - deflateRatePerSec * deltaSec).toFixed(4)));
}
