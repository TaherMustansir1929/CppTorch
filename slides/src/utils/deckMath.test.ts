import test from 'node:test';
import assert from 'node:assert/strict';
import {
  clampSlideIndex,
  nextSlideIndex,
  prevSlideIndex,
  slideProgressPercent,
  multiplyMatrices,
  getMatrixHighlightIndices,
  computeSpeedupRatio,
  evaluateLeakyRelu,
  evaluateMsePoint,
  computeAdamStep,
  generateOptimizerTrajectory,
  fitTransformStandardScaler,
  classifySvmPoint,
  sampleNavierStokesVorticity,
  NODE_INFLATION_CONFIG,
  applyNodeInflationClick,
  decayNodeInflation,
} from './deckMath';

test('slide navigation clamps properly within bounds', () => {
  assert.equal(clampSlideIndex(-3, 13), 0);
  assert.equal(clampSlideIndex(18, 13), 12);
  assert.equal(nextSlideIndex(0, 13), 1);
  assert.equal(nextSlideIndex(12, 13), 12);
  assert.equal(prevSlideIndex(0, 13), 0);
  assert.equal(prevSlideIndex(5, 13), 4);
  assert.equal(slideProgressPercent(0, 13), 8);
  assert.equal(slideProgressPercent(12, 13), 100);
});

test('matrix multiplication computes C = A * B accurately', () => {
  const A = [
    [1, 2],
    [3, 4],
  ];
  const B = [
    [5, 6],
    [7, 8],
  ];
  assert.deepEqual(multiplyMatrices(A, B), [
    [19, 22],
    [43, 50],
  ]);

  const highlight = getMatrixHighlightIndices(1, 2, 3);
  assert.deepEqual(highlight.rowA, [3, 4, 5]);
  assert.deepEqual(highlight.colB, [2, 5, 8]);
  assert.deepEqual(highlight.cellC, [1, 2]);
});

test('benchmark speedup ratio matches 15x calculation', () => {
  assert.equal(computeSpeedupRatio(420, 28), 15);
  assert.equal(computeSpeedupRatio(12400, 830), 14.9);
});

test('activation and loss math evaluate ReLU, LeakyReLU, and MSE', () => {
  assert.equal(evaluateLeakyRelu(2.5, 0), 2.5);
  assert.equal(evaluateLeakyRelu(-2.0, 0), 0);
  assert.equal(evaluateLeakyRelu(-2.0, 0.2), -0.4);
  assert.equal(evaluateMsePoint(0.75, 1.0), 0.0625);
});

test('Adam optimizer step updates biased moments and parameters', () => {
  const next = computeAdamStep({ theta: 1.0, m: 0, v: 0, t: 0 }, 0.5, 0.1);
  assert.equal(next.t, 1);
  assert.ok(next.theta < 1.0);
  assert.ok(next.m > 0);
  assert.ok(next.v > 0);

  const adamPts = generateOptimizerTrajectory('Adam', 20);
  assert.equal(adamPts.length, 20);
});

test('StandardScaler centers mean to 0 and std to 1', () => {
  const res = fitTransformStandardScaler([10, 20, 30]);
  assert.equal(res.mean, 20);
  assert.ok(Math.abs(res.scaled[1]) < 1e-9);
  assert.ok(res.scaled[0] < 0 && res.scaled[2] > 0);
});

test('SVM classifier and Navier-Stokes vorticity return valid outputs', () => {
  assert.equal(classifySvmPoint(0, 2), 1);
  assert.equal(classifySvmPoint(2, 0), -1);
  const w = sampleNavierStokesVorticity(0.5, 0.5, 1.0);
  assert.equal(typeof w, 'number');
  assert.ok(Number.isFinite(w));
});

test('node inflation accumulates on clicks, caps at max threshold, and deflates gradually', () => {
  let inflation = 0;
  inflation = applyNodeInflationClick(inflation);
  assert.equal(inflation, NODE_INFLATION_CONFIG.step);

  for (let i = 0; i < 10; i++) {
    inflation = applyNodeInflationClick(inflation);
  }
  assert.equal(inflation, NODE_INFLATION_CONFIG.maxInflation);

  const afterHalfSec = decayNodeInflation(inflation, 0.5);
  assert.ok(afterHalfSec < inflation && afterHalfSec > 0);

  const afterLongWait = decayNodeInflation(afterHalfSec, 10);
  assert.equal(afterLongWait, 0);
});
