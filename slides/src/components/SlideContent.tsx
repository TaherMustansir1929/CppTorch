import {
  Activity,
  ArrowRight,
  BookOpen,
  Cpu,
  Database,
  ExternalLink,
  GitBranch,
  GraduationCap,
  Grid3X3,
  Layers,
  Play,
  RotateCcw,
  Ruler,
  Sliders,
  Sparkles,
  Waves,
  Zap,
} from 'lucide-react';
import {
  CLASS_SECTION,
  DOCS_SITE_URL,
  GITHUB_REPO_URL,
  UNIVERSITY_NAME,
  type SlideDefinition,
} from '../data/slides';
import type { SlideSceneState } from '../three/sceneBuilders';
import {
  BENCHMARK_ITEMS,
  computeSpeedupRatio,
  evaluateLeakyRelu,
  evaluateMsePoint,
  type OptimizerName,
} from '../utils/deckMath';

interface SlideContentProps {
  slide: SlideDefinition;
  sceneState: SlideSceneState;
  onUpdateState: (patch: Partial<SlideSceneState>) => void;
  onRestartDeck: () => void;
}

const OPTIMIZER_RULES: Record<OptimizerName, string> = {
  SGD: 'θ = θ − lr · (g + momentum · v)',
  Adagrad: 'θ = θ − lr · g / (√G + ε)',
  RMSprop: 'v = ρ·v + (1−ρ)·g²;  θ = θ − lr·g / (√v + ε)',
  Adam: 'm = β₁m + (1−β₁)g;  θ = θ − lr · m̂ / (√v̂ + ε)',
};

const EXTENSION_SNIPPETS: Record<number, { title: string; code: string }> = {
  0: {
    title: 'include/cpptorch/activations/swish.h',
    code: `class Activation_Swish : public Activation {
public:
  Matrix forward(const Matrix& inputs) override;
  Matrix backward(const Matrix& dvalues) override;
};`,
  },
  1: {
    title: 'include/cpptorch/activations/registry.h',
    code: `inline std::unique_ptr<Activation>
make_activation(ActivationType type) {
  case ActivationType::Swish:
    return std::make_unique<Activation_Swish>();
}`,
  },
  2: {
    title: 'include/cpptorch/cpptorch.h',
    code: `#pragma once
#include "core/matrix.h"
#include "activations/swish.h"
#include "model/model.h" // untouched!`,
  },
};

export function SlideContent({
  slide,
  sceneState,
  onUpdateState,
  onRestartDeck,
}: SlideContentProps) {
  // Slide 1: Welcome & Product Reveal
  if (slide.id === 1) {
    return (
      <div className="flex flex-col justify-between h-full py-2">
        <div className="space-y-6">
          <div className="flex items-center gap-3 text-sm font-code font-semibold text-[#14213d]/80">
            <GraduationCap className="w-5 h-5 text-[#14213d]" strokeWidth={2} />
            <span>{UNIVERSITY_NAME} · DSA Project · C++20</span>
          </div>

          <div>
            <h1 className="text-5xl lg:text-6xl font-black tracking-tight text-[#14213d] leading-none">
              {slide.title}
            </h1>
            <p className="mt-3 text-xl lg:text-2xl font-medium text-[#14213d]/85 text-balance">
              {slide.subtitle}
            </p>
          </div>

          <div className="bg-[#14213d] text-white p-5 border-t-4 border-[#ff6b5b]">
            <p className="text-lg lg:text-xl font-medium leading-snug">
              &ldquo;{slide.pitch}&rdquo;
            </p>
          </div>

          <div className="space-y-2.5 pt-1">
            <p className="text-xs font-code font-semibold uppercase tracking-wider text-[#14213d]/70">
              Authors
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {slide.authors?.map((author) => (
                <div
                  key={author.roll}
                  className="border border-[#14213d]/25 bg-white/60 px-3.5 py-2.5"
                >
                  <div className="font-bold text-[#14213d] text-base">{author.name}</div>
                  <div className="font-code text-xs text-[#14213d]/80 tabular-nums">
                    {author.roll}
                  </div>
                </div>
              ))}
            </div>
            <p className="pt-1 flex items-start gap-2 font-code text-xs text-[#14213d]/65 leading-relaxed">
              <span className="mt-1.5 w-1.5 h-1.5 bg-[#ff6b5b] shrink-0" aria-hidden="true" />
              <span>
                Note: Click the 3D elements, sliders, and buttons across every slide—seriously, they&apos;re meant to be poked.
              </span>
            </p>
          </div>
        </div>

        <div className="pt-5 flex items-center justify-between gap-4 border-t border-[#14213d]/15">
          <button
            type="button"
            onClick={() => onUpdateState({ pulseTrigger: sceneState.pulseTrigger + 1 })}
            className="inline-flex items-center gap-2 bg-[#ff6b5b] text-[#14213d] font-bold text-sm px-4 py-2.5 border-2 border-[#14213d] hover:bg-[#14213d] hover:text-white transition-colors cursor-pointer"
          >
            <Zap className="w-4 h-4" strokeWidth={2} />
            <span>Pulse Neural Cluster</span>
          </button>
          <div className="flex items-center gap-2.5">
            <span className="font-code text-[11px] text-[#14213d]/50 tracking-wide">
              {CLASS_SECTION}
            </span>
            <div className="w-4 h-4 bg-[#ff6b5b] border border-[#14213d]" aria-hidden="true" />
          </div>
        </div>
      </div>
    );
  }

  // Slide 3: Core Design Philosophy (3-column interactive cards)
  if (slide.id === 3) {
    const icons = [Layers, GitBranch, Cpu];
    return (
      <div className="flex flex-col justify-between h-full gap-4">
        <div>
          <h1 className="text-3xl lg:text-4xl font-bold text-[#14213d] tracking-tight">
            {slide.title}
          </h1>
          <p className="text-base text-[#14213d]/80 mt-1">{slide.subtitle}</p>
        </div>

        <div className="grid grid-cols-1 gap-3">
          {slide.cards?.map((card, idx) => {
            const Icon = icons[idx % icons.length];
            const isSelected = sceneState.selectedPillarIndex === idx;
            return (
              <button
                key={card.title}
                type="button"
                onClick={() =>
                  onUpdateState({
                    selectedPillarIndex: isSelected ? null : idx,
                  })
                }
                className={`text-left p-4 border-t-4 border-[#ff6b5b] transition-colors cursor-pointer ${
                  isSelected
                    ? 'bg-[#14213d] text-white ring-2 ring-[#ff6b5b]'
                    : 'bg-[#14213d] text-white hover:bg-[#1e3159]'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2.5">
                    <Icon className="w-4 h-4 text-[#ff6b5b]" strokeWidth={2} />
                    <span className="font-bold text-base">{card.title}</span>
                  </div>
                  <span className="font-code text-xs text-[#ff6b5b] tabular-nums">
                    {card.tag}
                  </span>
                </div>
                <p className="mt-1.5 text-sm text-[#f4ede0]/90 leading-relaxed">
                  {card.detail}
                </p>
              </button>
            );
          })}
        </div>

        {sceneState.selectedPillarIndex !== null && (
          <button
            type="button"
            onClick={() => onUpdateState({ selectedPillarIndex: null })}
            className="self-start inline-flex items-center gap-1.5 text-xs font-code font-semibold text-[#14213d] border border-[#14213d] px-3 py-1.5 bg-white/70 hover:bg-[#14213d] hover:text-white transition-colors cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" strokeWidth={2} />
            <span>Reset Camera Focus</span>
          </button>
        )}
      </div>
    );
  }

  // Slide 5: Performance Engine — Benchmark Card
  if (slide.id === 5) {
    const visibleItems =
      sceneState.benchmarkMode === 'epoch'
        ? BENCHMARK_ITEMS
        : BENCHMARK_ITEMS.filter((i) => i.category === 'operation');

    return (
      <div className="flex flex-col justify-between h-full gap-4">
        <div>
          <h1 className="text-3xl lg:text-4xl font-bold text-[#14213d] tracking-tight">
            {slide.title}
          </h1>
          <p className="text-base text-[#14213d]/80 mt-1">{slide.subtitle}</p>
        </div>

        <div className="bg-[#14213d] text-white p-5 border-t-4 border-[#ff6b5b] space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-lg font-bold text-[#ff6b5b]">
              {slide.highlightBadge}
            </span>
            <span className="font-code text-xs text-[#f4ede0]/80">
              {slide.equation}
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left font-code text-xs sm:text-sm tabular-nums">
              <thead>
                <tr className="border-b border-white/20 text-[#f4ede0]/75">
                  <th className="py-2 pr-3 font-medium">Operation</th>
                  <th className="py-2 px-2 font-medium">Single-Thread</th>
                  <th className="py-2 px-2 font-medium text-[#ff6b5b]">
                    OpenMP + BLAS
                  </th>
                  <th className="py-2 pl-2 text-right font-medium">Speedup</th>
                </tr>
              </thead>
              <tbody>
                {visibleItems.map((row) => {
                  const ratio = computeSpeedupRatio(
                    row.singleThreadMs,
                    row.parallelMs
                  );
                  const singleLabel =
                    row.unit === 's'
                      ? `${(row.singleThreadMs / 1000).toFixed(1)} s`
                      : `${row.singleThreadMs} ms`;
                  const parLabel =
                    row.unit === 's'
                      ? `${(row.parallelMs / 1000).toFixed(2)} s`
                      : `${row.parallelMs} ms`;
                  return (
                    <tr key={row.id} className="border-b border-white/10 last:border-0">
                      <td className="py-2 pr-3 text-white">{row.operation}</td>
                      <td className="py-2 px-2 text-[#f4ede0]/80">{singleLabel}</td>
                      <td className="py-2 px-2 font-bold text-[#ff6b5b]">
                        {parLabel}
                      </td>
                      <td className="py-2 pl-2 text-right font-bold text-white">
                        {ratio}×
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-white/15">
            <div className="flex items-center gap-2 text-xs text-[#f4ede0]/85">
              <Cpu className="w-4 h-4 text-[#ff6b5b]" strokeWidth={2} />
              <span>OpenMP Threads · CBLAS dgemm</span>
            </div>
            <div className="font-code text-2xl font-bold text-[#ff6b5b] tabular-nums">
              15×
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => onUpdateState({ benchmarkMode: 'epoch' })}
            className={`px-3 py-2 text-xs font-code font-semibold border border-[#14213d] cursor-pointer transition-colors ${
              sceneState.benchmarkMode === 'epoch'
                ? 'bg-[#14213d] text-white'
                : 'bg-white/70 text-[#14213d] hover:bg-[#14213d]/10'
            }`}
          >
            All + Training Epoch
          </button>
          <button
            type="button"
            onClick={() => onUpdateState({ benchmarkMode: 'operation' })}
            className={`px-3 py-2 text-xs font-code font-semibold border border-[#14213d] cursor-pointer transition-colors ${
              sceneState.benchmarkMode === 'operation'
                ? 'bg-[#14213d] text-white'
                : 'bg-white/70 text-[#14213d] hover:bg-[#14213d]/10'
            }`}
          >
            Single Operation Only
          </button>
          <button
            type="button"
            onClick={() =>
              onUpdateState({ replayTrigger: sceneState.replayTrigger + 1 })
            }
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-code font-semibold bg-[#ff6b5b] text-[#14213d] border border-[#14213d] hover:bg-[#14213d] hover:text-white transition-colors cursor-pointer"
          >
            <Play className="w-3.5 h-3.5" strokeWidth={2} />
            <span>Replay 3D Race</span>
          </button>
        </div>
      </div>
    );
  }

  // Slide 11: Examples — Neural SVM & Navier-Stokes
  if (slide.id === 11) {
    return (
      <div className="flex flex-col justify-between h-full gap-4">
        <div>
          <h1 className="text-3xl lg:text-4xl font-bold text-[#14213d] tracking-tight">
            {slide.title}
          </h1>
          <p className="text-base text-[#14213d]/80 mt-1">{slide.subtitle}</p>
        </div>

        <div className="grid grid-cols-1 gap-3">
          <div className="bg-[#14213d] text-white p-4 border-t-4 border-[#ff6b5b]">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-[#ff6b5b]" strokeWidth={2} />
                <h2 className="font-bold text-base">Neural SVM (svm_dataset.csv)</h2>
              </div>
              <span className="font-code text-xs font-bold text-[#ff6b5b] tabular-nums">
                98.3% accuracy
              </span>
            </div>
            <p className="mt-1.5 font-code text-xs text-[#f4ede0]/85">
              2 → 32 ReLU → 16 ReLU → 1 Linear · MSE on ±1 targets · 80/20 split
            </p>
            <p className="mt-1 text-xs text-[#f4ede0]/80">
              Reports epoch loss, precision, recall, F1, RMSE, and confusion matrix.
            </p>
          </div>

          <div className="bg-[#14213d] text-white p-4 border-t-4 border-[#ff6b5b]">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Waves className="w-4 h-4 text-[#ff6b5b]" strokeWidth={2} />
                <h2 className="font-bold text-base">
                  Navier-Stokes 2D (ns2d_subset.csv)
                </h2>
              </div>
              <span className="font-code text-xs font-bold text-[#ff6b5b] tabular-nums">
                Continuity RMS 0.014
              </span>
            </div>
            <p className="mt-1.5 font-code text-xs text-[#f4ede0]/85">
              Predict (ω, u, v) from (x, y, t) on time-ordered held-out frames
            </p>
            <p className="mt-1 text-xs text-[#f4ede0]/80">
              Computes per-field R², MAE, and vorticity-transport PDE residual RMS.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {(['both', 'svm', 'navier'] as const).map((focus) => (
            <button
              key={focus}
              type="button"
              onClick={() => onUpdateState({ exampleFocus: focus })}
              className={`px-3 py-2 text-xs font-code font-semibold border border-[#14213d] cursor-pointer transition-colors ${
                sceneState.exampleFocus === focus
                  ? 'bg-[#14213d] text-white'
                  : 'bg-white/70 text-[#14213d] hover:bg-[#14213d]/10'
              }`}
            >
              {focus === 'both'
                ? 'Show Both'
                : focus === 'svm'
                ? 'Focus SVM'
                : 'Focus Fluid Grid'}
            </button>
          ))}
          <button
            type="button"
            onClick={() =>
              onUpdateState({
                snapSvmPlaneTrigger: sceneState.snapSvmPlaneTrigger + 1,
              })
            }
            className="px-3 py-2 text-xs font-code font-semibold bg-[#ff6b5b] text-[#14213d] border border-[#14213d] hover:bg-[#14213d] hover:text-white transition-colors cursor-pointer"
          >
            Snap Optimal SVM Plane
          </button>
        </div>
      </div>
    );
  }

  // Slide 13: Thank You & Closing
  if (slide.id === 13) {
    return (
      <div className="flex flex-col justify-between h-full py-2">
        <div className="space-y-5">
          <div>
            <h1 className="text-5xl lg:text-6xl font-black tracking-tight text-[#14213d] leading-none">
              {slide.title}
            </h1>
            <p className="mt-3 text-lg lg:text-xl font-medium text-[#14213d]/85">
              CppTorch — A Minimal Neural-Networks Library in C++20
            </p>
          </div>

          <div className="border-t-2 border-[#ff6b5b] pt-4 space-y-2.5">
            <p className="text-xs font-code font-semibold uppercase tracking-wider text-[#14213d]/70">
              Questions &amp; Discussion · Project Team
            </p>
            <div className="space-y-2">
              {slide.authors?.map((author, idx) => {
                const isSelected = sceneState.selectedAuthorIndex === idx;
                return (
                  <button
                    key={author.roll}
                    type="button"
                    onClick={() =>
                      onUpdateState({
                        selectedAuthorIndex: isSelected ? null : idx,
                      })
                    }
                    className={`w-full flex items-center justify-between gap-3 px-4 py-2.5 border transition-colors cursor-pointer ${
                      isSelected
                        ? 'bg-[#14213d] text-white border-[#14213d]'
                        : 'bg-white/70 text-[#14213d] border-[#14213d]/30 hover:border-[#14213d]'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <span className="font-bold">{author.name}</span>
                      {isSelected && (
                        <span className="bg-white text-[#14213d] font-code text-[11px] font-semibold px-2 py-0.5 border border-[#14213d]/30 whitespace-nowrap">
                          {author.role}
                        </span>
                      )}
                    </div>
                    <span
                      className={`font-code text-xs tabular-nums shrink-0 ${
                        isSelected ? 'text-[#ff6b5b]' : 'text-[#14213d]/80'
                      }`}
                    >
                      {author.roll}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="border-l-4 border-[#14213d] bg-[#ebe2d0] px-4 py-3 font-code text-xs text-[#14213d] space-y-2.5">
            <a
              href={DOCS_SITE_URL}
              target="_blank"
              rel="noopener noreferrer"
              data-clickable="true"
              className="w-full flex items-center justify-between gap-3 px-3.5 py-2.5 bg-[#14213d] text-white border-2 border-[#14213d] hover:bg-[#ff6b5b] hover:text-[#14213d] transition-colors group"
            >
              <div className="flex items-center gap-2 min-w-0">
                <BookOpen className="w-4 h-4 text-[#ff6b5b] group-hover:text-[#14213d] shrink-0" strokeWidth={2} />
                <span className="text-[11px] uppercase tracking-wider font-bold text-[#ff6b5b] group-hover:text-[#14213d] shrink-0">
                  Documentation
                </span>
                <span className="font-bold truncate">cpptorch.vercel.app</span>
              </div>
              <ExternalLink className="w-3.5 h-3.5 shrink-0" strokeWidth={2} />
            </a>

            <div className="flex flex-wrap items-center justify-between gap-2">
              <a
                href={GITHUB_REPO_URL}
                target="_blank"
                rel="noopener noreferrer"
                data-clickable="true"
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-white text-[#14213d] border border-[#14213d] font-semibold hover:bg-[#ff6b5b] hover:text-[#14213d] transition-colors"
              >
                <GitBranch className="w-3.5 h-3.5 text-[#ff6b5b] shrink-0" strokeWidth={2} />
                <span className="text-[10px] uppercase tracking-wider text-[#14213d]/70">
                  GitHub Repo:
                </span>
                <span>github.com/cpptorch/cpptorch</span>
                <ExternalLink className="w-3 h-3 shrink-0" strokeWidth={2} />
              </a>
              <span className="text-[#14213d]/75">include/cpptorch/cpptorch.h</span>
            </div>
          </div>
        </div>

        <div className="pt-4 flex flex-wrap items-center gap-3 border-t border-[#14213d]/15">
          <button
            type="button"
            onClick={() =>
              onUpdateState({ burstTrigger: sceneState.burstTrigger + 1 })
            }
            className="inline-flex items-center gap-2 bg-[#ff6b5b] text-[#14213d] font-bold text-sm px-4 py-2.5 border-2 border-[#14213d] hover:bg-[#14213d] hover:text-white transition-colors cursor-pointer"
          >
            <Sparkles className="w-4 h-4" strokeWidth={2} />
            <span>Emit Signal Burst</span>
          </button>
          <button
            type="button"
            onClick={onRestartDeck}
            className="inline-flex items-center gap-2 bg-white/80 text-[#14213d] font-bold text-sm px-4 py-2.5 border-2 border-[#14213d] hover:bg-[#14213d] hover:text-white transition-colors cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" strokeWidth={2} />
            <span>Replay from Slide 1</span>
          </button>
        </div>
      </div>
    );
  }

  // Default Split / Structured Technical Slide Layout (Slides 2, 4, 6, 7, 8, 9, 10, 12)
  const activeStepSnippet =
    slide.id === 12 && sceneState.selectedStep !== null
      ? EXTENSION_SNIPPETS[sceneState.selectedStep]
      : null;

  return (
    <div className="flex flex-col justify-between h-full gap-4">
      <div className="space-y-4">
        <div>
          <h1 className="text-3xl lg:text-4xl font-bold text-[#14213d] tracking-tight">
            {slide.title}
          </h1>
          <p className="text-base text-[#14213d]/80 mt-1">{slide.subtitle}</p>
        </div>

        {/* Slide 6: Layer horizontal flow diagram */}
        {slide.id === 6 && (
          <div className="flex flex-wrap items-center gap-1.5 py-1">
            {['Input', 'Dense', 'Activation', '(Dropout)', 'Output'].map(
              (node, idx, arr) => (
                <div key={node} className="flex items-center gap-1.5">
                  <span className="bg-[#14213d] text-white font-code text-xs px-2.5 py-1.5">
                    {node}
                  </span>
                  {idx < arr.length - 1 && (
                    <ArrowRight
                      className="w-3.5 h-3.5 text-[#ff6b5b]"
                      strokeWidth={2}
                    />
                  )}
                </div>
              )
            )}
          </div>
        )}

        {/* Bullet list with Coral square accents */}
        {slide.bullets && (
          <ul className="space-y-2">
            {slide.bullets.map((bullet) => (
              <li
                key={bullet}
                className="flex items-start gap-2.5 text-sm lg:text-[15px] text-[#14213d] leading-snug"
              >
                <span
                  className="mt-1.5 w-2 h-2 shrink-0 bg-[#ff6b5b]"
                  aria-hidden="true"
                />
                <span>{bullet}</span>
              </li>
            ))}
          </ul>
        )}

        {/* Equation block */}
        {slide.equation && (
          <div className="bg-[#14213d] text-white px-4 py-3 border-t-4 border-[#ff6b5b]">
            <div className="font-code text-sm lg:text-base font-semibold text-[#ff6b5b]">
              {slide.equation}
            </div>
            {slide.equationLabel && (
              <div className="text-xs text-[#f4ede0]/80 mt-1">
                {slide.equationLabel}
              </div>
            )}
          </div>
        )}

        {/* Code snippet block (4pt Navy border-left, Cream fill, Geist Mono) */}
        {(activeStepSnippet || slide.codeSnippet) && (
          <div className="border-l-4 border-[#14213d] bg-[#ebe2d0] px-4 py-3">
            {(activeStepSnippet?.title || slide.codeTitle) && (
              <div className="font-code text-[11px] font-semibold text-[#14213d]/70 mb-1.5">
                {activeStepSnippet ? activeStepSnippet.title : slide.codeTitle}
              </div>
            )}
            <pre className="font-code text-xs text-[#14213d] overflow-x-auto leading-relaxed">
              <code>
                {activeStepSnippet ? activeStepSnippet.code : slide.codeSnippet}
              </code>
            </pre>
          </div>
        )}
      </div>

      {/* Per-slide interactive 3D controls */}
      <div className="pt-3 border-t border-[#14213d]/15">
        {slide.id === 2 && (
          <div className="flex flex-wrap gap-1.5">
            {['Header-Only', 'OpenMP+BLAS', 'Training Loop', 'Extensible'].map(
              (label, idx) => {
                const active = sceneState.selectedFeatureIndex === idx;
                return (
                  <button
                    key={label}
                    type="button"
                    onClick={() =>
                      onUpdateState({
                        selectedFeatureIndex: active ? null : idx,
                      })
                    }
                    className={`px-2.5 py-1.5 text-xs font-code font-semibold border border-[#14213d] cursor-pointer transition-colors ${
                      active
                        ? 'bg-[#ff6b5b] text-[#14213d]'
                        : 'bg-white/70 text-[#14213d] hover:bg-[#14213d] hover:text-white'
                    }`}
                  >
                    {label}
                  </button>
                );
              }
            )}
          </div>
        )}

        {slide.id === 4 && (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <label className="flex items-center gap-2.5 text-xs font-code font-semibold text-[#14213d]">
              <Grid3X3 className="w-4 h-4 text-[#ff6b5b]" strokeWidth={2} />
              <span>Matrix Size: {sceneState.matrixSize}×{sceneState.matrixSize}</span>
              <input
                type="range"
                min={2}
                max={5}
                step={1}
                value={sceneState.matrixSize}
                onChange={(e) => {
                  const nextSize = Number(e.target.value);
                  onUpdateState({
                    matrixSize: nextSize,
                    selectedCell: [
                      Math.min(sceneState.selectedCell[0], nextSize - 1),
                      Math.min(sceneState.selectedCell[1], nextSize - 1),
                    ],
                  });
                }}
                className="cpptorch-slider w-24"
              />
            </label>
            <button
              type="button"
              onClick={() => {
                const nextCol =
                  (sceneState.selectedCell[1] + 1) % sceneState.matrixSize;
                const nextRow =
                  nextCol === 0
                    ? (sceneState.selectedCell[0] + 1) % sceneState.matrixSize
                    : sceneState.selectedCell[0];
                onUpdateState({ selectedCell: [nextRow, nextCol] });
              }}
              className="px-3 py-1.5 text-xs font-code font-semibold bg-[#14213d] text-white hover:bg-[#ff6b5b] hover:text-[#14213d] border border-[#14213d] transition-colors cursor-pointer"
            >
              Trace C[{sceneState.selectedCell[0]}][{sceneState.selectedCell[1]}]
            </button>
          </div>
        )}

        {slide.id === 6 && (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              {(['shallow', 'medium', 'deep'] as const).map((depth) => (
                <button
                  key={depth}
                  type="button"
                  onClick={() =>
                    onUpdateState({ networkDepth: depth, selectedNeuronId: null })
                  }
                  className={`px-2.5 py-1.5 text-xs font-code font-semibold border border-[#14213d] cursor-pointer transition-colors ${
                    sceneState.networkDepth === depth
                      ? 'bg-[#14213d] text-white'
                      : 'bg-white/70 text-[#14213d] hover:bg-[#14213d]/10'
                  }`}
                >
                  {depth === 'shallow'
                    ? '2-6-1'
                    : depth === 'medium'
                    ? '2-8-5-1'
                    : '2-10-7-4-1'}
                </button>
              ))}
            </div>
            {sceneState.selectedNeuronId !== null && (
              <button
                type="button"
                onClick={() => onUpdateState({ selectedNeuronId: null })}
                className="px-2.5 py-1.5 text-xs font-code font-semibold bg-[#ff6b5b] text-[#14213d] border border-[#14213d] cursor-pointer"
              >
                Clear Neuron #{sceneState.selectedNeuronId}
              </button>
            )}
          </div>
        )}

        {slide.id === 7 && (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <label className="flex items-center gap-2.5 text-xs font-code font-semibold text-[#14213d]">
              <Sliders className="w-4 h-4 text-[#ff6b5b]" strokeWidth={2} />
              <span>
                ReLU α: {sceneState.reluAlpha.toFixed(2)} (f(-2) ={' '}
                {evaluateLeakyRelu(-2, sceneState.reluAlpha).toFixed(2)})
              </span>
              <input
                type="range"
                min={0}
                max={0.5}
                step={0.05}
                value={sceneState.reluAlpha}
                onChange={(e) =>
                  onUpdateState({ reluAlpha: Number(e.target.value) })
                }
                className="cpptorch-slider w-24"
              />
            </label>
            <button
              type="button"
              onClick={() =>
                onUpdateState({
                  mseSampleIndex: (sceneState.mseSampleIndex + 1) % 4,
                })
              }
              className="px-3 py-1.5 text-xs font-code font-semibold bg-[#14213d] text-white border border-[#14213d] hover:bg-[#ff6b5b] hover:text-[#14213d] transition-colors cursor-pointer"
            >
              MSE Sample #{sceneState.mseSampleIndex + 1} (L=
              {evaluateMsePoint(0.25 * sceneState.mseSampleIndex, 0).toFixed(3)})
            </button>
          </div>
        )}

        {slide.id === 8 && (
          <div className="space-y-2">
            <div className="flex flex-wrap gap-1.5">
              {(['SGD', 'Adagrad', 'RMSprop', 'Adam'] as OptimizerName[]).map(
                (opt) => {
                  const active = sceneState.selectedOptimizer === opt;
                  return (
                    <button
                      key={opt}
                      type="button"
                      onClick={() => onUpdateState({ selectedOptimizer: opt })}
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-code font-semibold border border-[#14213d] cursor-pointer transition-colors ${
                        active
                          ? 'bg-[#14213d] text-white'
                          : 'bg-white/70 text-[#14213d] hover:bg-[#14213d]/10'
                      }`}
                    >
                      <span
                        className={`w-2 h-2 ${
                          active ? 'bg-[#ff6b5b]' : 'bg-[#14213d]/40'
                        }`}
                      />
                      <span>{opt}</span>
                    </button>
                  );
                }
              )}
            </div>
            <div className="font-code text-xs text-[#14213d]/85">
              {sceneState.selectedOptimizer}: {OPTIMIZER_RULES[sceneState.selectedOptimizer]}
            </div>
          </div>
        )}

        {slide.id === 9 && (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap gap-1.5">
              {(['all', 'Forward', 'Backward', 'Update'] as const).map((st) => (
                <button
                  key={st}
                  type="button"
                  onClick={() => onUpdateState({ loopStage: st })}
                  className={`px-2.5 py-1.5 text-xs font-code font-semibold border border-[#14213d] cursor-pointer transition-colors ${
                    sceneState.loopStage === st
                      ? 'bg-[#14213d] text-white'
                      : 'bg-white/70 text-[#14213d] hover:bg-[#14213d]/10'
                  }`}
                >
                  {st === 'all' ? 'Run Loop' : st}
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2 text-xs font-code font-semibold text-[#14213d]">
              <span>Speed: {sceneState.loopSpeed.toFixed(1)}×</span>
              <input
                type="range"
                min={0.5}
                max={2.5}
                step={0.25}
                value={sceneState.loopSpeed}
                onChange={(e) =>
                  onUpdateState({ loopSpeed: Number(e.target.value) })
                }
                className="cpptorch-slider w-20"
              />
            </label>
          </div>
        )}

        {slide.id === 10 && (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <Database className="w-4 h-4 text-[#ff6b5b]" strokeWidth={2} />
              <select
                aria-label="Dataset selection"
                value={sceneState.datasetType}
                onChange={(e) =>
                  onUpdateState({
                    datasetType: e.target.value as SlideSceneState['datasetType'],
                    resampleTrigger: sceneState.resampleTrigger + 1,
                  })
                }
                className="border border-[#14213d] bg-white px-2.5 py-1.5 font-code text-xs text-[#14213d] cursor-pointer"
              >
                <option value="spiral">Datasets::spiral_data</option>
                <option value="housing">Datasets::california_housing</option>
                <option value="audio">Datasets::gender_audio_data</option>
                <option value="svm">Datasets::svm_data</option>
              </select>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() =>
                  onUpdateState({ scalerFitMode: !sceneState.scalerFitMode })
                }
                className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-code font-semibold border border-[#14213d] cursor-pointer ${
                  sceneState.scalerFitMode
                    ? 'bg-[#ff6b5b] text-[#14213d]'
                    : 'bg-white/70 text-[#14213d]'
                }`}
              >
                <Ruler className="w-3.5 h-3.5" strokeWidth={2} />
                <span>{sceneState.scalerFitMode ? 'Fit Mode: ON' : 'Fit Scaler'}</span>
              </button>
              <button
                type="button"
                onClick={() =>
                  onUpdateState({
                    resampleTrigger: sceneState.resampleTrigger + 1,
                  })
                }
                className="px-2.5 py-1.5 text-xs font-code font-semibold bg-[#14213d] text-white border border-[#14213d] cursor-pointer"
              >
                Resample
              </button>
            </div>
          </div>
        )}

        {slide.id === 12 && (
          <div className="flex flex-wrap items-center gap-1.5">
            {['1 · Header', '2 · Factory', '3 · Export'].map((label, idx) => (
              <button
                key={label}
                type="button"
                onClick={() =>
                  onUpdateState({
                    selectedStep: sceneState.selectedStep === idx ? null : idx,
                  })
                }
                className={`px-2.5 py-1.5 text-xs font-code font-semibold border border-[#14213d] cursor-pointer ${
                  sceneState.selectedStep === idx
                    ? 'bg-[#ff6b5b] text-[#14213d]'
                    : 'bg-white/70 text-[#14213d] hover:bg-[#14213d] hover:text-white'
                }`}
              >
                {label}
              </button>
            ))}
            {sceneState.selectedStep !== null && (
              <button
                type="button"
                onClick={() => onUpdateState({ selectedStep: null })}
                className="px-2.5 py-1.5 text-xs font-code font-semibold border border-[#14213d] bg-[#14213d] text-white cursor-pointer"
              >
                Reset
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
