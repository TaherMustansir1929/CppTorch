import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import gsap from 'gsap';
import {
  ChevronLeft,
  ChevronRight,
  LayoutGrid,
} from 'lucide-react';
import { SLIDES } from './data/slides';
import {
  clampSlideIndex,
  nextSlideIndex,
  prevSlideIndex,
  slideProgressPercent,
} from './utils/deckMath';
import type { SceneCallbacks, SlideSceneState } from './three/sceneBuilders';
import { ThreeViewport } from './components/ThreeViewport';
import { SlideContent } from './components/SlideContent';
import { SlideOverviewModal } from './components/SlideOverviewModal';

const INITIAL_SCENE_STATE: SlideSceneState = {
  pulseTrigger: 0,
  selectedFeatureIndex: null,
  selectedPillarIndex: null,
  matrixSize: 3,
  selectedCell: [0, 0],
  benchmarkMode: 'epoch',
  replayTrigger: 0,
  networkDepth: 'medium',
  selectedNeuronId: null,
  reluAlpha: 0,
  mseSampleIndex: 0,
  selectedOptimizer: 'Adam',
  loopStage: 'all',
  loopSpeed: 1,
  datasetType: 'spiral',
  scalerFitMode: false,
  resampleTrigger: 0,
  exampleFocus: 'both',
  snapSvmPlaneTrigger: 0,
  selectedStep: null,
  burstTrigger: 0,
  selectedAuthorIndex: null,
};

export default function App() {
  const [slideIndex, setSlideIndex] = useState(0);
  const [sceneState, setSceneState] = useState<SlideSceneState>(INITIAL_SCENE_STATE);
  const [showOverview, setShowOverview] = useState(false);
  const [navbarVisible, setNavbarVisible] = useState(false);
  const [navFocusWithin, setNavFocusWithin] = useState(false);

  const slidePanelRef = useRef<HTMLDivElement | null>(null);
  const navContainerRef = useRef<HTMLDivElement | null>(null);
  const currentSlide = SLIDES[slideIndex];
  const totalSlides = SLIDES.length;
  const progressPct = slideProgressPercent(slideIndex, totalSlides);
  const isNavbarOpen = navbarVisible || navFocusWithin || showOverview;

  const updateSceneState = useCallback((patch: Partial<SlideSceneState>) => {
    setSceneState((prev) => ({ ...prev, ...patch }));
  }, []);

  const goToSlide = useCallback(
    (targetIdx: number) => {
      setSlideIndex(clampSlideIndex(targetIdx, totalSlides));
    },
    [totalSlides]
  );

  const handleNext = useCallback(() => {
    setSlideIndex((cur) => nextSlideIndex(cur, totalSlides));
  }, [totalSlides]);

  const handlePrev = useCallback(() => {
    setSlideIndex((cur) => prevSlideIndex(cur, totalSlides));
  }, [totalSlides]);

  // Animate slide content panel on slide change
  useEffect(() => {
    const el = slidePanelRef.current;
    if (!el) return;
    const prefersReduced = window.matchMedia?.(
      '(prefers-reduced-motion: reduce)'
    ).matches;
    if (prefersReduced) return;

    const tween = gsap.fromTo(
      el,
      { opacity: 0.2, y: 12 },
      { opacity: 1, y: 0, duration: 0.32, ease: 'power2.out' }
    );
    return () => {
      tween.kill();
    };
  }, [slideIndex]);

  // Auto-hide top navbar unless cursor approaches top edge
  useEffect(() => {
    const REVEAL_THRESHOLD_Y = 48;
    const HIDE_THRESHOLD_Y = 104;

    const onMouseMove = (e: MouseEvent) => {
      if (e.clientY <= REVEAL_THRESHOLD_Y) {
        setNavbarVisible(true);
      } else if (e.clientY > HIDE_THRESHOLD_Y) {
        setNavbarVisible(false);
      }
    };

    const onMouseLeave = () => {
      setNavbarVisible(false);
    };

    window.addEventListener('mousemove', onMouseMove, { passive: true });
    document.addEventListener('mouseleave', onMouseLeave);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseleave', onMouseLeave);
    };
  }, []);

  // Keyboard navigation
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isInteractiveTarget =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'SELECT' ||
          target.tagName === 'TEXTAREA');
      if (isInteractiveTarget) return;

      if (e.key === 'Escape' && showOverview) {
        setShowOverview(false);
        return;
      }
      if (e.key === 'o' || e.key === 'O') {
        setShowOverview((v) => !v);
        return;
      }
      if (showOverview) return;

      if (e.key === 'ArrowRight' || e.key === 'PageDown') {
        e.preventDefault();
        handleNext();
      } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        e.preventDefault();
        handlePrev();
      } else if (e.key === ' ' && target?.tagName !== 'BUTTON') {
        e.preventDefault();
        handleNext();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleNext, handlePrev, showOverview]);

  const sceneCallbacks = useMemo<SceneCallbacks>(
    () => ({
      onTriggerPulse: () =>
        setSceneState((prev) => ({ ...prev, pulseTrigger: prev.pulseTrigger + 1 })),
      onSelectFeature: (idx) =>
        setSceneState((prev) => ({
          ...prev,
          selectedFeatureIndex: prev.selectedFeatureIndex === idx ? null : idx,
        })),
      onSelectPillar: (idx) =>
        setSceneState((prev) => ({
          ...prev,
          selectedPillarIndex: prev.selectedPillarIndex === idx ? null : idx,
        })),
      onSelectMatrixCell: (row, col) =>
        setSceneState((prev) => ({ ...prev, selectedCell: [row, col] })),
      onSelectNeuron: (id) =>
        setSceneState((prev) => ({
          ...prev,
          selectedNeuronId: prev.selectedNeuronId === id ? null : id,
        })),
      onSelectOptimizer: (opt) =>
        setSceneState((prev) => ({ ...prev, selectedOptimizer: opt })),
      onSelectLoopStage: (stage) =>
        setSceneState((prev) => ({ ...prev, loopStage: stage })),
      onSelectStep: (step) =>
        setSceneState((prev) => ({
          ...prev,
          selectedStep: prev.selectedStep === step ? null : step,
        })),
      onSelectAuthor: (idx) =>
        setSceneState((prev) => ({
          ...prev,
          selectedAuthorIndex: prev.selectedAuthorIndex === idx ? null : idx,
        })),
      onSnapSvm: () =>
        setSceneState((prev) => ({
          ...prev,
          snapSvmPlaneTrigger: prev.snapSvmPlaneTrigger + 1,
        })),
      onToggleFitMode: () =>
        setSceneState((prev) => ({
          ...prev,
          scalerFitMode: !prev.scalerFitMode,
        })),
      onResampleData: () =>
        setSceneState((prev) => ({
          ...prev,
          resampleTrigger: prev.resampleTrigger + 1,
        })),
      onSelectMseSample: (idx) =>
        setSceneState((prev) => ({ ...prev, mseSampleIndex: idx })),
    }),
    []
  );

  return (
    <div
      data-slide={currentSlide.id}
      className="relative min-h-screen lg:h-screen flex flex-col bg-[#f4ede0] text-[#14213d] overflow-x-hidden"
    >
      {/* Top-edge hover / touch hotzone to summon auto-hidden navbar */}
      <div
        aria-hidden="true"
        onMouseEnter={() => setNavbarVisible(true)}
        onClick={() => setNavbarVisible(true)}
        className="fixed top-0 inset-x-0 h-3 z-30"
      />

      {/* Auto-Hiding Top Navbar + Progress Stripe */}
      <div
        ref={navContainerRef}
        data-testid="auto-hide-navbar"
        data-visible={isNavbarOpen ? 'true' : 'false'}
        onMouseEnter={() => setNavbarVisible(true)}
        onFocusCapture={() => setNavFocusWithin(true)}
        onBlurCapture={(e) => {
          if (!navContainerRef.current?.contains(e.relatedTarget as Node | null)) {
            setNavFocusWithin(false);
          }
        }}
        className={`fixed top-0 left-0 right-0 z-40 transition-transform duration-200 ease-out shadow-[0_6px_20px_rgba(20,33,61,0.14)] ${
          isNavbarOpen ? 'translate-y-0' : '-translate-y-full'
        }`}
      >
        <div className="h-1 w-full bg-[#14213d]/15">
          <div
            className="h-full bg-[#ff6b5b] transition-all duration-300"
            style={{ width: `${progressPct}%` }}
          />
        </div>

        <header className="flex items-center justify-between gap-4 sm:gap-8 px-4 sm:px-8 py-3 border-b-2 border-[#14213d] bg-[#f4ede0]">
          <button
            type="button"
            onClick={() => goToSlide(0)}
            className="text-lg sm:text-xl font-black tracking-tight text-[#14213d] whitespace-nowrap shrink-0 cursor-pointer hover:text-[#ff6b5b] transition-colors"
          >
            CppTorch
          </button>

          <div className="flex items-center gap-3 text-xs sm:text-sm font-code text-[#14213d] whitespace-nowrap">
            <span className="font-bold tabular-nums">
              {String(currentSlide.id).padStart(2, '0')} / {totalSlides}
            </span>
            <span aria-hidden="true" className="text-[#ff6b5b] font-bold">
              ·
            </span>
            <span className="hidden sm:inline font-medium">
              {currentSlide.sectionLabel}
            </span>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setShowOverview(true)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-code font-semibold bg-[#14213d] text-white border border-[#14213d] hover:bg-[#ff6b5b] hover:text-[#14213d] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <LayoutGrid className="w-3.5 h-3.5" strokeWidth={2} />
              <span>All Slides</span>
            </button>
          </div>
        </header>
      </div>

      {/* Main Presentation Stage */}
      <main className="flex-1 min-h-0 px-4 sm:px-8 py-4 lg:py-5 flex flex-col gap-3">
        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
          {/* Left Column: Slide Copy, Equations, Code & Interactive Controls */}
          <section
            ref={slidePanelRef}
            aria-label={`Slide ${currentSlide.id}: ${currentSlide.title}`}
            className="lg:col-span-5 flex flex-col justify-between bg-[#f4ede0] border-2 border-[#14213d] p-5 sm:p-6 overflow-y-auto"
          >
            <SlideContent
              slide={currentSlide}
              sceneState={sceneState}
              onUpdateState={updateSceneState}
              onRestartDeck={() => goToSlide(0)}
            />
          </section>

          {/* Right Column: Interactive 3D Viewport */}
          <section
            aria-label="Interactive 3D Visualization"
            className="lg:col-span-7 min-h-[340px] lg:min-h-0 h-full"
          >
            <ThreeViewport
              slideId={currentSlide.id}
              sceneState={sceneState}
              callbacks={sceneCallbacks}
              interactionHint={currentSlide.interactionHint}
            />
          </section>
        </div>
      </main>

      {/* Bottom Slide Navigation Footer */}
      <footer className="shrink-0 flex items-center justify-between gap-3 px-4 sm:px-8 py-3 border-t-2 border-[#14213d] bg-[#f4ede0]">
        <button
          type="button"
          onClick={handlePrev}
          disabled={slideIndex === 0}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs sm:text-sm font-bold border-2 border-[#14213d] bg-white/80 text-[#14213d] hover:bg-[#14213d] hover:text-white disabled:opacity-35 disabled:pointer-events-none transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <ChevronLeft className="w-4 h-4" strokeWidth={2} />
          <span>Previous</span>
        </button>

        {/* Direct 1..13 slide indicator strip */}
        <nav
          aria-label="Slide Navigation"
          className="hidden md:flex items-center gap-1"
        >
          {SLIDES.map((s, idx) => {
            const active = idx === slideIndex;
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => goToSlide(idx)}
                aria-label={`Go to Slide ${s.id}: ${s.title}`}
                aria-current={active ? 'step' : undefined}
                className={`w-7 h-7 font-code text-xs font-bold border transition-colors cursor-pointer ${
                  active
                    ? 'bg-[#ff6b5b] text-[#14213d] border-[#14213d]'
                    : 'bg-white/65 text-[#14213d]/80 border-[#14213d]/25 hover:border-[#14213d] hover:text-[#14213d]'
                }`}
              >
                {s.id}
              </button>
            );
          })}
        </nav>

        <div className="md:hidden font-code text-xs font-bold tabular-nums text-[#14213d]">
          Slide {currentSlide.id} of {totalSlides}
        </div>

        <button
          type="button"
          onClick={handleNext}
          disabled={slideIndex === totalSlides - 1}
          className="inline-flex items-center gap-1.5 px-4 py-2 text-xs sm:text-sm font-bold border-2 border-[#14213d] bg-[#14213d] text-white hover:bg-[#ff6b5b] hover:text-[#14213d] disabled:opacity-35 disabled:pointer-events-none transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <span>Next</span>
          <ChevronRight className="w-4 h-4" strokeWidth={2} />
        </button>
      </footer>

      {showOverview && (
        <SlideOverviewModal
          slides={SLIDES}
          currentIndex={slideIndex}
          onSelectSlide={goToSlide}
          onClose={() => setShowOverview(false)}
        />
      )}
    </div>
  );
}
