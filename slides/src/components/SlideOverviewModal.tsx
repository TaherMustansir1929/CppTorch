import { X } from 'lucide-react';
import type { SlideDefinition } from '../data/slides';

interface SlideOverviewModalProps {
  slides: SlideDefinition[];
  currentIndex: number;
  onSelectSlide: (index: number) => void;
  onClose: () => void;
}

export function SlideOverviewModal({
  slides,
  currentIndex,
  onSelectSlide,
  onClose,
}: SlideOverviewModalProps) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Slide Deck Overview"
      className="fixed inset-0 z-50 flex items-center justify-center bg-[#14213d]/75 p-4 sm:p-8"
    >
      <div className="w-full max-w-5xl max-h-[86vh] flex flex-col bg-[#f4ede0] border-2 border-[#14213d]">
        <div className="flex items-center justify-between gap-4 border-b-2 border-[#14213d] px-5 py-3.5 bg-[#14213d] text-white">
          <div>
            <h2 className="text-lg font-bold tracking-tight">
              CppTorch Presentation — All 13 Slides
            </h2>
            <p className="text-xs font-code text-[#f4ede0]/80">
              Click any slide to jump directly · Press Esc to close
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close slide overview"
            className="inline-flex items-center justify-center w-9 h-9 border border-white/30 hover:bg-[#ff6b5b] hover:text-[#14213d] transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" strokeWidth={2} />
          </button>
        </div>

        <div className="p-5 overflow-y-auto grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {slides.map((slide, idx) => {
            const isCurrent = idx === currentIndex;
            return (
              <button
                key={slide.id}
                type="button"
                onClick={() => {
                  onSelectSlide(idx);
                  onClose();
                }}
                className={`text-left p-4 border-2 transition-colors cursor-pointer ${
                  isCurrent
                    ? 'bg-[#14213d] text-white border-[#ff6b5b]'
                    : 'bg-white/75 text-[#14213d] border-[#14213d]/25 hover:border-[#14213d]'
                }`}
              >
                <div className="flex items-center justify-between gap-2 font-code text-xs mb-1.5">
                  <span
                    className={
                      isCurrent ? 'text-[#ff6b5b] font-bold' : 'text-[#14213d]/70'
                    }
                  >
                    {String(slide.id).padStart(2, '0')} / 13
                  </span>
                  <span
                    className={
                      isCurrent ? 'text-[#f4ede0]/80' : 'text-[#14213d]/60'
                    }
                  >
                    {slide.slug}
                  </span>
                </div>
                <div className="font-bold text-base leading-snug">
                  {slide.title}
                </div>
                <div
                  className={`text-xs mt-1 line-clamp-2 ${
                    isCurrent ? 'text-[#f4ede0]/85' : 'text-[#14213d]/75'
                  }`}
                >
                  {slide.subtitle}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
