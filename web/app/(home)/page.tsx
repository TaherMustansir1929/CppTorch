import Link from 'next/link';
import { CodeSample } from '@/components/code-sample';
import { GitHubIcon, LinkedInIcon, YouTubeIcon } from '@/components/social-icons';
import { repoUrl, socialLinks } from '@/lib/shared';

const installCommand = `git clone https://github.com/TaherMustansir1929/CppTorch.git
cd CppTorch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /your/prefix`;

const quickstart = `#include <cpptorch/cpptorch.h>

int main()
{
    using namespace cpptorch;

    // 1. Data: a 3-class spiral with 100 samples per class.
    auto [X, y] = Datasets::spiral_data(100, 3);

    // 2. Model: 2 -> 64 (ReLU) -> 32 (ReLU) -> 3 (Softmax).
    Model model(ModelType::Classification);

    model.add_layer({.n_inputs = X.cols(), .n_neurons = 64, .activation_type = ActivationType::ReLU});
    model.add_layer({.n_inputs = 64, .n_neurons = 32, .activation_type = ActivationType::ReLU});
    model.add_layer({.n_inputs = 32, .n_neurons = 3, .activation_type = ActivationType::Softmax});

    // 3. Train with Adam and a decaying learning rate.
    model.set_optimizer({.optmzr_type = OptimizerType::Adam, .learning_rate = 0.01, .decay = 1e-3});
    model.train(X, y, /*n_epochs=*/100);

    // 4. Evaluate.
    auto result = model.evaluate(X, y);
    std::cout << "Accuracy: " << result.accuracy * 100.0 << "%\\n";

    return 0;
}`;

const features = [
  {
    title: 'numpy-style arrays',
    description:
      'Matrix with numpy broadcasting, flat contiguous storage and BLAS-backed matrix products.',
  },
  {
    title: 'Full training loop',
    description:
      'Model::train() runs forward, backward and update passes with mini-batching, learning-rate decay and L1/L2 regularization.',
  },
  {
    title: 'Composable components',
    description:
      'Activations, losses and optimizers sit behind a base class and are plugged in through factories.',
  },
  {
    title: 'Performance',
    description:
      'OpenMP-parallel kernels, a fused softmax + cross-entropy gradient, zero-copy mini-batches and buffer reuse.',
  },
  {
    title: 'Header-only',
    description:
      'No ABI to track. Consume it as a CMake package, a subdirectory or a single copied header folder.',
  },
  {
    title: 'Tested',
    description:
      'A 15-test CTest suite that runs on Linux, macOS and Windows, plus the California housing and audio datasets.',
  },
];

const socials = [
  { label: 'GitHub', href: socialLinks.github, icon: GitHubIcon },
  { label: 'LinkedIn', href: socialLinks.linkedin, icon: LinkedInIcon },
  { label: 'YouTube', href: socialLinks.youtube, icon: YouTubeIcon },
];

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="relative overflow-hidden border-b">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_60%_at_50%_0%,var(--cpptorch-coral)_0%,transparent_70%)] opacity-15"
        />
        <div className="relative mx-auto flex w-full max-w-(--fd-layout-width) flex-col items-center gap-6 px-4 py-20 text-center md:py-28">
          <h1 className="max-w-3xl text-4xl font-black tracking-tight text-balance md:text-6xl">
            CppTorch - A minimal neural networks library in C++
          </h1>

          <p className="max-w-2xl text-lg text-pretty text-fd-muted-foreground">
            numpy-style arrays, dense layers, activations, losses, optimizers and
            a complete training loop — built with OpenMP and OpenBLAS.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/docs"
              className="inline-flex h-10 items-center rounded-lg bg-fd-primary px-5 text-sm font-semibold text-fd-primary-foreground transition-opacity hover:opacity-90"
            >
              Read the docs
            </Link>
            <a
              href={repoUrl}
              target="_blank"
              rel="noreferrer noopener"
              className="inline-flex h-10 items-center gap-2 rounded-lg border bg-fd-card px-5 text-sm font-semibold transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground"
            >
              <GitHubIcon className="size-4" />
              Star on GitHub
            </a>
          </div>

          <span className="inline-flex items-center gap-2 rounded-full border bg-fd-card px-3 py-1 text-xs font-medium text-fd-muted-foreground">
            <span className="size-1.5 rounded-full bg-fd-accent" />
            C++20 · Header-only · BLAS accelerated
          </span>

          <div className="mt-4 w-full max-w-2xl text-left">
            <CodeSample code={installCommand} lang="bash" title="Install from source" />
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="mx-auto w-full max-w-(--fd-layout-width) px-4 py-16">
        <h2 className="text-center text-2xl font-bold tracking-tight md:text-3xl">
          Everything a small deep learning framework needs
        </h2>
        <p className="mx-auto mt-2 max-w-2xl text-center text-fd-muted-foreground">
          Small enough to read in one sitting, complete enough to train a real model.
        </p>

        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="rounded-xl border bg-fd-card p-5 transition-colors hover:border-fd-accent"
            >
              <h3 className="font-semibold">{feature.title}</h3>
              <p className="mt-1.5 text-sm text-fd-muted-foreground">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Quick start */}
      <section className="border-y bg-fd-card/40">
        <div className="mx-auto grid w-full max-w-(--fd-layout-width) gap-8 px-4 py-16 lg:grid-cols-2 lg:items-center">
          <div>
            <h2 className="text-2xl font-bold tracking-tight md:text-3xl">Quick start</h2>
            <p className="mt-3 text-fd-muted-foreground">
              One header, one model, one call to <code className="rounded bg-fd-muted px-1 py-0.5 text-sm">train()</code>.
              The library handles mini-batching, learning-rate decay, regularization and the
              validation loop for you.
            </p>
            <ul className="mt-5 space-y-2 text-sm text-fd-muted-foreground">
              <li>· Ship as a CMake package, a subdirectory or vendored headers</li>
              <li>· Bring your own activation, loss or optimizer through the registries</li>
              <li>· Datasets ship with the library — California housing and Kaggle audio</li>
            </ul>
            <Link
              href="/docs/getting-started"
              className="mt-6 inline-flex h-10 items-center rounded-lg border bg-fd-card px-5 text-sm font-semibold transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground"
            >
              Installation guide
            </Link>
          </div>

          <CodeSample code={quickstart} lang="cpp" title="examples/quickstart.cpp" />
        </div>
      </section>

      {/* Footer */}
      <footer className="mx-auto flex w-full max-w-(--fd-layout-width) flex-col gap-6 px-4 py-12 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="font-semibold">CppTorch</p>
          <p className="mt-1 text-sm text-fd-muted-foreground">
            A minimal neural networks library in C++20. Released under the MIT License.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {socials.map((social) => (
            <a
              key={social.label}
              href={social.href}
              target="_blank"
              rel="noreferrer noopener"
              aria-label={social.label}
              title={social.label}
              className="inline-flex size-9 items-center justify-center rounded-lg border bg-fd-card text-fd-muted-foreground transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground"
            >
              <social.icon className="size-4" />
            </a>
          ))}
        </div>
      </footer>
    </div>
  );
}
