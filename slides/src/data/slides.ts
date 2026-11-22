export interface AuthorEntry {
  name: string;
  roll: string;
  role: string;
}

export interface SlideDefinition {
  id: number;
  slug: string;
  sectionLabel: string;
  title: string;
  subtitle: string;
  layout: 'hero' | 'split' | 'cards' | 'benchmark' | 'closing';
  bigIdea: string;
  bullets?: string[];
  codeTitle?: string;
  codeSnippet?: string;
  equation?: string;
  equationLabel?: string;
  highlightBadge?: string;
  cards?: Array<{
    title: string;
    detail: string;
    tag?: string;
  }>;
  metrics?: Array<{
    label: string;
    value: string;
    accent?: boolean;
  }>;
  authors?: AuthorEntry[];
  pitch?: string;
  transitionNote: string;
  interactionHint: string;
}

export const UNIVERSITY_NAME = 'FAST NUCES';
export const CLASS_SECTION = 'BAI-3C';
export const DOCS_SITE_URL = 'https://cpptorch.vercel.app';
export const GITHUB_REPO_URL = 'https://github.com/TaherMustansir1929/CppTorch';

export const AUTHORS: AuthorEntry[] = [
  { name: 'Taher', roll: '25K-0119', role: 'Team Lead' },
  { name: 'Kashif', roll: '25K-0018', role: 'Feature Engineer' },
  { name: 'Yahya', roll: '25K-0078', role: 'Foundational Developer' },
];

export const SLIDES: SlideDefinition[] = [
  {
    id: 1,
    slug: 'welcome',
    sectionLabel: '01 · Reveal',
    title: 'CppTorch',
    subtitle: 'A Minimal Neural-Networks Library in C++20',
    layout: 'hero',
    bigIdea: "Training a neural network from scratch, because apparently I wasn't suffering enough.",
    pitch: "Training a neural network from scratch, because apparently I wasn't suffering enough.",
    authors: AUTHORS,
    transitionNote:
      'Before we look under the hood, let us define what CppTorch is and why it exists.',
    interactionHint: 'Click the 3D neural cluster or Pulse Neural Cluster repeatedly to inflate nodes.',
  },
  {
    id: 2,
    slug: 'what-is-cpptorch',
    sectionLabel: '02 · Overview',
    title: 'What Is CppTorch?',
    subtitle: 'The elevator pitch',
    layout: 'split',
    bigIdea: 'A transparent, header-only C++20 deep learning core with zero framework bloat.',
    bullets: [
      'Header-only C++20 neural-networks library.',
      'Depends only on OpenMP + CBLAS (OpenBLAS).',
      'NumPy-style Matrix with broadcasting and BLAS-backed products.',
      'Full training loop: forward, backward, and update with mini-batching.',
      'Educational by design: easy to read, easy to extend.',
    ],
    codeTitle: 'Stack Overview',
    codeSnippet: `#include <cpptorch/cpptorch.h>

// Dependencies: C++20 + OpenMP + OpenBLAS
// Zero Python runtime, zero protobuf graphs
using namespace cpptorch;`,
    transitionNote:
      'These features follow a strict design philosophy. Let us look at the rules that shaped the library.',
    interactionHint: 'Click any floating 3D architecture card to focus its layer in the stack.',
  },
  {
    id: 3,
    slug: 'design-philosophy',
    sectionLabel: '03 · Philosophy',
    title: 'Design Philosophy',
    subtitle: 'Built for learning, not production',
    layout: 'cards',
    bigIdea: 'Three architectural rules keep the codebase readable from top to bottom.',
    cards: [
      {
        title: 'Header-Only',
        detail: 'No ABI friction or complex linking—drop include/cpptorch/ into any C++20 target.',
        tag: '01',
      },
      {
        title: 'Extension Points',
        detail: 'Add activations, losses, and optimizers via factories without touching model.h.',
        tag: '02',
      },
      {
        title: 'Minimal Core',
        detail: 'Matrix primitives plus a clean training loop—zero hidden autograd graph magic.',
        tag: '03',
      },
    ],
    transitionNote:
      'Everything starts with the array primitive. Without a solid Matrix, there is no network.',
    interactionHint: 'Click any 3D pillar to focus the camera and inspect its architectural principle.',
  },
  {
    id: 4,
    slug: 'matrix-numcpp',
    sectionLabel: '04 · Primitives',
    title: 'Matrix & NumCpp',
    subtitle: 'NumPy-style array primitives in C++20',
    layout: 'split',
    bigIdea: 'Contiguous row-major memory enables both zero-copy views and native BLAS products.',
    bullets: [
      'Matrix: row-major flat contiguous storage, BLAS-friendly layout.',
      'Single-row and single-column broadcasting rules.',
      'NumCpp ops: dot, sin, cos, exp, log, one_hot, argmax, mean.',
      'Zero-copy MatrixView slices for mini-batches.',
    ],
    equation: 'C[i][j] = Σ_k A[i][k] · B[k][j]',
    equationLabel: 'OpenMP-parallel element-wise kernels + CBLAS dgemm',
    codeTitle: 'include/cpptorch/core/matrix.h',
    codeSnippet: `Matrix A(rows, inner), B(inner, cols);
A(i, k) = value;
Matrix C = NumCpp::dot(A, B);
MatrixView batch = X.slice_rows(start, end);`,
    transitionNote:
      'Arrays store data, but the real speed happens when those arrays compute in parallel.',
    interactionHint: 'Adjust matrix dimension or select an output cell to trace row A × column B.',
  },
  {
    id: 5,
    slug: 'performance-engine',
    sectionLabel: '05 · Performance',
    title: 'Performance Engine',
    subtitle: 'OpenMP multithreading + OpenBLAS — the heart of CppTorch speed',
    layout: 'benchmark',
    bigIdea: '15× faster training loops on identical hardware using the same algorithm.',
    equation: 'T_single / T_parallel ≈ 15×',
    equationLabel: 'Measured on identical hardware, same algorithm, threading & BLAS enabled',
    highlightBadge: '15× faster training loops',
    metrics: [
      { label: 'Matrix dot (1024×1024)', value: '420 ms → 28 ms', accent: true },
      { label: 'Element-wise ReLU (1M)', value: '95 ms → 7 ms', accent: false },
      { label: 'Full epoch (spiral ×100)', value: '12.4 s → 0.83 s', accent: true },
    ],
    transitionNote:
      'This is the engine under the hood. Now let us see how layers use it to transform data.',
    interactionHint: 'Switch between Single Operation and Full Epoch views to replay the 3D bar race.',
  },
  {
    id: 6,
    slug: 'layers',
    sectionLabel: '06 · Architecture',
    title: 'Layers',
    subtitle: 'Dense, Dropout, and the Layer Wrapper',
    layout: 'split',
    bigIdea: 'Composable layers cache forward activations for clean analytical backpropagation.',
    bullets: [
      'Layer_Dense: output = inputs * weights + biases.',
      'Layer_Dropout: inverted dropout scaled during training.',
      'Layer_Wrapper: bundles dense + optional dropout + activation.',
      'Forward pass caches inputs for exact backward gradients.',
    ],
    codeTitle: 'Layer Hyperparameters',
    codeSnippet: `HyperParameters_Layer_Wrapper h1{
  .inputs = 2,
  .neurons = 32,
  .activation = ActivationType::ReLU,
  .dropout_rate = 0.0
};
model.add_layer(h1);`,
    transitionNote:
      'Layers transform data, but they need an objective. Activations shape predictions; losses score error.',
    interactionHint: 'Change network depth or click any 3D neuron to trace its connected weight edges.',
  },
  {
    id: 7,
    slug: 'activations-losses',
    sectionLabel: '07 · Objectives',
    title: 'Activations & Losses',
    subtitle: 'Shaping predictions and scoring error',
    layout: 'split',
    bigIdea: 'Fused Softmax + Cross-Entropy avoids materializing the full Jacobian matrix.',
    bullets: [
      'Activations: Linear, ReLU f(x) = max(0, x), and Softmax.',
      'Losses: Mean Squared Error and Categorical Cross-Entropy.',
      'Fused Softmax + CCE backward pass computes (ŷ − y) / N directly.',
      'Loss::accuracy() evaluates top-1 classification predictions.',
    ],
    equation: 'ReLU(x) = max(0, x)   |   MSE = (1/n) Σ (y_pred − y_true)²',
    equationLabel: 'Non-linear shaping & scalar objective scoring',
    codeTitle: 'Fused Gradient Path',
    codeSnippet: `// Fused Softmax + Cross-Entropy gradient:
// Avoids N×K×K Jacobian allocation
dinputs = y_pred - y_true_one_hot;
dinputs /= static_cast<double>(samples);`,
    transitionNote:
      'With predictions and a loss in hand, we need an optimizer to actually move the weights.',
    interactionHint: 'Drag the ReLU leakiness slider to morph the 3D activation surface in real time.',
  },
  {
    id: 8,
    slug: 'optimizers',
    sectionLabel: '08 · Optimization',
    title: 'Optimizers',
    subtitle: 'From SGD to Adam',
    layout: 'split',
    bigIdea: 'Consistent learning-rate decay and adaptive moment state across four classic optimizers.',
    bullets: [
      'Optimizer_SGD: vanilla or momentum-accelerated descent.',
      'Optimizer_Adagrad: per-parameter squared-gradient accumulation.',
      'Optimizer_RMSprop: exponential moving average of squared gradients.',
      'Optimizer_Adam: momentum + adaptive rates + bias correction.',
    ],
    equation: 'm = β₁·m + (1−β₁)·g   |   θ = θ − lr · m̂ / (√v̂ + ε)',
    equationLabel: 'Adam bias-corrected parameter update',
    codeTitle: 'Optimizer Configuration',
    codeSnippet: `model.set_optimizer(OptimizerType::Adam, {
  .learning_rate = 0.005,
  .decay = 1e-4,
  .beta_1 = 0.9, .beta_2 = 0.999
});`,
    transitionNote:
      'Optimizers update parameters, but Model orchestrates the entire epoch.',
    interactionHint: 'Select SGD, Adagrad, RMSprop, or Adam to isolate its 3D convergence trajectory.',
  },
  {
    id: 9,
    slug: 'model-training-loop',
    sectionLabel: '09 · Orchestration',
    title: 'Model & Training Loop',
    subtitle: 'Forward, backward, update — every epoch',
    layout: 'split',
    bigIdea: 'One deterministic loop executes mini-batches, regularization, and validation metrics.',
    bullets: [
      'Model::train() executes Forward → Backward → Update per batch.',
      'Mini-batch slicing via zero-copy MatrixView.',
      'Supports LR decay, L1/L2 weight regularization, and validation split.',
      'EvaluationResult tracks loss, accuracy, RMSE, and .display().',
    ],
    codeTitle: 'Model::train() Invocation',
    codeSnippet: `EvaluationResult res = model.train(
  X_train, y_train,
  /*epochs=*/100, /*batch_size=*/64,
  /*X_val=*/&X_val, /*y_val=*/&y_val
);
res.display();`,
    transitionNote:
      'A model needs data to learn. CppTorch bundles datasets and a scaler to prepare them.',
    interactionHint: 'Inspect Forward, Backward, or Update stages and tune the loop playback speed.',
  },
  {
    id: 10,
    slug: 'datasets-scaler',
    sectionLabel: '10 · Data Pipeline',
    title: 'Datasets & StandardScaler',
    subtitle: 'Loading and preprocessing data',
    layout: 'split',
    bigIdea: 'Built-in synthetic and CSV benchmarks flow cleanly through train-only normalization.',
    bullets: [
      'Bundled: spiral_data, sine_data, california_housing, gender_audio.',
      'Added: svm_data() (x1, x2, label ±1) & load_navier_stokes().',
      'StandardScaler: fit on training split, transform test unchanged.',
      'Resources resolver: set_dir → env var → compile def → ./resources/.',
    ],
    codeTitle: 'Preprocessing Pipeline',
    codeSnippet: `auto [X_train, X_test, y_train, y_test] =
  Datasets::train_test_split(data, 0.8);
StandardScaler scaler;
X_train = scaler.fit_transform(X_train);
X_test  = scaler.transform(X_test);`,
    transitionNote:
      'Enough theory—let us see CppTorch solve real problems: an SVM and a Navier-Stokes flow field.',
    interactionHint: 'Switch datasets or toggle Fit Mode to watch samples normalize inside the 3D scaler.',
  },
  {
    id: 11,
    slug: 'examples',
    sectionLabel: '11 · Case Studies',
    title: 'Examples: Neural SVM & Navier-Stokes',
    subtitle: 'Classification and PDE field approximation',
    layout: 'cards',
    bigIdea: 'The same minimal core trains both decision boundaries and fluid vorticity surrogates.',
    cards: [
      {
        title: 'Neural SVM',
        detail:
          'Architecture 2 → 32 ReLU → 16 ReLU → 1 Linear trained with MSE on ±1 targets from svm_dataset.csv.',
        tag: '98.3% Test Accuracy',
      },
      {
        title: 'Navier-Stokes 2D',
        detail:
          'Maps (x, y, t) → (ω, u, v) on time-ordered frames from ns2d_subset.csv with PDE residual diagnostics.',
        tag: 'R² & Vorticity RMS',
      },
    ],
    metrics: [
      { label: 'SVM Test Accuracy', value: '98.3%', accent: true },
      { label: 'SVM Precision / F1', value: '0.98 / 0.98', accent: false },
      { label: 'NS2D Continuity RMS', value: '0.014', accent: true },
    ],
    transitionNote:
      'These examples work because the library is open for extension. Here is how you add your own component.',
    interactionHint: 'Snap the SVM hyperplane or switch focus between the Neural SVM and Fluid Vorticity grid.',
  },
  {
    id: 12,
    slug: 'extending-cpptorch',
    sectionLabel: '12 · Extensibility',
    title: 'Extending CppTorch',
    subtitle: 'Add components without touching the core',
    layout: 'split',
    bigIdea: 'Golden rule: never modify model.h or include/cpptorch/core/ when adding features.',
    bullets: [
      '1. Create Header: derive from Activation, Loss, or Optimizer.',
      '2. Register in Factory: add enum value and bind in registry.',
      '3. Export in cpptorch.h: expose header in umbrella include.',
      'Zero changes required in training loop or matrix engine.',
    ],
    codeTitle: 'activation_registry.h',
    codeSnippet: `inline std::unique_ptr<Activation>
make_activation(ActivationType type) {
  switch (type) {
    case ActivationType::Swish:
      return std::make_unique<Activation_Swish>();
  }
}`,
    transitionNote: 'That completes the tour. Thank you for exploring CppTorch with us.',
    interactionHint: 'Click step 1, 2, or 3 in the 3D workflow to inspect its extension hook.',
  },
  {
    id: 13,
    slug: 'thank-you',
    sectionLabel: '13 · Closing',
    title: 'Thank You!',
    subtitle: 'Questions & Discussion',
    layout: 'closing',
    bigIdea: 'CppTorch — A Minimal Neural-Networks Library in C++20',
    authors: AUTHORS,
    codeSnippet: 'github.com/TaherMustansir1929/CppTorch  ·  include/cpptorch/cpptorch.h',
    transitionNote: 'End of presentation. Use Replay Deck to return to Slide 1.',
    interactionHint: 'Click the CppTorch 3D emblem or select a team member to emit their colored signal burst.',
  },
];
