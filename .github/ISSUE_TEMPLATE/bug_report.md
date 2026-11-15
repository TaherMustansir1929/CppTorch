name: Bug report

description: Report a problem with CppTorch
labels: bug
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to report a bug. Please fill in the sections
        below — a minimal reproducer is worth more than a long description.

  - type: input
    id: version
    attributes:
      label: CppTorch version
      placeholder: v0.1.0 or a commit hash
    validations:
      required: true

  - type: dropdown
    id: platform
    attributes:
      label: Platform
      options:
        - Linux
        - macOS
        - Windows
    validations:
      required: true

  - type: input
    id: compiler
    attributes:
      label: Compiler and version
      placeholder: g++ 13.2.0 / clang++ 18.1.0 / MSVC 19.40
    validations:
      required: true

  - type: input
    id: cmake
    attributes:
      label: CMake version
      placeholder: 3.28
    validations:
      required: true

  - type: input
    id: blas
    attributes:
      label: BLAS provider and version
      placeholder: OpenBLAS 0.3.26
    validations:
      required: true

  - type: textarea
    id: reproducer
    attributes:
      label: Minimal reproducer
      description: The smallest program that shows the problem.
      render: cpp
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected behaviour
    validations:
      required: true

  - type: textarea
    id: actual
    attributes:
      label: Actual behaviour
      description: Include the full output and any compiler warnings.
    validations:
      required: true

  - type: textarea
    id: notes
    attributes:
      label: Additional context
      description: Anything else that might help, such as data or screenshots.
