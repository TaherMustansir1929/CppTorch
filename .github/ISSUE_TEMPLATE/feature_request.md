name: Feature request

description: Suggest a new capability for CppTorch
labels: enhancement
body:
  - type: textarea
    id: problem
    attributes:
      label: Problem
      description: What are you trying to do that the library does not support today?
    validations:
      required: true

  - type: textarea
    id: proposal
    attributes:
      label: Proposed solution
      description: Describe the API you have in mind. A short code sketch is ideal.
    validations:
      required: true

  - type: dropdown
    id: area
    attributes:
      label: Which area does this touch?
      options:
        - Core (Matrix / NumCpp)
        - Layers / activations
        - Losses
        - Optimizers
        - Model / training loop
        - Datasets
        - Build system / packaging
        - Documentation
    validations:
      required: true

  - type: checkboxes
    id: core
    attributes:
      label: Core compatibility
      options:
        - label: This can be added without changing the training loop in `model.h`.
          required: false
        - label: This requires changing the core, and I have explained why above.
          required: false

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives considered
