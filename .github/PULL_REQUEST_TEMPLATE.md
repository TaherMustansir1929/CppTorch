## Summary

<!-- What does this pull request change? Link the issue it closes, e.g. "Closes #42". -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation
- [ ] Build / packaging
- [ ] Refactor

## Area touched

<!-- Which part of the library does this affect? -->

- [ ] Core (`core/matrix.h`, `core/numcpp.h`)
- [ ] Layers / activations
- [ ] Losses
- [ ] Optimizers
- [ ] Model / training loop (`model.h`)
- [ ] Datasets
- [ ] Build system / packaging
- [ ] Docs / tests only

## Does this change the core?

- [ ] **No** — the training loop in `model.h` is unchanged.
- [ ] **Yes** — explain why the core had to change and what could break.

## Checklist

- [ ] The project builds without new warnings.
- [ ] `ctest --output-on-failure` passes.
- [ ] New public API is documented with Doxygen comments.
- [ ] The API reference in `README.md` is updated for new public API.
- [ ] A test covers the new behaviour.
- [ ] `clang-format` was run on the changed files.
- [ ] `CHANGELOG.md` has an entry under *Unreleased*.

## Screenshots or output

<!-- For changes that affect the printed output, paste a before/after snippet. -->
