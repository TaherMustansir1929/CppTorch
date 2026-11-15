# CppTorch documentation site

The documentation website for **CppTorch**, built with
[Fumadocs](https://fumadocs.dev) on Next.js.

## Run locally

```bash
npm install
npm run dev
```

Open http://localhost:3000 — the docs live at `/docs`.

Other scripts:

```bash
npm run build        # production build
npm run start        # serve the production build
npm run types:check  # next typegen && tsc --noEmit
```

## Writing documentation

Pages live in `content/docs/` as MDX files. The sidebar order and the
separators are controlled by the `meta.json` files next to them.

```
content/docs/
├── meta.json            # sidebar order for the root folder
├── index.mdx            # Introduction
├── getting-started.mdx  # Installation & first model
├── model.mdx            # The training API
├── datasets.mdx         # Bundled datasets & preprocessing
├── core/                # Matrix, MatrixView, NumCpp
├── layers/              # Dense, dropout, activations
├── losses.mdx
├── optimizers.mdx
├── architecture.mdx
├── extending.mdx
└── contributing.mdx
```

A page is a normal MDX file with YAML frontmatter:

```mdx
---
title: Getting Started
description: Requirements, installation options and your first CppTorch model.
icon: Rocket
---

## Requirements
```

Useful MDX components: `<Cards>` / `<Card>`, `<Callout>`, `<Tabs>` /
`<Tab>`, `<Steps>`, `<Accordion>` and `<FileTree>`.

## Theme

The brand palette lives in `app/global.css`:

| Token | Value | Usage |
| --- | --- | --- |
| Navy | `#14213D` | Background (dark) / foreground (light) |
| Cream | `#F4EDE0` | Background (light) / foreground (dark) |
| Coral | `#FF6B5B` | Accent |

## Social links

GitHub, LinkedIn and YouTube are configured in `lib/shared.ts` and rendered in
the navbar and the footer. The brand icons are in
`components/social-icons.tsx`.

## Structure

| Route | Description |
| --- | --- |
| `app/(home)` | The landing page and other pages. |
| `app/docs` | The documentation layout and pages. |
| `app/api/search/route.ts` | The Route Handler for search. |
| `lib/source.ts` | Content source adapter, built on `loader()`. |
| `lib/layout.shared.tsx` | Shared layout options (navbar, links). |
| `components/` | MDX components and icons. |

Read the [Fumadocs documentation](https://fumadocs.dev) for the full API.
