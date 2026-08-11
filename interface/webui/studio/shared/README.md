# Studio shared assets

Cross-studio frontend primitives. Mounted by every studio backend at **`/shared`**.

## Layout (industry convention)

| Path | Role |
|------|------|
| `css/tokens.css` | Design tokens (`:root` CSS variables) |
| `css/base.css` | Shared chrome: reset, header, buttons, sidebar |
| `graph-bootstrap.js` | D3 helpers (`GraphBoot`) used by graph studios |

Folder name is **`shared/`** (not `_shared`). That matches Feature-Sliced Design, Nx scopes, and common React monorepo practice. The underscore form is mainly a Next.js *private folder* convention and is not needed here.

## Usage in a studio page

```html
<link rel="stylesheet" href="shared/css/tokens.css">
<link rel="stylesheet" href="shared/css/base.css">
<link rel="stylesheet" href="static/style.css">  <!-- page-only overrides -->

<script src="shared/graph-bootstrap.js"></script>
<script src="static/script.js"></script>
```

Keep page-specific rules (DAG edges, STM rack, LTM force nodes, Spec drawer) in each studio’s own `style.css`.
