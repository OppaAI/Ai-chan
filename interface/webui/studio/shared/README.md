# Studio shared assets

Cross-studio frontend primitives. Mounted by every studio backend at **`/shared`**.

## Layout

| Path | Role |
|------|------|
| `css/tokens.css` | Design tokens (`:root` CSS variables) |
| `css/base.css` | Shared chrome: reset, header, buttons, sidebar |
| `graph-bootstrap.js` | D3 helpers (`GraphBoot`) for graph studios |

## Entry scripts

| Studio | Entry JS |
|--------|----------|
| MCP | `mcp-servers.js` |
| KB | `kb-graph.js` |
| DAG / Spec / LTM / STM / Approval | `script.js` (rename to functional names in a later pass) |

## Usage

```html
<link rel="stylesheet" href="shared/css/tokens.css">
<link rel="stylesheet" href="shared/css/base.css">
<link rel="stylesheet" href="static/style.css">

<script src="shared/graph-bootstrap.js"></script>
<script src="static/script.js"></script>
```

Page-specific rules stay in each studio’s `style.css`.
