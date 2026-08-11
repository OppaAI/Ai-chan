# Studio shared assets

Cross-studio frontend primitives. Mounted by every studio backend at **`/shared`**.

## Layout

| Path | Role |
|------|------|
| `css/tokens.css` | Design tokens (`:root` CSS variables) |
| `css/base.css` | Shared chrome: reset, header, buttons, sidebar |
| `graph-bootstrap.js` | D3 helpers (`GraphBoot`) for graph studios |

## Page entry scripts (by function)

| Studio | Entry JS |
|--------|----------|
| DAG | `dag-graph.js` |
| Spec | `spec-graph.js` |
| LTM | `ltm-graph.js` |
| STM | `stm-rack.js` (+ `valence.js`) |
| KB | `kb-graph.js` |
| MCP | `mcp-servers.js` |
| Approval | `approval-drafts.js` |

## Usage

```html
<link rel="stylesheet" href="shared/css/tokens.css">
<link rel="stylesheet" href="shared/css/base.css">
<link rel="stylesheet" href="static/style.css">  <!-- page-only overrides -->

<script src="shared/graph-bootstrap.js"></script>
<script src="static/<studio>-....js"></script>
```

Keep page-specific rules (DAG edges, STM rack, LTM force nodes, Spec drawer) in each studio’s own `style.css`.
