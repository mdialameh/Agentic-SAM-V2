# README Demo Design

## Goal

Show the existing `demo/demo-.gif` prominently in the GitHub README, where
GitHub will render and animate it automatically.

## Design

Add a `## Demo` section immediately after the introductory research disclaimer
and before the Architecture section. Embed the GIF with repository-relative
Markdown:

```markdown
![Agentic-SAM v2 demo](demo/demo-.gif)
```

The relative path keeps the image valid on GitHub without relying on a branch-
specific or external URL. The descriptive alt text provides a useful fallback
if the GIF cannot load.

## Scope

Commit the design record, the README edit, and `demo/demo-.gif`. Leave all other
pre-existing uncommitted changes untouched.

## Verification

- Confirm the Markdown image path resolves to the tracked GIF.
- Confirm the commit contains only the intended files.
- Push `main` and verify that the remote SHA matches the local SHA.
- Confirm `.env` remains ignored and no credential helper is persisted.
