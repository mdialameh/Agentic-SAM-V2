# README Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display the repository's demo GIF automatically near the top of the GitHub README.

**Architecture:** Track the existing GIF in the repository and reference it from `README.md` with a relative Markdown image path. GitHub's README renderer will load and animate the GIF without scripts or external hosting.

**Tech Stack:** GitHub-flavored Markdown, GIF, Git

## Global Constraints

- Place the Demo section after the research disclaimer and before Architecture.
- Use the repository-relative path `demo/demo-.gif`.
- Use the alt text `Agentic-SAM v2 demo`.
- Leave all pre-existing uncommitted changes outside `README.md` untouched.

---

### Task 1: Embed and Publish the Demo

**Files:**
- Modify: `README.md`
- Create: `demo/demo-.gif` (already supplied in the working tree)

**Interfaces:**
- Consumes: GitHub's repository-relative Markdown image resolution.
- Produces: A README `Demo` section that renders `demo/demo-.gif`.

- [ ] **Step 1: Add the Markdown embed**

Insert this block after the introductory disclaimer and before
`## Architecture (one process, no micro-services)`:

```markdown
## Demo

![Agentic-SAM v2 demo](demo/demo-.gif)
```

- [ ] **Step 2: Verify the relative asset path**

Run:

```bash
test -f demo/demo-.gif
rg -n '^## Demo$|^!\[Agentic-SAM v2 demo\]\(demo/demo-\.gif\)$' README.md
```

Expected: `test` exits successfully and `rg` prints both inserted lines.

- [ ] **Step 3: Review the staged scope**

Run:

```bash
git add README.md demo/demo-.gif
git diff --cached --name-status
```

Expected: the staged paths are exactly `README.md` and `demo/demo-.gif`.

- [ ] **Step 4: Commit the README demo**

Run:

```bash
git commit -m "Add animated demo to README"
```

Expected: Git creates a commit containing the README and GIF.

- [ ] **Step 5: Push and verify**

Load `GITHUB_TOKEN` from `.env` into the command environment, push `main`,
and compare `git rev-parse HEAD` with `git ls-remote origin refs/heads/main`.
Expected: both full commit SHAs match; `.env` remains ignored and no local
credential helper is configured.
