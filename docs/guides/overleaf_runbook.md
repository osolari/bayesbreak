# Overleaf sync: the runbook

An ordered procedure. Follow it top to bottom. Every step says what to run, what a good
result looks like, and what to do when it isn't.

If you are an agent picking this up cold: **run `scripts/overleaf_sync.sh doctor` first.**
It tells you which of the situations below you are in.

---

## The one rule

`push` mirrors with `--delete`. Anything on Overleaf that is absent locally is **destroyed**.

So: `compare` before every `push`, without exception. Everything else in this document is
elaboration on that.

---

## Situation A — connecting a project for the first time

You have an Overleaf project and a local directory, and neither has ever been synced.

### A1. Get the project id

From the Overleaf URL. `https://www.overleaf.com/project/6a973ad7432b6c56f6942855` →
the id is `6a973ad7432b6c56f6942855`. It is not secret; it is useless without an
authenticated account.

### A2. Register it with a guessed subpath

```
# name          project id                subpath
VSMR            6a973ad7432b6c56f6942855  .
```

Guess `.` to start. Step A4 tells you whether you were right.

### A3. Clone the mirror

```bash
scripts/overleaf_sync.sh init VSMR
```

Fails here mean one of: wrong project id, no credential, or no git-integration access on
the account. `doctor` distinguishes them.

### A4. Find the real subpath — do this before anything else

This is the step people skip, and skipping it is the single most common way to lose
work. Compare what Overleaf has at *its* root against your tree:

```bash
ls .overleaf/VSMR/                 # Overleaf's top level
ls <docs-dir>/VSMR/          # yours
```

If Overleaf's root is `book/ paper/ shared/` and your directory contains a single folder
that in turn contains `book/ paper/ shared/`, then **that folder is the subpath**.

Worked example from this repo: the data factory's Overleaf root was `book/ configs/
specs/ shared/`, but locally those lived under `mt_data_factory/data_factory_source/`.
With subpath `.`, `compare` reported **266 phantom deletions**. With subpath
`data_factory_source`, it reported 3 real differences.

**The signature of a wrong subpath:** `compare` shows a large only-local count *and* a
large only-Overleaf count with almost nothing in "content differs". The two trees are
not the same tree. `doctor` detects this pattern and says so.

### A5. Reconcile before you push

Go to Situation B. Do not push yet.

---

## Situation B — the two sides have diverged

Both the repo and Overleaf have content, and they disagree. This is the case that needs
care.

### B1. Take the inventory

```bash
scripts/overleaf_sync.sh compare VSMR
```

Three sets come back:

| Set | Meaning | If you push now |
|---|---|---|
| `ONLY ON OVERLEAF` | exists there, not here | **destroyed** |
| `ONLY LOCAL` | exists here, not there | added there |
| `CONTENT DIFFERS` | both have it, contents disagree | overwritten there |

### B2. Do not infer direction from file sizes

It is wrong about as often as it is right. Both of these happened in this repo on the
same day:

- `VSMR`: Overleaf was **newer** and had 31 files the repo lacked — but its chapters were
  *shorter*, because a restructure had moved content into new consolidated files.
- `ladder`: Overleaf's `paper/sections/03_method.tex` was **32 lines against 199
  locally**. Overleaf was a stale stub. A "pull because Overleaf looks newer" would have
  destroyed ~340 lines of written text.

### B3. Commit first

```bash
git add -A && git commit -m "wip: pre-Overleaf reconcile"
```

`pull` refuses to run against a dirty worktree for exactly this reason: git is your undo.
If any affected file has uncommitted edits that you cannot commit, copy the tree
somewhere outside the repo first — git cannot restore what it never saw.

### B4. Decide per file, with a stated rule

Write the rule down before you start, and apply it uniformly. The rule used here was:

> Take the remote text by default. Keep local content only where it holds something the
> remote genuinely lacks — measured results, new data, a section written since the last
> upload. When unsure, prefer the remote and re-add the local part deliberately.

Before accepting any deletion, check four things:

1. **Is the content gone, or renamed?** grep the whole mirror for a distinctive number or
   caption from the local file. Restructures rename and consolidate constantly.
   ```bash
   grep -rn "2m42s" .overleaf/VSMR/
   ```
2. **Does the remote document its own removals?** A well-maintained project often ships a
   changelog saying what was merged where. That is authoritative — read it before
   overruling it. The VSMR project had `CONTENT_PRESERVATION_AND_REMOVAL_LOG.md` naming
   every removed file and where its content went.
3. **Is it a build dependency rather than prose?** A `.bst`, `.sty` or `\input`-ed
   fragment can be referenced by nothing obvious and still be load-bearing. Check what
   actually references it:
   ```bash
   grep -rn "iclr2026_conference" .overleaf/VSMR/ --include=*.tex
   ```
4. **Do the numbers survive?** For each file you are about to drop, check its measured
   values appear somewhere in the remote tree.

### B5. Apply, then verify

Pure additions first — files that exist only on Overleaf can be copied in without losing
anything. Then the overwrites. Then re-run `compare`: the only remaining differences
should be the ones you intended.

---

## Situation C — routine day-to-day

```bash
# you edited locally
scripts/overleaf_sync.sh compare VSMR      # expect: 0 / 0 / n
scripts/overleaf_sync.sh push VSMR -m "revise section 3"

# a collaborator edited on Overleaf
scripts/overleaf_sync.sh pull VSMR
git diff -- <docs-dir>/VSMR          # review before committing
```

`push` fetches first and refuses if Overleaf has commits the mirror lacks, so you cannot
silently clobber someone working in the editor. It tells you to pull.

---

## Situation D — a brand-new project with nothing local

```bash
# register, then:
scripts/overleaf_sync.sh init my-paper     # creates the local dir and clones the mirror
scripts/overleaf_sync.sh pull my-paper     # fills it
```

`init` creates the local tree when it is absent, so this works from nothing.

---

## Adding a manuscript later

The tool discovers directories automatically. Adding one is: create the Overleaf project,
add a registry line, `init`, check the subpath with `compare`, reconcile, push.

---

## What never to do

- **Never push without comparing.** It deletes.
- **Never put a token in the registry, a remote URL, a command-line argument, or a
  commit.** Arguments are visible in the process table and in shell history. Auth is the
  credential helper's job; the script contains no credential logic at all.
- **Never "fix" a phantom-deletion report by accepting the deletions.** It is a wrong
  subpath, not a real divergence.
- **Never pull with uncommitted changes using `--force`** unless you have a copy outside
  the repo. The flag exists for the case where you genuinely want to discard local work.

---

## Reference

- `./troubleshooting.md` — symptom → cause → fix
- `.claude/skills/overleaf.md` — the agent-facing summary
- `scripts/overleaf_sync.sh doctor` — automated diagnosis
