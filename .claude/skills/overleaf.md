# Overleaf sync — agent instructions

Manuscript directories in this repo are connected to Overleaf projects over Overleaf's
git integration. Develop here, review on Overleaf.

Tool: `scripts/overleaf_sync.sh`. Registry: `<docs-dir>/overleaf-projects.tsv`.
Mirrors: gitignored `.overleaf/<name>/`.

---

## Start here

**Run `scripts/overleaf_sync.sh doctor` before doing anything else.** It checks the
environment, the credential, and every registered project, and it detects the two states
that otherwise look like something else: a wrong subpath, and a missing source root.

Then pick your situation:

| You want to | Go to |
|---|---|
| Connect a project for the first time | §2 |
| Push local edits up | §3 |
| Bring collaborator edits down | §4 |
| Reconcile two sides that disagree | §5 |
| Something is broken | `TROUBLESHOOTING.md` |

---

## 1. The one rule

**`push` mirrors with `--delete`. Anything on Overleaf and absent locally is destroyed.**

Always `compare` first. The script prompts before deleting, but by then you should
already know what the answer is.

---

## 2. Connecting a project

```bash
scripts/overleaf_sync.sh init <name>     # registry line must exist first
ls .overleaf/<name>/                     # what Overleaf has at its root
ls <docs-dir>/<name>/                    # what you have
```

**Now determine the subpath.** If Overleaf's root listing matches a directory *inside*
your manuscript folder rather than the folder itself, that inner directory is the
subpath, and it goes in the registry's third column.

Getting this wrong is the most common failure and the most damaging: with the wrong
subpath, `compare` reports the entire Overleaf project as deletable. In this project one
manuscript showed **266 phantom deletions** until the subpath was set to
`data_factory_source`.

Then reconcile (§5) before pushing. Never push a freshly connected project blind.

---

## 3. Pushing

```bash
scripts/overleaf_sync.sh compare <name>          # expect 0 only-Overleaf
scripts/overleaf_sync.sh push <name> -m "..."
```

If `compare` shows anything under `ONLY ON OVERLEAF`, stop and go to §5. Those files
will be deleted.

`push` fetches first and refuses when Overleaf has commits the mirror lacks, so you
cannot silently overwrite someone editing in the browser.

---

## 4. Pulling

```bash
git status                                  # pull refuses on a dirty worktree
scripts/overleaf_sync.sh pull <name>
git diff -- <docs-dir>/<name>               # review before committing
```

The dirty-worktree refusal is deliberate: a pull overwrites tracked files and git is the
only way back.

---

## 5. Reconciling a divergence

### Do not guess direction from file sizes

Both of these were true in this repo on the same day:

- one project's Overleaf copy was **newer** but had **shorter** chapters, because a
  restructure had moved content into new consolidated files
- another project's Overleaf copy was a **stale stub**: 32 lines against 199 locally

A blanket pull would have destroyed ~340 lines of written text. A blanket push would
have destroyed 31 files. Decide per file.

### Procedure

1. `git add -A && git commit -m "wip: pre-reconcile"` — git is the undo. If a file has
   uncommitted edits you cannot commit, copy the tree outside the repo first.
2. `scripts/overleaf_sync.sh compare <name>` — read all three sets.
3. State a rule and apply it uniformly. The rule used here was: *take the remote text by
   default; keep local only where it holds something the remote genuinely lacks
   (measured results, new data, sections written since the last upload); when unsure,
   prefer remote and re-add the local part deliberately.*
4. Before accepting any deletion, check four things:
   - **renamed, not gone?** `grep -rn "<a distinctive number>" .overleaf/<name>/`
   - **does the remote document its own removals?** A changelog naming what was merged
     where is authoritative. Read it before overruling it.
   - **is it a build dependency?** A `.bst` or `.sty` can look like dead weight and be
     load-bearing. `grep -rn "<stem>" .overleaf/<name>/ --include=*.tex`
   - **do the numbers survive?** Check each measured value appears somewhere remote.
5. Copy in the pure additions first (only-on-Overleaf files lose nothing), then apply
   overwrites, then re-run `compare`. Only your intended differences should remain.

### When the divergence is large

Reconciling dozens of files by eye is where mistakes happen. Classify each file
mechanically, then verify the classifications adversarially — try to *refute* every
"keep local" claim rather than confirm it, and default to remote when a claim cannot be
backed by a concrete value. That pass is what caught the stale-stub case here.

---

## 6. The registry

TAB-separated: `name <TAB> project-id <TAB> [subpath]`.

- **project id** — last segment of the Overleaf URL. Not a credential; safe to commit.
- **subpath** — the directory mapping to the Overleaf project **root**. `.` when the
  manuscript directory *is* the root.

Manuscript directories are discovered automatically, so adding one later is just a
registry line plus `init`.

---

## 7. Excludes

Built in: `.git/`, `*.zip`, `*.pptx`, `*.docx`, `outputs/`, `__pycache__/`, `.DS_Store`
and every latexmk product. Per-manuscript additions go in
`<source-root>/.overleafignore`, rsync syntax.

Excludes apply to **both** sides, so an excluded path that exists on Overleaf is left
alone rather than deleted. That is what lets Overleaf keep its own compiled `outputs/`.

`compare`/`status` use `rsync --checksum`, so a reported difference is real content, not
mtime skew from a fresh clone.

---

## 8. Auth

A git credential helper is configured for `git.overleaf.com`. The script contains no
credential logic and needs none.

- **Never** put a token in the registry, a remote URL, a command-line argument, a log or
  a commit. Arguments are visible in shell history and the process table.
- Enter it once at the `Password:` prompt of
  `git ls-remote https://git.overleaf.com/<PROJECT-ID>`.
- Overleaf tokens are **account-scoped**, not per project. One token reaches every
  project on the account; treat a leak as an account compromise and rotate at
  Overleaf → Account Settings → Git Integration.

---

## 9. Things that will confuse you

- **Overleaf's default branch is `main`** on current projects; older ones may be
  `master`. The script reads the mirror's actual `HEAD`, so both work.
- **Overleaf history is usually one squashed `"Update on Overleaf."` commit.** There is
  no useful upstream log to read.
- **A repo pre-commit hook that trims whitespace** will create one-byte diffs against
  Overleaf after every pull. Harmless; they clear on the next push.
- **The mirror is disposable.** If it gets into a bad state, delete `.overleaf/<name>/`
  and `init` again. No repo content lives there.
- **A wrong subpath fails loudly.** If `compare` ever reports a suspiciously clean
  `0 / 0 / 0`, check that the source root exists.
- **Your shell's working directory may persist between commands.** Use absolute paths.

---

## 10. Never

- push without comparing
- accept a phantom-deletion report as a real divergence — fix the subpath instead
- put a credential anywhere in the repo or on a command line
- `pull --force` over uncommitted work without a copy outside the repo

---

## See also

- `RUNBOOK.md` — the full ordered procedure with worked examples
- `TROUBLESHOOTING.md` — symptom → cause → fix
- `scripts/overleaf_sync.sh doctor` — automated diagnosis
