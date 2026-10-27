# Overleaf sync: troubleshooting

Every entry here is a failure that actually happened while building and running this
system, not a hypothetical. Start with `scripts/overleaf_sync.sh doctor`.

---

## `compare` reports hundreds of deletions on a project you just connected

**Cause.** The subpath is wrong. The Overleaf project root maps to a directory *inside*
your manuscript folder, not to the folder itself.

**Recognise it.** A large only-local count *and* a large only-Overleaf count, with almost
nothing in "content differs". The two trees are not the same tree.

**Fix.**
```bash
ls .overleaf/<name>/          # Overleaf's root
```
Find the local directory whose contents match that listing, and put its path in the third
column of the registry.

**Real case.** `mt_data_factory` with subpath `.` reported 266 deletions. With subpath
`data_factory_source` it reported 3 real differences.

---

## `compare` says everything is clean but nothing is actually working

**Cause.** The configured source root does not exist, so rsync fails and its error goes
to stderr while the counts come out zero.

**This used to be silent.** It is now a hard error — `source_root` dies with the offending
path. If you are on an older copy of the script, that is your bug.

**Real case.** A directory was moved up one level, the registry still pointed at the old
path, and `compare template` printed a clean `0 / 0 / 0` while doing nothing at all.

---

## `push` wants to delete files you have never heard of

**Cause.** Either the subpath is wrong (see above), or Overleaf genuinely has content
your repo does not.

**Fix.** Do not confirm the prompt. Run `compare`, read the only-Overleaf list, and work
through Situation B of the runbook. Those files may be the newer version.

---

## `pull` refuses: "has uncommitted changes"

**Working as intended.** A pull overwrites tracked files, and git is the only way to get
them back.

**Fix.** Commit or stash. Use `--force` only when you intend to discard the local edits,
and only if anything irreplaceable has been copied outside the repo first.

---

## Files differ by one byte after every pull

**Cause.** A repo pre-commit hook that trims trailing whitespace or fixes end-of-file
newlines. The hook rewrites the file after it lands, so it no longer matches Overleaf.

**Fix.** Harmless. Push once and it clears. Verify it really is whitespace:
```bash
diff <(sed -e 's/[[:space:]]*$//' .overleaf/<name>/<file>) \
     <(sed -e 's/[[:space:]]*$//' <source-root>/<file>)
```
Empty output means whitespace only.

---

## `git clone` fails or asks for a password repeatedly

**Cause.** No credential helper, or a revoked token.

**Fix.**
```bash
git config --global credential.https://git.overleaf.com.helper store
git config --global credential.https://git.overleaf.com.username git
git ls-remote https://git.overleaf.com/<PROJECT-ID>   # paste the token at the prompt
```
Enter it at the prompt — never on a command line, where it lands in shell history and the
process table.

**Note.** Overleaf git tokens are **account-scoped**, not per project. One token reaches
every project on the account, so a leak is an account-level problem.

---

## The script exits silently with status 1 and prints nothing

**Cause.** `set -euo pipefail` plus a bash gotcha: a function whose **last** statement is
a false `[[ ... ]] && cmd` returns 1, and that trips `set -e` at the call site.

**Fix.** End such functions with an explicit `return 0`. This bit the exclude-list builder
during development and produced exactly this symptom — no output, exit 1.

---

## A `cd` in one command affects the next one

**Cause.** The shell working directory persists between tool calls in some agent
harnesses.

**Fix.** Use absolute paths, or `cd` to the repo root at the start of every command. A
backup once got written into a mirror directory this way.

---

## `push` refuses: Overleaf has commits the mirror lacks

**Working as intended.** Someone edited in the Overleaf editor.

**Fix.** `pull`, review with `git diff`, then push.

---

## An operator or path in the docs does not exist

Check against the live system rather than the manuscript. Chapter listings are
illustrative and drift from the normative schemas. In this project the chapter's operator
listing used field names (`exact_tool_name`, `candidate_evidence_reads`) that **fail** the
normative schema, which uses `name` and `candidate_reads`.

---

## Quick reference

| Symptom | Likely cause | First command |
|---|---|---|
| hundreds of deletions | wrong subpath | `ls .overleaf/<name>/` |
| clean but broken | source root missing | `doctor` |
| one-byte diffs | pre-commit hook | push once |
| refuses to pull | dirty worktree | `git status` |
| silent exit 1 | `set -e` on a false `&&` | `bash -x` the script |
| password prompt loops | no credential helper | `doctor` |
