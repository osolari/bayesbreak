#!/usr/bin/env bash
# Sync manuscript packages to/from their Overleaf git repos.
#
# Portable: set OVERLEAF_DOCS_DIR to the directory holding your manuscripts
# (default docs/manuscripts). No other per-repo change is needed.
#
# Model: mirror clone + rsync. Each registered manuscript gets a gitignored mirror
# clone under .overleaf/<name>/ that tracks the Overleaf project repo. Source is
# rsynced in/out with an exclude list, so build artifacts and large binaries never
# reach Overleaf and Overleaf history stays separate from LADDER history.
#
# Auth: relies on the git credential helper configured for git.overleaf.com.
# This script never reads, stores, logs or accepts a token.
#
# Usage:
#   scripts/overleaf_sync.sh list
#   scripts/overleaf_sync.sh doctor            diagnose a broken setup
#   scripts/overleaf_sync.sh init   <name>
#   scripts/overleaf_sync.sh status  <name|all>
#   scripts/overleaf_sync.sh compare <name|all>
#   scripts/overleaf_sync.sh push   <name|all> [-m "message"] [--dry-run] [--yes]
#   scripts/overleaf_sync.sh pull   <name|all> [--dry-run] [--force]

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# Root holding one subdirectory per manuscript. Override per repo:
#   export OVERLEAF_DOCS_DIR=papers
DOCS_SUBDIR="${OVERLEAF_DOCS_DIR:-docs/manuscripts}"
MANUSCRIPT_DIR="$REPO_ROOT/$DOCS_SUBDIR"
MIRROR_ROOT="$REPO_ROOT/.overleaf"
REGISTRY="$MANUSCRIPT_DIR/overleaf-projects.tsv"
OVERLEAF_HOST="https://git.overleaf.com"
NET_TIMEOUT="${OVERLEAF_NET_TIMEOUT:-180}"

# Never shipped to Overleaf: VCS metadata, original upload archives, binary
# deliverables Overleaf does not need, and latexmk build products.
DEFAULT_EXCLUDES=(
  ".git/" ".gitignore" ".overleafignore"
  "*.zip" "*.pptx" "*.docx"
  "outputs/"
  "__pycache__/" ".DS_Store"
  "*.aux" "*.log" "*.out" "*.fls" "*.fdb_latexmk" "*.synctex.gz"
  "*.toc" "*.lof" "*.lot" "*.bbl" "*.blg" "*.bcf" "*.run.xml"
  "*.nav" "*.snm" "*.vrb" "*.idx" "*.ilg" "*.ind"
)

die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '%s\n' "$*"; }

# ---------------------------------------------------------------- registry ---

# Every directory under the docs dir is a candidate manuscript.
discover() {
  find "$MANUSCRIPT_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
}

project_id_for() {
  local name="$1"
  [[ -f "$REGISTRY" ]] || return 1
  awk -v n="$name" '
    /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
    $1 == n { print $2; found = 1; exit }
    END { exit !found }
  ' "$REGISTRY"
}

# Optional 3rd column: the subdirectory that maps to the Overleaf project ROOT.
# Defaults to "." - needed when the Overleaf top level sits deeper in your repo,
# e.g. a package whose sources live under a src/ or <name>_source/ subdirectory.
subpath_for() {
  local name="$1" sub=""
  if [[ -f "$REGISTRY" ]]; then
    sub="$(awk -v n="$name" '
      /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
      $1 == n { if (NF >= 3) print $3; exit }
    ' "$REGISTRY")"
  fi
  printf '%s' "${sub:-.}"
}

# Absolute path of the tree that mirrors the Overleaf project root.
# Dies if it does not exist: a wrong subpath must fail loudly, never be reported
# as "no differences" by compare/status while rsync quietly errors.
source_root() {
  local name="$1" sub path
  sub="$(subpath_for "$name")"
  if [[ "$sub" == "." ]]; then
    path="$MANUSCRIPT_DIR/$name"
  else
    path="$MANUSCRIPT_DIR/$name/$sub"
  fi
  [[ -d "$path" ]] || die "source root for '$name' does not exist: ${path#$REPO_ROOT/}
  Fix the subpath column for '$name' in ${REGISTRY#$REPO_ROOT/}, or restore the directory."
  printf '%s' "$path"
}

registered_names() {
  [[ -f "$REGISTRY" ]] || return 0
  awk '!/^[[:space:]]*#/ && NF >= 2 { print $1 }' "$REGISTRY" | sort
}

require_project_id() {
  local name="$1" pid
  pid="$(project_id_for "$name")" || die \
    "'$name' is not registered. Add a line to ${REGISTRY#$REPO_ROOT/}:
    $name<TAB><overleaf-project-id>
  The project id is the last path segment of the Overleaf project URL."
  [[ "$pid" == "<fill-me>" ]] && die "'$name' has a placeholder project id in ${REGISTRY#$REPO_ROOT/}."
  printf '%s' "$pid"
}

# ----------------------------------------------------------------- excludes --

# Shared defaults plus an optional per-manuscript .overleafignore (rsync syntax).
build_exclude_args() {
  local name="$1"
  local -n _out="$2"
  _out=()
  local pat
  for pat in "${DEFAULT_EXCLUDES[@]}"; do _out+=(--exclude "$pat"); done
  local custom; custom="$(source_root "$name")/.overleafignore"
  if [[ -f "$custom" ]]; then _out+=(--exclude-from "$custom"); fi
  # Explicit: a trailing false test would return 1 and trip `set -e` at the call site.
  return 0
}

# ------------------------------------------------------------------ mirrors --

mirror_path() { printf '%s/%s' "$MIRROR_ROOT" "$1"; }

require_mirror() {
  local name="$1" mp
  mp="$(mirror_path "$name")"
  [[ -d "$mp/.git" ]] || die "no mirror for '$name'. Run: scripts/overleaf_sync.sh init $name"
  printf '%s' "$mp"
}

# Read the mirror's actual branch; Overleaf projects observed to use main.
# The literal is only a fallback for a detached HEAD.
mirror_branch() {
  git -C "$1" symbolic-ref --quiet --short HEAD 2>/dev/null || printf 'main'
}

# --------------------------------------------------------------- commands ----

cmd_list() {
  printf '%-22s %-14s %-12s %s\n' "MANUSCRIPT" "REGISTERED" "MIRROR" "PROJECT ID"
  local name pid mark mirror
  while read -r name; do
    [[ -z "$name" ]] && continue
    if pid="$(project_id_for "$name")"; then mark="yes"; else mark="NO"; pid="-"; fi
    if [[ -d "$(mirror_path "$name")/.git" ]]; then mirror="cloned"; else mirror="-"; fi
    printf '%-22s %-14s %-12s %s\n' "$name" "$mark" "$mirror" "$pid"
  done < <(discover)
  info ""
  info "Unregistered dirs are ignored by push/pull until you add them to ${REGISTRY#$REPO_ROOT/}."
}

cmd_init() {
  local name="$1" pid mp sub target
  pid="$(require_project_id "$name")"
  # Create the local tree if absent: init is also how you attach a brand-new
  # Overleaf project that has no repo-side copy yet, which `pull` then fills.
  sub="$(subpath_for "$name")"
  if [[ "$sub" == "." ]]; then target="$MANUSCRIPT_DIR/$name"; else target="$MANUSCRIPT_DIR/$name/$sub"; fi
  if [[ ! -d "$target" ]]; then
    mkdir -p "$target"
    info "created ${target#$REPO_ROOT/}"
  fi
  mp="$(mirror_path "$name")"
  if [[ -d "$mp/.git" ]]; then
    info "mirror already exists: ${mp#$REPO_ROOT/}"
    return 0
  fi
  mkdir -p "$MIRROR_ROOT"
  info "cloning Overleaf project into ${mp#$REPO_ROOT/} ..."
  timeout "$NET_TIMEOUT" git clone "$OVERLEAF_HOST/$pid" "$mp" \
    || die "clone failed for '$name'. Check the project id and that your Overleaf credential is valid."
  info "done. Next: scripts/overleaf_sync.sh status $name"
}

cmd_status() {
  local name="$1" mp branch src
  mp="$(require_mirror "$name")"
  branch="$(mirror_branch "$mp")"
  src="$(source_root "$name")"
  local -a ex; build_exclude_args "$name" ex

  info "=== $name ==="
  info "-- local -> Overleaf (files rsync would change) --"
  rsync -rinc --delete "${ex[@]}" "$src/" "$mp/" | sed 's/^/   /' || true

  info "-- Overleaf remote vs mirror --"
  timeout "$NET_TIMEOUT" git -C "$mp" fetch --quiet origin "$branch" 2>/dev/null \
    || { info "   (fetch failed - offline or credential issue)"; return 0; }
  local behind ahead
  behind="$(git -C "$mp" rev-list --count "HEAD..origin/$branch" 2>/dev/null || echo 0)"
  ahead="$(git -C "$mp" rev-list --count "origin/$branch..HEAD" 2>/dev/null || echo 0)"
  info "   mirror is $ahead ahead, $behind behind origin/$branch"
}

# Three-way inventory before any destructive sync. rsync's itemised dry-run
# already classifies exactly the sets we care about:
#   *deleting     -> exists only on Overleaf (a push would destroy it)
#   >f+++++++++   -> exists only locally    (a push would add it)
#   >fc........   -> exists on both, content differs (a real conflict)
cmd_compare() {
  local name="$1" mp src items
  mp="$(require_mirror "$name")"
  src="$(source_root "$name")"
  local -a ex; build_exclude_args "$name" ex
  items="$(rsync -rinc --delete "${ex[@]}" "$src/" "$mp/" || true)"

  local only_ol only_loc differ
  only_ol="$(printf  '%s\n' "$items" | awk '/^\*deleting/ { $1=""; sub(/^ +/,""); print }')"
  only_loc="$(printf '%s\n' "$items" | awk '/^>f\+\+\+/ { print $2 }')"
  differ="$(printf   '%s\n' "$items" | awk '/^>f/ && !/^>f\+\+\+/ && $1 ~ /c/ { print $2 }')"

  local n_ol n_loc n_diff
  n_ol="$(printf   '%s' "$only_ol"  | grep -c . || true)"
  n_loc="$(printf  '%s' "$only_loc" | grep -c . || true)"
  n_diff="$(printf '%s' "$differ"   | grep -c . || true)"

  info "=== $name ==="
  info "  ONLY ON OVERLEAF ($n_ol) - pull these in, or a push destroys them:"
  printf '%s\n' "$only_ol" | grep . | sed 's/^/     /' || info "     (none)"
  info "  ONLY LOCAL ($n_loc) - a push adds these to Overleaf:"
  printf '%s\n' "$only_loc" | grep . | sed 's/^/     /' || info "     (none)"
  info "  CONTENT DIFFERS ($n_diff) - needs a per-file decision:"
  printf '%s\n' "$differ" | grep . | sed 's/^/     /' || info "     (none)"
  return 0
}

cmd_push() {
  local name="$1" msg="$2" dry="$3" assume_yes="$4"
  local mp branch src
  mp="$(require_mirror "$name")"
  branch="$(mirror_branch "$mp")"
  src="$(source_root "$name")"
  local -a ex; build_exclude_args "$name" ex

  # Land Overleaf-side commits first so we never push a non-fast-forward.
  timeout "$NET_TIMEOUT" git -C "$mp" fetch --quiet origin "$branch" \
    || die "fetch failed for '$name'"
  if [[ "$(git -C "$mp" rev-list --count "HEAD..origin/$branch")" != "0" ]]; then
    die "'$name' has Overleaf-side commits not in the mirror.
  Run: scripts/overleaf_sync.sh pull $name   (review, then push again)"
  fi

  # rsync --delete removes files on the Overleaf side; surface that before doing it.
  local deletions
  deletions="$(rsync -rinc --delete "${ex[@]}" "$src/" "$mp/" | grep '^\*deleting' || true)"
  if [[ -n "$deletions" ]]; then
    info "the following files will be DELETED from the Overleaf project:"
    printf '%s\n' "$deletions" | sed 's/^/   /'
    if [[ "$dry" != "yes" && "$assume_yes" != "yes" ]]; then
      read -r -p "proceed? [y/N] " reply
      [[ "$reply" == "y" || "$reply" == "Y" ]] || die "aborted"
    fi
  fi

  if [[ "$dry" == "yes" ]]; then
    info "-- dry run: local -> Overleaf --"
    rsync -rinc --delete "${ex[@]}" "$src/" "$mp/" | sed 's/^/   /' || true
    return 0
  fi

  rsync -ric --delete "${ex[@]}" "$src/" "$mp/" >/dev/null
  git -C "$mp" add -A
  if git -C "$mp" diff --cached --quiet; then
    info "$name: nothing to push"
    return 0
  fi
  git -C "$mp" commit --quiet -m "$msg"
  timeout "$NET_TIMEOUT" git -C "$mp" push --quiet origin "$branch" \
    || die "push failed for '$name'"
  info "$name: pushed to Overleaf ($branch)"
}

cmd_pull() {
  local name="$1" dry="$2" force="$3"
  local mp branch src
  mp="$(require_mirror "$name")"
  branch="$(mirror_branch "$mp")"
  src="$(source_root "$name")"
  local -a ex; build_exclude_args "$name" ex

  # Pulling overwrites tracked source. Require a clean worktree so git is the undo.
  if [[ "$force" != "yes" ]] && ! git -C "$REPO_ROOT" diff --quiet -- "$DOCS_SUBDIR/$name"; then
    die "$DOCS_SUBDIR/$name has uncommitted changes.
  Commit or stash them first (git is your rollback for an overwriting pull), or pass --force."
  fi

  timeout "$NET_TIMEOUT" git -C "$mp" fetch --quiet origin "$branch" || die "fetch failed for '$name'"
  git -C "$mp" reset --quiet --hard "origin/$branch"

  local rsync_mode=(-ric)
  if [[ "$dry" == "yes" ]]; then rsync_mode=(-rinc); fi
  info "-- Overleaf -> local ($name) --"
  # No --delete here: Overleaf never sees the excluded files, so a delete pass
  # would wipe outputs/ and the archives it simply does not know about.
  rsync "${rsync_mode[@]}" "${ex[@]}" "$mp/" "$src/" | sed 's/^/   /' || true
  if [[ "$dry" == "yes" ]]; then return 0; fi
  info "$name: pulled. Review with: git -C $REPO_ROOT diff -- $DOCS_SUBDIR/$name"
}


# ------------------------------------------------------------------ doctor ----

# One command that answers "why isn't this working". Written because the failure
# modes look alike from the outside: a wrong subpath, a missing mirror and a bad
# credential all present as "compare says something weird".
cmd_doctor() {
  local problems=0 warnings=0
  info "=== environment ==="
  for tool in git rsync awk; do
    if command -v "$tool" >/dev/null 2>&1; then
      info "  ok       $tool"
    else
      info "  MISSING  $tool"; problems=$((problems+1))
    fi
  done

  local helper
  helper="$(git config --get credential.https://git.overleaf.com.helper || true)"
  if [[ -n "$helper" ]]; then
    info "  ok       credential helper for git.overleaf.com ($helper)"
  else
    info "  MISSING  no credential helper for git.overleaf.com"
    info "           git config --global credential.https://git.overleaf.com.helper store"
    info "           git config --global credential.https://git.overleaf.com.username git"
    problems=$((problems+1))
  fi

  if [[ -f "$REPO_ROOT/.pre-commit-config.yaml" ]]; then
    info "  note     this repo runs pre-commit; hooks that trim whitespace or fix"
    info "           end-of-file newlines will create one-byte diffs against Overleaf"
    info "           after every pull. Harmless, and they clear on the next push."
  fi

  info ""
  info "=== registry ==="
  if [[ ! -f "$REGISTRY" ]]; then
    info "  MISSING  ${REGISTRY#$REPO_ROOT/}"; problems=$((problems+1))
    return 1
  fi
  info "  ok       ${REGISTRY#$REPO_ROOT/}"

  local name pid sub src mp
  while read -r name; do
    [[ -z "$name" ]] && continue
    info ""
    info "=== $name ==="

    if ! pid="$(project_id_for "$name")"; then
      info "  skip     not registered"
      continue
    fi
    if [[ "$pid" == "<fill-me>" ]]; then
      info "  TODO     placeholder project id; put the real one in the registry"
      warnings=$((warnings+1)); continue
    fi
    info "  ok       project id $pid"

    sub="$(subpath_for "$name")"
    if [[ "$sub" == "." ]]; then src="$MANUSCRIPT_DIR/$name"; else src="$MANUSCRIPT_DIR/$name/$sub"; fi
    if [[ -d "$src" ]]; then
      info "  ok       source root ${src#$REPO_ROOT/} (subpath: $sub)"
    else
      info "  BROKEN   source root does not exist: ${src#$REPO_ROOT/}"
      info "           the subpath column is wrong, or the directory moved."
      info "           Compare the Overleaf project's top level with your tree:"
      info "             ls .overleaf/$name/            # what Overleaf has at its root"
      info "             find docs -type d -name '<one of those>'"
      problems=$((problems+1)); continue
    fi

    mp="$(mirror_path "$name")"
    if [[ ! -d "$mp/.git" ]]; then
      info "  TODO     no mirror yet: scripts/overleaf_sync.sh init $name"
      warnings=$((warnings+1)); continue
    fi
    info "  ok       mirror cloned ($(git -C "$mp" symbolic-ref --quiet --short HEAD || echo detached))"

    if timeout "$NET_TIMEOUT" git -C "$mp" ls-remote --exit-code origin >/dev/null 2>&1; then
      info "  ok       remote reachable and credential accepted"
    else
      info "  BROKEN   cannot reach the remote, or the credential was rejected"
      info "           check the token at Overleaf -> Account Settings -> Git Integration"
      problems=$((problems+1)); continue
    fi

    local -a ex; build_exclude_args "$name" ex
    local items only_ol only_loc differ total_local
    items="$(rsync -rinc --delete "${ex[@]}" "$src/" "$mp/" 2>/dev/null || true)"
    only_ol="$(printf '%s\n' "$items" | grep -c '^\*deleting' || true)"
    only_loc="$(printf '%s\n' "$items" | grep -c '^>f+++' || true)"
    differ="$(printf '%s\n' "$items" | grep '^>f' | grep -vc '^>f+++' || true)"
    total_local="$(find "$src" -type f -not -path '*/.git/*' | wc -l | tr -d ' ')"
    info "  state    only-Overleaf=$only_ol only-local=$only_loc differ=$differ (local files: $total_local)"

    # The signature of a wrong subpath: almost nothing matches in either direction.
    if [[ "$only_loc" -ge 20 && "$only_ol" -ge 20 && "$differ" -eq 0 ]]; then
      info "  LIKELY   the subpath is wrong. Everything reads as only-local AND"
      info "           only-Overleaf with no overlap, which means the two trees are"
      info "           not the same tree. Fix the subpath before pushing: a push"
      info "           would delete $only_ol file(s) from Overleaf."
      problems=$((problems+1))
    elif [[ "$only_ol" -gt 0 && "$total_local" -le 2 ]]; then
      info "  ACTION   the local tree is essentially empty and Overleaf has $only_ol file(s)."
      info "           This is a fresh connection, not a divergence. Pull, do not push:"
      info "             scripts/overleaf_sync.sh pull $name"
      warnings=$((warnings+1))
    elif [[ "$only_ol" -gt 0 ]]; then
      info "  WARN     a push right now would DELETE $only_ol file(s) from Overleaf."
      info "           Both sides have content, so this is a real divergence."
      info "           Reconcile before pushing - see the runbook, situation B:"
      info "             scripts/overleaf_sync.sh compare $name"
      warnings=$((warnings+1))
    fi

    if ! git -C "$REPO_ROOT" diff --quiet -- "${src#$REPO_ROOT/}" 2>/dev/null; then
      info "  WARN     uncommitted local changes; pull will refuse until you commit"
      warnings=$((warnings+1))
    fi
  done < <(discover)

  info ""
  info "=== summary: $problems problem(s), $warnings warning(s) ==="
  [[ "$problems" -eq 0 ]]
}

# ------------------------------------------------------------------- main ----

main() {
  local cmd="${1:-}"; shift || true
  [[ -z "$cmd" ]] && { cmd_list; exit 0; }

  case "$cmd" in
    list) cmd_list; exit 0 ;;
    init|status|push|pull|compare) ;;
    doctor) cmd_doctor; exit $? ;;
    *) die "unknown command '$cmd' (list|doctor|init|status|compare|push|pull)" ;;
  esac

  local target="${1:-}"; shift || true
  [[ -z "$target" ]] && die "$cmd needs a manuscript name (or 'all')"

  local msg="Sync from LADDER repo" dry="no" force="no" assume_yes="no"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -m|--message) msg="${2:?-m needs a value}"; shift 2 ;;
      --dry-run)    dry="yes"; shift ;;
      --force)      force="yes"; shift ;;
      --yes|-y)     assume_yes="yes"; shift ;;
      *) die "unknown option '$1'" ;;
    esac
  done

  local -a targets
  if [[ "$target" == "all" ]]; then
    [[ "$cmd" == "init" ]] && die "'init all' is not supported; init one project at a time"
    mapfile -t targets < <(registered_names)
    [[ ${#targets[@]} -eq 0 ]] && die "no registered manuscripts in ${REGISTRY#$REPO_ROOT/}"
  else
    targets=("$target")
  fi

  local n
  for n in "${targets[@]}"; do
    case "$cmd" in
      init)   cmd_init   "$n" ;;
      status)  cmd_status  "$n" ;;
      compare) cmd_compare "$n" ;;
      push)   cmd_push   "$n" "$msg" "$dry" "$assume_yes" ;;
      pull)   cmd_pull   "$n" "$dry" "$force" ;;
    esac
  done
}

main "$@"
