#!/usr/bin/env python3
"""Apply the independent Phase 6 mathematical and statistical corrections.

The script is intentionally assertion-heavy: every replacement must match the
Phase 5 source exactly, so a source drift fails rather than silently editing the
wrong passage.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_all_checked(text: str, old: str, new: str, *, expected: int, label: str) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} matches, found {count}")
    return text.replace(old, new)


# ---------------------------------------------------------------------------
# 1. Formal setup: structural support and a descriptor-indexed reporting target
# ---------------------------------------------------------------------------
rel = "book/chapters/03-formal-setup.tex"
text = read(rel)
text = replace_once(
    text,
    r"$m(\theta)$ & Observation-scale mean parameter induced by $\theta$, i.e. $m(\theta)=\mathbb{E}[Y\mid\theta]$. \\",
    r"$m_\star(\theta)=m(\theta;d_\star)$ & Declared segment functional on the reporting scale, evaluated at a fixed reference descriptor $d_\star$ when the observation mean depends on exposure or trial structure. Examples include a Gaussian mean, a Poisson rate at unit exposure, and a Binomial success probability. \\",
    label="formal glossary target",
)
text = replace_once(
    text,
    r"$A^{(r)}_{ij}$ & Block moment numerator $A^{(0)}_{ij}\,\mathbb{E}[m(\theta)^r\mid y_{i+1:j}]$ (for integer $r\in\{1,2,\dots\}$; conjugate-case definition). In the non-conjugate Laplace discussion of Section~\ref{sec:nonconj}, the superscript is a test function rather than a moment order; those objects are denoted $M^{[g]}_{ij}$ to avoid collision. \\",
    r"$A^{(r)}_{ij}$ & Block moment numerator $A^{(0)}_{ij}\,\mathbb{E}[m_\star(\theta)^r\mid y_{i+1:j}]$ (for integer $r\in\{1,2,\dots\}$; conjugate-case definition). In the non-conjugate Laplace discussion of Section~\ref{sec:nonconj}, the superscript is a test function rather than a moment order; those objects are denoted $M^{[g]}_{ij}$ to avoid collision. \\",
    label="formal glossary moment",
)
old = r"""\begin{assumption}[Offline segmentation support and block independence]
\label{ass:standing-offline}
The observations are ordered on a fixed set of candidate boundary locations; the admissible
segment-count set $\{1,\dots,k_{\max}\}$ is finite; every admissible block has finite nonnegative
block evidence, with strictly positive evidence on the support used by the DP and score $0$ (or
log-score $-\infty$) on excluded blocks; segment parameters are conditionally independent across
blocks given a segmentation; and the conditional partition prior factorizes over segments as in
Assumption~\ref{ass:factorizing-prior}. For approximate non-conjugate blocks, the same statement
applies to the approximate score array, while comparison with the exact non-conjugate model
requires the additional error hypothesis of Assumption~\ref{ass:uniform-block-error}.
\end{assumption}"""
new = r"""\begin{assumption}[Offline segmentation support and block independence]
\label{ass:standing-offline}
The observations are ordered on a fixed set of candidate boundary locations and the segment-count
set $\{1,\dots,k_{\max}\}$ is finite. Structural admissibility of a candidate block is determined
before observing the response values by the candidate grid, declared segment-length restrictions,
and prior support. Every structurally admissible block has finite nonnegative block evidence; a zero
evidence contributes zero to the likelihood-weighted partition sum but does not change the prior
support or its normalizer $C_k$. Structurally excluded blocks have local prior factor zero and are
stored with log-score $-\infty$. Segment parameters are conditionally independent across blocks
given a segmentation, and the conditional partition prior factorizes over segments as in
Assumption~\ref{ass:factorizing-prior}. For approximate non-conjugate blocks, the same statement
applies to the supplied approximate score array, while comparison with the exact non-conjugate model
requires the additional error hypothesis of Assumption~\ref{ass:uniform-block-error}.
\end{assumption}"""
text = replace_once(text, old, new, label="standing support")
old = r"""\begin{definition}[Segmentation space and admissible support]
\label{def:segmentation-space}
For fixed $n$ and $k$, let
\[
\mathcal{T}_{k,n}:=\{t=(t_0,\dots,t_k):0=t_0<t_1<\cdots<t_k=n\}.
\]
The \emph{admissible support} $\mathcal{T}^{+}_{k,n}\subseteq\mathcal{T}_{k,n}$ consists of
those boundary vectors for which every block has positive prior cohesion and finite positive
block evidence. Equivalently, $t\in\mathcal{T}^{+}_{k,n}$ only when
$g(\Delta_x(t_{q-1},t_q))A^{(0)}_{t_{q-1},t_q}>0$ for all $q$, with the obvious replacement of
$A^{(0)}$ by an approximate evidence when a non-conjugate routine is being analyzed as an
approximate score model.
\end{definition}"""
new = r"""\begin{definition}[Segmentation space and structural support]
\label{def:segmentation-space}
For fixed $n$ and $k$, let
\[
\mathcal{T}_{k,n}:=\{t=(t_0,\dots,t_k):0=t_0<t_1<\cdots<t_k=n\}.
\]
The \emph{structural support} $\mathcal{T}^{+}_{k,n}\subseteq\mathcal{T}_{k,n}$ consists of
those boundary vectors for which every block satisfies the declared grid and length restrictions
and has positive local prior factor $\gamma_x(t_{q-1},t_q)$. Conditional on the design and model
specification, this set is fixed independently of the observed response values and of
$A^{(0)}_{ij}$. A structurally admissible partition may have zero likelihood for the observed data;
it then contributes zero to the posterior partition sum, while the prior normalizer $C_k$ continues
to sum over the full structural support.
\end{definition}"""
text = replace_once(text, old, new, label="structural support definition")
text = text.replace(r"$p(\theta_i\mid y)$ and $\mathbb{E}[m(\theta_i)\mid y]$ with $m(\theta)$ the", r"$p(\theta_i\mid y)$ and $\mathbb{E}[m_\star(\theta_i)\mid y]$ with $m_\star$ the")
write(rel, text)

# Apply the reporting-functional notation consistently in the book and paper.
for rel, expected in {
    "book/chapters/04-block-evidence.tex": 4,
    "book/chapters/05-sum-product.tex": 1,
    "book/chapters/10-nonconjugate.tex": 2,
    "book/chapters/11-prediction-metrics.tex": 1,
    "paper/sections/03-model-block-evidence.tex": 4,
    "paper/sections/04-exact-inference.tex": 1,
}.items():
    text = read(rel)
    text = replace_all_checked(text, r"m(\theta)", r"m_\star(\theta)", expected=expected, label=f"m-star {rel}")
    write(rel, text)

# Clarify the definition of the reporting target in the main paper.
rel = "paper/sections/03-model-block-evidence.tex"
text = read(rel)
text = replace_once(
    text,
    r"For an observation-scale target $m_\star(\theta)$, define",
    r"Fix a scientifically declared reference descriptor $d_\star$ when the conditional observation mean depends on exposure or trial structure, and write $m_\star(\theta)=m(\theta;d_\star)$ for the segment functional reported by the analysis (for example, a Poisson rate at unit exposure rather than an exposure-specific count mean). Define",
    label="paper target definition",
)
old = r"""\begin{assumption}[Offline support and block independence]
\label{ass:offline}
The candidate segment-count set $\{1,\ldots,k_{\max}\}$ is finite. Every admissible block has a finite positive local evidence, while excluded blocks have score zero or log-score $-\infty$. Conditional on a partition, block parameters are independent across blocks. The conditional partition prior factorizes through local terms as in Assumption~\ref{ass:factorized-prior}. For approximate block routines, the same statements define an exact posterior for the supplied approximate score array; comparison to the intended nonconjugate model additionally requires Assumption~\ref{ass:block-error}.
\end{assumption}"""
new = r"""\begin{assumption}[Offline structural support and block independence]
\label{ass:offline}
The candidate segment-count set $\{1,\ldots,k_{\max}\}$ is finite. Structural admissibility of a segment is fixed by the candidate grid, declared length restrictions, and prior support before the response values are observed. A structurally admissible segment has finite nonnegative local evidence; zero evidence removes its likelihood contribution but does not alter the partition-prior normalizer. Structurally excluded segments have local prior factor zero and log-score $-\infty$. Conditional on a partition, segment parameters are independent across segments. The conditional partition prior factorizes through local terms as in Assumption~\ref{ass:factorized-prior}. For approximate segment routines, the same statements define an exact posterior for the supplied approximate score array; comparison to the intended nonconjugate model additionally requires Assumption~\ref{ass:block-error}.
\end{assumption}"""
text = replace_once(text, old, new, label="paper structural support")
write(rel, text)

# Clarify the nonconjugate reporting functional definition after notation replacement.
rel = "book/chapters/10-nonconjugate.tex"
text = read(rel)
text = replace_once(
    text,
    r"with $m_\star(\theta):=\mathbb{E}[Y\mid\theta]$. We describe four complementary approaches:",
    r"where $m_\star(\theta)=m(\theta;d_\star)$ is the declared reporting functional at a fixed reference descriptor. We describe four complementary approaches:",
    label="nonconjugate target definition",
)
write(rel, text)

# ---------------------------------------------------------------------------
# 2. Beta-observation quadrature: retain the integral, remove unsupported rate
# ---------------------------------------------------------------------------
rel = "book/chapters/04-block-evidence.tex"
text = read(rel)
old = r"""\begin{proposition}[Beta-block evidence via 1-D quadrature]
\label{prop:beta-block}
Assume $y_t\in(0,1)$ with latent mean $\mu$ constant within a block, known precisions
$\phi_t>0$, and prior $\mu\sim\mathrm{Beta}(a_0,b_0)$. Define the integrand
\[
\begin{aligned}
\ell_{ij}(\mu)
&:= \sum_{t=i+1}^j
   \log\mathrm{BetaPDF}\!\left(y_t\mid\phi_t\mu,\phi_t(1-\mu)\right),\\
\log\pi(\mu)
&:= \log\mathrm{BetaPDF}(\mu\mid a_0,b_0).
\end{aligned}
\]
The block evidence and moment numerators admit deterministic one-dimensional quadrature
representations:
\[
A^{(r)}_{ij}
= \int_0^1 \mu^r\,\exp\!\big\{\ell_{ij}(\mu)+\log\pi(\mu)\big\}\,d\mu,
\qquad r\in\{0,1,2,\dots\},
\]
and any Gauss--Legendre rule with $Q$ nodes returns an approximation
\[
\widehat A^{(r)}_{ij}
= \sum_{q=1}^Q w_q^{\mathrm{GL}}\,\mu_q^r\,\exp\{\ell_{ij}(\mu_q)+\log\pi(\mu_q)\},
\]
whose log-evidence error is $\varepsilon_Q\downarrow 0$ as $Q\to\infty$ at the rate $O(Q^{-2s})$
for integrands with $2s$ continuous derivatives, using $s$ for smoothness rather than the moment
order $r$.
\end{proposition}
\begin{proof}
The unknown block parameter is one-dimensional and supported on $(0,1)$. Multiplication of the
Beta-observation likelihood by the Beta prior gives the displayed nonnegative integrand. Its
integral is exactly the block evidence; multiplication by $\mu^r$ gives the moment numerator.
The finite-node expression is the chosen quadrature approximation, not an exact conjugate formula.
\end{proof}

\begin{remark}[Quadrature stability]
\label{rem:beta-quadrature-stability}
The block log-posterior $\ell_{ij}(\mu)+\log\pi(\mu)$ is not in general log-concave in $\mu$;
implementations check quadrature stability by increasing the node count $Q$ and controlling
behavior near the endpoints of $(0,1)$.
\end{remark}"""
new = r"""\begin{proposition}[Beta-block evidence as a one-dimensional integral]
\label{prop:beta-block}
Assume $y_t\in(0,1)$ with latent mean $\mu$ constant within a block, known precisions
$\phi_t>0$, and prior $\mu\sim\mathrm{Beta}(a_0,b_0)$. Define
\[
\begin{aligned}
\ell_{ij}(\mu)
&:= \sum_{t=i+1}^j
   \log\mathrm{BetaPDF}\!\left(y_t\mid\phi_t\mu,\phi_t(1-\mu)\right),\\
\log\pi(\mu)
&:= \log\mathrm{BetaPDF}(\mu\mid a_0,b_0).
\end{aligned}
\]
Then the exact block evidence and moment numerators are the one-dimensional integrals
\[
A^{(r)}_{ij}
= \int_0^1 \mu^r\,\exp\!\big\{\ell_{ij}(\mu)+\log\pi(\mu)\big\}\,d\mu,
\qquad r\in\{0,1,2,\dots\}.
\]
A $Q$-node Gauss--Legendre rule on $(0,1)$ gives the deterministic approximation
\[
\widehat A^{(r)}_{ij}
= \sum_{q=1}^Q w_q^{\mathrm{GL}}\,\mu_q^r\,\exp\{\ell_{ij}(\mu_q)+\log\pi(\mu_q)\}.
\]
Its convergence and error rate depend on the endpoint behavior and regularity of the transformed
integrand. No monotonicity in $Q$ and no routine-wide algebraic rate are asserted here.
\end{proposition}
\begin{proof}
The unknown block parameter is one-dimensional and supported on $(0,1)$. Multiplication of the
Beta-observation likelihood by the Beta prior gives the displayed nonnegative integrand. Its
integral is exactly the block evidence; multiplication by $\mu^r$ gives the moment numerator. The
finite weighted sum is the Gauss--Legendre approximation after the standard affine map from
$(-1,1)$ to $(0,1)$.
\end{proof}

\begin{remark}[Quadrature error assessment]
\label{rem:beta-quadrature-stability}
The block log-posterior $\ell_{ij}(\mu)+\log\pi(\mu)$ is not generally log-concave and may have
non-smooth endpoint behavior after exponentiation. Implementations therefore compare nested node
counts or an independently more accurate reference calculation, evaluate in log space, and inspect
endpoint contributions. A certified error statement must be tied to the chosen transformation,
node rule, and integrand regularity.
\end{remark}"""
text = replace_once(text, old, new, label="beta quadrature proposition")
write(rel, text)

# ---------------------------------------------------------------------------
# 3. Sum-product theorem scope, complexity, decision theory, and memory remarks
# ---------------------------------------------------------------------------
rel = "book/chapters/05-sum-product.tex"
text = read(rel)
text = replace_once(
    text,
    r"restate that $k$ ordered boundaries require at least $k$ admissible positions on each side.",
    r"restate that a $k$-segment prefix requires at least $k$ observations and a $k$-segment suffix requires at least $k$ observations.",
    label="forward support explanation",
)
old = r"""\begin{enumerate}
\item The posterior $P(k\mid y)$ given by \eqref{eq:post-k} is exact.
\item The boundary marginal $P(t_p=h\mid y,k)$ given by \eqref{eq:boundary-post} is exact.
\item For any fixed $k$, the joint MAP segmentation \eqref{eq:joint-map-k} is obtained exactly by the
max-sum recursion $M_{k+1,j}=\max_h\{M_{k,h}+\log\widetilde A^{(0)}_{hj}\}$ with backpointers
$h^\star(k+1,j)$ and subsequent backtracking (the full routine of Section~\ref{sec:algorithms};
detailed proof in Appendix~\ref{app:max-sum-proof}).
\item The segment-wise moments \eqref{eq:segmom-seg} and regression-curve moments \eqref{eq:segmom}
are exact posterior moments conditional on $k$.
\end{enumerate}"""
new = r"""\begin{enumerate}
\item The posterior $P(k\mid y)$ given by \eqref{eq:post-k} is exact.
\item The boundary marginal $P(t_p=h\mid y,k)$ given by \eqref{eq:boundary-post} is exact.
\item The segment-wise moments \eqref{eq:segmom-seg} and regression-curve moments \eqref{eq:segmom}
are exact posterior moments conditional on $k$.
\end{enumerate}
The separate max-sum recursion and backtracking theorem for the joint MAP partition is stated and
proved immediately after its statement in Theorem~\ref{thm:map-correctness}."""
text = replace_once(text, old, new, label="DP theorem scope")
text = text.replace(r"\mathbb{E}[m_\star(\theta)^r\mid(i,j]]", r"\mathbb{E}[m_\star(\theta)^r\mid(i,j]]")
old = r"""\begin{theorem}[Time and space complexity]
\label{thm:dp-complexity}
Assume single-block evidences $A^{(0)}_{ij}$ are available for all $0\le i<j\le n$.
Then computing all forward and backward messages via \eqref{eq:LR} requires
$\mathcal{O}(k_{\max} n^2)$ arithmetic operations. Storing the block-evidence matrix requires
$\mathcal{O}(n^2)$ memory. Storing all forward and backward DP layers requires
$\mathcal{O}(k_{\max} n)$ memory; max-sum backpointers require an additional
$\mathcal{O}(k_{\max} n)$; storing all fixed-$k$ boundary marginals requires
$\mathcal{O}(k_{\max} n)$; and storing Bayes-curve moment arrays for $R$ requested moments requires
$\mathcal{O}(Rn)$ after the block matrix is available, or $\mathcal{O}(Rn^2)$ if all
block-specific moment numerators are retained.
\end{theorem}

\begin{proof}
For each fixed $k$, the forward recursion computes $L_{k+1,j}$ for $j=1,\dots,n$, and each
$L_{k+1,j}$ sums at most $j-k$ terms. Thus the forward work per $k$ is
$\sum_{j=1}^n \mathcal{O}(j)=\mathcal{O}(n^2)$. The backward recursion is analogous. Repeating
for $k=0,\dots,k_{\max}-1$ gives $\mathcal{O}(k_{\max}n^2)$ time. Memory follows by counting
array sizes.
\end{proof}"""
new = r"""\begin{theorem}[Time and space complexity]
\label{thm:dp-complexity}
Assume single-block evidences $A^{(0)}_{ij}$ are available for all $0\le i<j\le n$.
Computing all forward and backward messages via \eqref{eq:LR} requires
$\mathcal{O}(k_{\max}n^2)$ arithmetic operations. The triangular block-evidence matrix requires
$\mathcal{O}(n^2)$ memory, while all forward and backward layers require
$\mathcal{O}(k_{\max}n)$ memory. Max-sum values and backpointers each require
$\mathcal{O}(k_{\max}n)$ additional memory. Boundary-event probabilities for every $k$ and candidate
index require $\mathcal{O}(k_{\max}n)$ output storage. Ordered-boundary distributions for every
triple $(k,p,h)$ require $\mathcal{O}(k_{\max}^2n)$ output storage, or $\mathcal{O}(kn)$ for one
fixed segment count $k$. For $R$ requested reporting moments, retaining all block moment numerators
requires $\mathcal{O}(Rn^2)$ memory and the final Bayes curves require $\mathcal{O}(Rn)$; explicitly
retaining every block-cover contribution over all $k$ can require
$\mathcal{O}(k_{\max}n^2)$ additional workspace but is not necessary when those contributions are
accumulated sequentially.
\end{theorem}

\begin{proof}
For each fixed $k$, the forward recursion computes $L_{k+1,j}$ for $j=1,\dots,n$, and each
$L_{k+1,j}$ sums at most $j-k$ terms. Thus the forward work per $k$ is
$\sum_{j=1}^n \mathcal{O}(j)=\mathcal{O}(n^2)$. The backward recursion is analogous. Repeating
for $k=0,\dots,k_{\max}-1$ gives $\mathcal{O}(k_{\max}n^2)$ time. A message or backpointer array
has one entry per $(k,j)$ state. Boundary-event output has one value per $(k,h)$, whereas the full
ordered-boundary output has one value per $(k,p,h)$ and therefore
$\sum_{k\le k_{\max}}\mathcal O(kn)=\mathcal O(k_{\max}^2n)$ entries. The remaining statements
follow by counting the triangular block arrays, reporting-moment arrays, and optional block-cover
workspace.
\end{proof}"""
text = replace_once(text, old, new, label="DP complexity theorem")
old = r"""\begin{corollary}[Joint MAP is Bayes-optimal under 0--1 loss on the full segmentation]
\label{cor:map-01}
For the loss $\ell_{01}(t,t')=\mathbbm{1}\{t\neq t'\}$, the Bayes estimator $t^\star$ minimizing
$\mathbb{E}[\ell_{01}(t,t')\mid y,k]$ is the joint MAP segmentation
$\widehat{t}^{\,\mathrm{MAP}}(k)$ from \eqref{eq:joint-map-k}. For any additive per-segment loss
$\ell_{\mathrm{add}}(t,t')=\sum_q \ell_q(t_q,t'_q)$ that does not decompose as the pointwise
indicator on the full vector, the Bayes-optimal estimator can differ from
$\widehat{t}^{\,\mathrm{MAP}}(k)$.
\end{corollary}

\begin{proof}
The 0--1 loss's expectation equals $1-p(t=t'\mid y,k)$, which is minimized by the joint posterior
mode, i.e., the joint MAP. Pointwise (say Hamming-type) losses decompose differently and are
minimized by the marginal modes, which we have just noted need not coincide with the joint MAP.
\end{proof}"""
new = r"""\begin{corollary}[Decision rules for full-vector and coordinatewise losses]
\label{cor:map-01}
For the loss $\ell_{01}(t,t')=\mathbbm{1}\{t\neq t'\}$ on the full boundary vector, the Bayes
estimator is the joint MAP segmentation $\widehat{t}^{\,\mathrm{MAP}}(k)$ from
\eqref{eq:joint-map-k}. For a separable coordinatewise loss and an unconstrained Cartesian action
space, each coordinate can instead be chosen by minimizing its marginal posterior risk; under
coordinatewise 0--1 loss this gives the marginal modes. If the reported action must belong to the
strictly ordered partition space $\mathcal T^+_{k,n}$, coordinatewise marginal modes can be
inadmissible, and the Bayes action must minimize posterior expected loss over that constrained set.
It can differ from both the coordinatewise marginal modes and the joint MAP partition.
\end{corollary}

\begin{proof}
The posterior risk of $t'$ under full-vector 0--1 loss is $1-p(t=t'\mid y,k)$, so any joint
posterior mode minimizes it. Under a separable loss on an unconstrained Cartesian action space, the
posterior risk is a sum of coordinatewise risks, each minimized using the corresponding marginal
posterior. Restricting the action to strictly ordered boundary vectors couples the coordinates, so
independent coordinatewise minimization no longer solves the constrained decision problem.
\end{proof}"""
text = replace_once(text, old, new, label="decision corollary")
old = r"""\begin{remark}[Memory/time trade-offs for large $n$]
\label{rem:mem-time}
Two implementation variants are standard. (i) \emph{Streaming $k$-layers}: keeping only two
consecutive layers of $(L,R)$ gives memory $\mathcal{O}(n)$ (plus $\mathcal{O}(n^2)$ for the
block-evidence matrix) at no time cost, but backpointers and boundary marginals must then be
recomputed. (ii) \emph{Checkpointing}: storing every $\sqrt{k_{\max}}$th $k$-layer gives memory
$\mathcal{O}(\sqrt{k_{\max}}\,n)$ and at most $\mathcal{O}(\sqrt{k_{\max}}\,k_{\max}\,n^2)$ time,
because recomputing from the nearest checkpoint needs at most $\sqrt{k_{\max}}$ forward layers.
When the $n^2$ block-evidence matrix is itself prohibitive, an alternative is to evaluate block
evidences on demand during the DP sweep at cost $\mathcal{O}(k_{\max}n^2)$ total (constant-time
per block via prefix sums).
\end{remark}"""
new = r"""\begin{remark}[Memory/time trade-offs for large $n$]
\label{rem:mem-time}
Keeping only the current and previous $k$-layers reduces message memory to $\mathcal{O}(n)$, in
addition to any stored block array, when only terminal fixed-$k$ evidences are required. Posterior
marginals and MAP paths then require retained layers or a documented recomputation pass. Checkpoint
spacing can trade retained layers against recomputation, but its exact cost depends on which
summaries are requested and is therefore an implementation choice rather than a universal bound.
When the $n^2$ block-evidence matrix is prohibitive and every segment score is available in constant
time from prefix summaries, scores can instead be evaluated on demand during the DP sweep. This
uses less block storage but performs up to $\mathcal{O}(k_{\max}n^2)$ segment evaluations because
the same segment can be revisited at different DP layers.
\end{remark}"""
text = replace_once(text, old, new, label="memory tradeoff")
write(rel, text)

# ---------------------------------------------------------------------------
# 4. Joint-MAP implementation chapter: typos, references, and memory accounting
# ---------------------------------------------------------------------------
rel = "book/chapters/06-joint-map.tex"
text = read(rel)
text = replace_once(text, r"Each \texttt{BlockRoutine} must return: return a finite log evidence", r"Each \texttt{BlockRoutine} must return a finite log evidence", label="return typo")
text = replace_once(
    text,
    r"Similarly, Bayes regression moments (Theorem~\ref{thm:dp-correctness}, Step~6; cf.\ \eqref{eq:segmom})",
    r"Similarly, Bayes regression moments (Proposition~\ref{prop:block-covering-decomposition}; cf.\ \eqref{eq:segmom})",
    label="block covering reference",
)
text = replace_once(text, r"""This convention is part
for any routine""", r"""This convention is required
for any routine""", label="signed log typo")
old = r"""Conjugate block precomputation & $\Theta(n^2)$ & $\Theta(n^2)$ & prefix summaries; invalid blocks stored as $-\infty$ \\
Exact DP, fixed score matrix & $\Theta(k_{\max}n^2)$ & $\Theta(k_{\max}n)$--$\Theta(k_{\max}n^2)$ & logsumexp normalization \\
Boundary marginals and Bayes curve & $\Theta(k_{\max}n^2)$ & $\Theta(k_{\max}n^2)$ if all block covers are retained & common evidence normalizer \\
MAP backtracking & $\Theta(k_{\max}n^2)$ & $\Theta(k_{\max}n)$ plus backpointers & deterministic tie rule \\"""
new = r"""Conjugate block precomputation & $\Theta(n^2)$ & $\Theta(n^2)$ & prefix summaries; invalid blocks stored as $-\infty$ \\
Exact sum-product messages & $\Theta(k_{\max}n^2)$ & $\Theta(k_{\max}n)$ in addition to the block matrix & logsumexp normalization \\
Boundary and Bayes-curve extraction & $\Theta(k_{\max}n^2)$ for streamed fixed-$k$ summaries & output dependent; up to $\Theta(k_{\max}^2n)$ for every ordered-boundary table & common evidence normalizer \\
Max-sum recursion and backtracking & $\Theta(k_{\max}n^2)$ & $\Theta(k_{\max}n)$ for values and backpointers & deterministic tie rule \\"""
text = replace_once(text, old, new, label="complexity table rows")
old = r"""\item The forward DP (Algorithm~\ref{alg:dp-log}) runs in $\Theta(k_{\max}n^2)$ time. The required working memory is $\Theta(k_{\max}n)$ if only $p(y\mid k)$ and $p(k\mid y)$ are needed, and $\Theta(k_{\max}n^2)$ if boundary marginals or backpointers must be retained.
\item The max-sum recursion plus backtracking (Algorithms~\ref{alg:map-forward}--\ref{alg:map-backtrack}) runs in $\Theta(k_{\max}n^2)$ time and $\Theta(k_{\max}n)$ working space."""
new = r"""\item The forward and backward DP (Algorithm~\ref{alg:dp-log}) runs in $\Theta(k_{\max}n^2)$ time and stores $\Theta(k_{\max}n)$ messages. Boundary-event summaries for every $k$ require $\Theta(k_{\max}n)$ output storage; all ordered-boundary distributions for every $(k,p,h)$ require $\Theta(k_{\max}^2n)$ output storage. Explicit storage of every block-cover contribution can require $\Theta(k_{\max}n^2)$, but it is not required for streamed accumulation.
\item The max-sum recursion plus backtracking (Algorithms~\ref{alg:map-forward}--\ref{alg:map-backtrack}) runs in $\Theta(k_{\max}n^2)$ time and $\Theta(k_{\max}n)$ working space, including one predecessor per DP state."""
text = replace_once(text, old, new, label="complexity proposition items")
old = r"""Storing one message per $(k,j)$ is
$\Theta(k_{\max}n)$; retaining the full boundary-decomposition terms or equivalent per-state
marginal workspace produces the stated $\Theta(k_{\max}n^2)$ mode. The max-sum recurrence visits
the same predecessor sets, stores one value and one predecessor per state, and backtracking takes
only $O(k_{\max})$, proving (iii)."""
new = r"""Storing one message per $(k,j)$ is
$\Theta(k_{\max}n)$. Boundary-event output has one value per $(k,h)$, while the complete
ordered-boundary output has one value per $(k,p,h)$ and therefore $\Theta(k_{\max}^2n)$ entries.
Block-cover terms can be accumulated without retaining every $(k,i,j)$ contribution. The max-sum
recurrence visits the same predecessor sets, stores one value and one predecessor per state, and
backtracking takes only $O(k_{\max})$, proving (iii)."""
text = replace_once(text, old, new, label="complexity proof")
write(rel, text)

# ---------------------------------------------------------------------------
# 5. Poisson-process occupancy prior: conditional odds, not occupancies
# ---------------------------------------------------------------------------
rel = "book/chapters/07-design-priors.tex"
text = read(rel)
old = r"""\begin{proposition}[Interval probabilities from a Poisson boundary process]
\label{prop:pp-index-uniform}
Let potential interior boundaries arise from an inhomogeneous Poisson process of intensity
$\lambda(u)$ over physical coordinate intervals $(u_{j-1},u_j]$. Conditional on selecting a fixed
number of distinct candidate intervals and ignoring within-interval locations, an appropriate local
hazard weight is
\[
 h_x(j)\propto 1-\exp\!\left\{-\int_{u_{j-1}}^{u_j}\lambda(v)\,dv\right\}.
\]
For constant intensity and equal-width intervals, $h_x(j)$ is constant and the induced fixed-count
boundary prior is index-uniform when $c_x\equiv1$.
\end{proposition}
\begin{proof}
A Poisson process places at least one event in interval $I_j$ with probability
$1-e^{-\Lambda_j}$, where $\Lambda_j=\int_{I_j}\lambda(v)dv$. Conditional fixed-count selection
from candidate intervals therefore has product weights proportional to these interval probabilities.
If all $\Lambda_j$ are equal, every ordered subset of a given size receives the same product.
\end{proof}"""
new = r"""\begin{proposition}[Fixed-count interval occupancy from a Poisson boundary process]
\label{prop:pp-index-uniform}
Let potential interior boundaries arise from an inhomogeneous Poisson process of intensity
$\lambda(u)$ over disjoint physical coordinate intervals $(u_{j-1},u_j]$, and let
$\Lambda_j=\int_{u_{j-1}}^{u_j}\lambda(v)\,dv$. If the process is reduced to the occupancy
indicators of these intervals and one conditions on exactly $m$ distinct occupied candidate
intervals, then a subset $B$ with $|B|=m$ has probability proportional to
\[
 \prod_{j\in B}\frac{1-e^{-\Lambda_j}}{e^{-\Lambda_j}}
 =\prod_{j\in B}\bigl(e^{\Lambda_j}-1\bigr).
\]
Thus the compatible fixed-count local boundary factor is
$h_x(j)\propto e^{\Lambda_j}-1$. For constant intensity and equal-width intervals this factor is
constant, so the induced fixed-count boundary prior is index-uniform when $c_x\equiv1$.
\end{proposition}
\begin{proof}
Counts in disjoint Poisson-process intervals are independent. Their occupancy indicators are
therefore independent Bernoulli variables with $p_j=1-e^{-\Lambda_j}$. For a subset $B$ of
occupied intervals,
\[
 \Pr(B)=\prod_{j\in B}p_j\prod_{j\notin B}(1-p_j)
 =\left\{\prod_j(1-p_j)\right\}\prod_{j\in B}\frac{p_j}{1-p_j}.
\]
After conditioning on $|B|=m$, the first factor is common to all subsets and the local odds are
$p_j/(1-p_j)=e^{\Lambda_j}-1$. Equal integrated intensities give equal odds and hence a uniform
prior over subsets of the fixed size.
\end{proof}"""
text = replace_once(text, old, new, label="Poisson occupancy proposition")
text = replace_once(
    text,
    r"The proposition weights candidate boundary intervals. It does not imply that multiplying each",
    r"The proposition weights candidate boundary intervals after conditioning on their number. Without that conditioning, both occupied and unoccupied Bernoulli factors remain in the probability. It does not imply that multiplying each",
    label="Poisson prior remark",
)
write(rel, text)

rel = "paper/sections/05-structured-extensions.tex"
text = read(rel)
old = r"""A boundary-process interpretation applies only to the hazard factor. If potential boundaries follow an inhomogeneous Poisson process with intensity $\lambda(v)$, then an interval-level candidate weight is
\begin{equation}
 h_u(j)\propto 1-\exp\!\left\{-\int_{u_{j-1}}^{u_j}\lambda(v)\dd v\right\}.
\label{eq:poisson-hazard}
\end{equation}
Constant intensity and equal-width intervals recover an index-uniform fixed-count boundary prior when $c_u\equiv1$. Equation~\eqref{eq:poisson-hazard} does not justify multiplying a segment likelihood by its physical length; a duration-dependent cohesion is a separate prior choice."""
new = r"""A boundary-process interpretation applies only to the hazard factor. Let
$\Lambda_j=\int_{u_{j-1}}^{u_j}\lambda(v)\dd v$ for an inhomogeneous Poisson process. Interval
occupancies are independent Bernoulli variables with probability $1-e^{-\Lambda_j}$. Conditional
on exactly $m$ occupied candidate intervals, the subset probability is proportional to the product
of Bernoulli odds, so the compatible local factor is
\begin{equation}
 h_u(j)\propto \frac{1-e^{-\Lambda_j}}{e^{-\Lambda_j}}=e^{\Lambda_j}-1.
\label{eq:poisson-hazard}
\end{equation}
Constant intensity and equal-width intervals recover an index-uniform fixed-count boundary prior when $c_u\equiv1$. Without conditioning on the number of occupied intervals, both occupied and unoccupied Bernoulli factors are part of the prior. Equation~\eqref{eq:poisson-hazard} does not justify multiplying a segment likelihood by its physical length; a duration-dependent cohesion is a separate prior choice."""
text = replace_once(text, old, new, label="paper Poisson hazard")
write(rel, text)

# ---------------------------------------------------------------------------
# 6. Latent-group criterion: exact implementation score and n_g scaling
# ---------------------------------------------------------------------------
rel = "book/chapters/09-score-templates.tex"
text = read(rel)
old = r"""When group labels are unknown, let $\tau_g$ denote a deterministic candidate changepoint template
for group $g$, and let $S_s(\tau_g)>0$ denote the score assigned to sequence $s$ under that template.
The score may be a product of sequence-specific segment marginal likelihoods and prior factors. It
is not assumed to integrate to one over the sample space of sequence $s$. For weights
$\pi_g\ge0$ with $\sum_g\pi_g=1$, define"""
new = r"""When group labels are unknown, let $\tau_g=(k_g,t^{(g)})$ denote a deterministic candidate
changepoint template for group $g$. The archived implementation uses the sequence--template score
\begin{equation}
 S_s(\tau)=\frac{p(k)}{C_k}\prod_{q=1}^{k}
 A^{(0,s)}_{t_{q-1},t_q}\,\gamma_u(t_{q-1},t_q),
\label{eq:template-sequence-score}
\end{equation}
with the convention that an impossible segment gives score zero. This positive score is not assumed
to integrate to one over the sample space of sequence $s$. For weights $\pi_g\ge0$ with
$\sum_g\pi_g=1$, define"""
text = replace_once(text, old, new, label="template score definition")
old = r"""For fixed $r$, maximization over the group weights gives
$\pi_g=S^{-1}\sum_s r_{sg}$. If $\log S_s(\tau)$ is additive over the segments of $\tau$, then the
template update is an exact max-sum recursion with group-specific weighted segment score
\begin{equation}
 \ell^{(g)}_{ij}=\sum_{s=1}^{S}r_{sg}\log A^{(s)}_{ij}+\log\gamma_u(i,j).
\label{eq:template-blockscore}
\end{equation}"""
new = r"""For fixed $r$, maximization over the group weights gives
$\pi_g=S^{-1}\sum_s r_{sg}$. Let $n_g=\sum_s r_{sg}$. Substituting
Equation~\eqref{eq:template-sequence-score} into $\mathcal Q$ shows that, for a candidate segment
count $k$, the group-$g$ template contribution is
\begin{equation}
 n_g\{\log p(k)-\log C_k\}
 +\sum_{q=1}^{k}\ell^{(g)}_{t_{q-1},t_q},
\qquad
 \ell^{(g)}_{ij}=\sum_{s=1}^{S}r_{sg}\log A^{(0,s)}_{ij}
                  +n_g\log\gamma_u(i,j).
\label{eq:template-blockscore}
\end{equation}
Thus the update is an exact max-sum recursion at fixed $k$, followed by comparison across $k$ using
the displayed count offset. The convention $0\log0=0$ applies when $r_{sg}=0$, while a positive
responsibility on a zero sequence--template score makes that candidate value $-\infty$."""
text = replace_once(text, old, new, label="template local score")
old = r"""\begin{theorem}[Monotonicity of exact latent-group updates]
\label{thm:em-monotone}
Consider an iteration that (i) sets $r$ according to Equation~\eqref{eq:template-resp}, (ii)
maximizes $\mathcal Q$ over $\pi$, and (iii) exactly maximizes each weighted template score over its
declared finite candidate set. Then the iteration cannot decrease $\mathcal F$.
\end{theorem}"""
new = r"""\begin{theorem}[Monotonicity of exact latent-group updates]
\label{thm:em-monotone}
Consider an iteration that (i) sets $r$ according to Equation~\eqref{eq:template-resp}, (ii)
maximizes $\mathcal Q$ over $\pi$, and (iii) exactly maximizes each responsibility-weighted template
criterion using the local score and segment-count offset in Equation~\eqref{eq:template-blockscore}
over its declared finite candidate set. Then the iteration cannot decrease $\mathcal F$.
\end{theorem}"""
text = replace_once(text, old, new, label="template monotonicity statement")
old = r"""  update each $\tau_g$ by exact responsibility-weighted max-sum recursion\;"""
new = r"""  for each group $g$, set $n_g\leftarrow\sum_s r_{sg}$, build the local scores in Equation~\eqref{eq:template-blockscore}, solve the max-sum recursion at every allowed $k$, and select the offset-adjusted maximizer using $n_g\{\log p(k)-\log C_k\}$\;"""
text = replace_once(text, old, new, label="template algorithm")
write(rel, text)

rel = "paper/sections/05-structured-extensions.tex"
text = read(rel)
old = r"""For sequence $s$ and deterministic segmentation template $\tau_g$, let $S_s(\tau_g)>0$ denote the specified sequence--template score. The archived implementation does not require these scores to integrate to one over the data space. We therefore define the finite criterion"""
new = r"""For sequence $s$ and deterministic segmentation template $\tau=(k,t)$, the archived implementation uses
\begin{equation}
 S_s(\tau)=\frac{p(k)}{C_k}\prod_{q=1}^{k}
 A^{(0,s)}_{t_{q-1},t_q}\gamma_u(t_{q-1},t_q).
\label{eq:sequence-template-score-paper}
\end{equation}
These positive scores are not required to integrate to one over the data space. We therefore define the finite criterion"""
text = replace_once(text, old, new, label="paper template score")
old = r"""The mixing-weight update is $\pi_g=S^{-1}\sum_s r_{sg}$. When $\log S_s(\tau)$ factorizes over template segments, the template update is a max-sum dynamic program with local score
\begin{equation}
\ell^{(g)}_{ij}=\sum_{s=1}^{S}r_{sg}\log A^{(s)}_{ij}+\log\gamma_u(i,j).
\label{eq:template-local-score}
\end{equation}"""
new = r"""The mixing-weight update is $\pi_g=S^{-1}\sum_s r_{sg}$. Let $n_g=\sum_s r_{sg}$. For a candidate segment count $k$, substitution of Equation~\eqref{eq:sequence-template-score-paper} into the minorizer gives the count offset $n_g\{\log p(k)-\log C_k\}$ and the max-sum local score
\begin{equation}
\ell^{(g)}_{ij}=\sum_{s=1}^{S}r_{sg}\log A^{(0,s)}_{ij}
                  +n_g\log\gamma_u(i,j).
\label{eq:template-local-score}
\end{equation}
The template update solves the fixed-$k$ max-sum problem and then compares the offset-adjusted terminal values across the allowed segment counts."""
text = replace_once(text, old, new, label="paper template local score")
text = replace_once(
    text,
    r"An iteration that sets $r$ by Equation~\eqref{eq:auxiliary-weights}, maximizes $\mathcal Q$ over $\pi$, and exactly maximizes every responsibility-weighted template score over its declared finite candidate set cannot decrease $\mathcal F$.",
    r"An iteration that sets $r$ by Equation~\eqref{eq:auxiliary-weights}, maximizes $\mathcal Q$ over $\pi$, and exactly maximizes every responsibility-weighted template criterion using Equation~\eqref{eq:template-local-score} and its segment-count offset over the declared finite candidate set cannot decrease $\mathcal F$.",
    label="paper template theorem",
)
write(rel, text)

# ---------------------------------------------------------------------------
# 7. Nonconjugate approximation statements: explicit assumptions and limits
# ---------------------------------------------------------------------------
rel = "book/chapters/10-nonconjugate.tex"
text = read(rel)
text = replace_once(
    text,
    r"""Suppose $b''(\theta)>0$ for all $\theta$ (canonical GLM), the prior is log-concave
$((\log\pi)''(\theta)\le 0)$, and $W_{ij}>0$.""",
    r"""Suppose $b$ and $\log\pi$ are twice differentiable on a convex parameter domain,
$b''(\theta)>0$ for all $\theta$ (canonical GLM), the prior is log-concave
$((\log\pi)''(\theta)\le 0)$, and $W_{ij}>0$.""",
    label="strict concavity assumptions",
)
old = r"""Index a sequence of blocks by an effective information size $N=W_{ij}\to\infty$. Let
$\widehat\theta_N$ be the unique interior maximizer of $\Psi_N$ and
$H_N=-\Psi_N''(\widehat\theta_N)$. Assume there are constants $c,C,\delta>0$, independent of the
block, such that: (i) $cN\le H_N\le CN$; (ii) $\Psi_N$ is four times continuously differentiable
on $|\theta-\widehat\theta_N|\le\delta$ and
$|\Psi_N^{(r)}(\theta)|\le CN$ for $r=3,4$ there; and (iii) the integral outside that neighborhood
is at most the local Gaussian leading term times $O(N^{-1})$. Then"""
new = r"""Index a sequence of blocks by an effective information size $N=W_{ij}\to\infty$. Let
$\widehat\theta_N$ be the unique interior maximizer of $\Psi_N$ and
$H_N=-\Psi_N''(\widehat\theta_N)$. Assume there are constants $c,C,\delta>0$, independent of the
block, such that: (i) $cN\le-\Psi_N''(\theta)\le CN$ throughout
$|\theta-\widehat\theta_N|\le\delta$; (ii) $\Psi_N$ is four times continuously differentiable on
that neighborhood and $|\Psi_N^{(r)}(\theta)|\le CN$ for $r=3,4$; and (iii) the integral outside
the neighborhood is at most the local Gaussian leading term times $O(N^{-1})$. Then"""
text = replace_once(text, old, new, label="Laplace assumptions")
text = replace_once(
    text,
    r"""For a twice continuously differentiable test function $g$ whose derivatives are uniformly bounded
in the same neighborhood,""",
    r"For a bounded twice continuously differentiable test function $g$ whose first two derivatives are uniformly bounded in the same neighborhood,",
    label="Laplace test function",
)
old = r"""\begin{proposition}[Monotone improvement of variational block bounds]
\label{prop:var-monotone}
Coordinate-ascent updates on variational parameters (e.g.\ $\xi$ for logistic blocks) and on
a Gaussian $q(\theta)$ increase $\mathcal{L}_{ij}(q)$ until a stationary point; the resulting
$\underline{A}^{(0)}_{ij}:=\exp\{\mathcal{L}_{ij}(q)\}$ is a certified lower bound on
$A^{(0)}_{ij}$.
\end{proposition}

\begin{proof}
Each coordinate update is an exact maximization of $\mathcal{L}_{ij}(q)$ with respect to a
subset of variables holding others fixed. Therefore $\mathcal{L}_{ij}(q)$ is non-decreasing
under the updates. Because $\mathcal{L}_{ij}$ is a Jensen lower bound on $\log A^{(0)}_{ij}$,
exponentiating yields a valid evidence lower bound.
\end{proof}"""
new = r"""\begin{proposition}[Monotone improvement of variational block bounds]
\label{prop:var-monotone}
Every exact coordinate-ascent update of the local variational parameters (for example $\xi$ in a
logistic block) or of a Gaussian $q(\theta)$ leaves the block evidence lower bound
$\mathcal L_{ij}(q)$ non-decreasing. If the iterates converge to an interior fixed point and the
objective is differentiable there, that fixed point is coordinatewise stationary. At every
iteration, $\underline{A}^{(0)}_{ij}:=\exp\{\mathcal{L}_{ij}(q)\}$ is a certified lower bound on
$A^{(0)}_{ij}$.
\end{proposition}

\begin{proof}
Each coordinate update is an exact maximization of $\mathcal{L}_{ij}(q)$ with respect to a subset
of variables holding the others fixed, so the ELBO is non-decreasing. At a converged interior fixed
point, the first-order derivative along each updated coordinate block vanishes under the stated
regularity, which is coordinatewise stationarity. Because $\mathcal{L}_{ij}$ is a Jensen lower
bound on $\log A^{(0)}_{ij}$ at every iterate, exponentiating yields a valid evidence lower bound.
\end{proof}"""
text = replace_once(text, old, new, label="variational monotonicity")
old = r"""\subsection{Expectation propagation (EP) for GLM blocks}
Expectation propagation (EP) \citep{minka2001ep} provides a flexible deterministic approximation to the block posterior and block evidence by iteratively refining local site approximations and matching moments.
EP approximates each likelihood term $p(y_t\mid\theta)$ by a Gaussian site
$\tilde t_t(\theta)=c_t\exp\{-\tfrac{1}{2}a_t\theta^2+b_t\theta\}$ so that moments match those
of the tilted distribution while the scalar $c_t>0$ records the site-normalization contribution.
After convergence, the approximate posterior is Gaussian and the approximate evidence is
$\widehat A^{(0,\mathrm{EP})}_{ij}=\int \pi(\theta)\prod_{t=i+1}^j \tilde t_t(\theta)\,d\theta$.
The constants $c_t$ are part of the returned log evidence; dropping them changes the DP score.
EP is neither a bound nor guaranteed to converge, but is often accurate in canonical GLMs."""
new = r"""\subsection{Expectation propagation (EP) for GLM blocks}
Expectation propagation (EP) \citep{minka2001ep} provides a deterministic approximation to the
block posterior and block evidence by iteratively refining local site approximations and matching
moments. In the formulation used here the prior is Gaussian, and each likelihood term
$p(y_t\mid\theta)$ is approximated by a Gaussian site
$\tilde t_t(\theta)=c_t\exp\{-\tfrac{1}{2}a_t\theta^2+b_t\theta\}$. After convergence, the product
of the Gaussian prior and sites is Gaussian and the approximate evidence is
$\widehat A^{(0,\mathrm{EP})}_{ij}=\int \pi(\theta)\prod_{t=i+1}^j \tilde t_t(\theta)\,d\theta$.
The constants $c_t$ are part of the returned log evidence; dropping them changes the DP score.
A non-Gaussian prior requires an additional site approximation or numerical integral and is not
covered by this Gaussian-posterior description. EP is neither a bound nor guaranteed to converge."""
text = replace_once(text, old, new, label="EP prior scope")
text = replace_once(text, r"\KwIn{Canonical GLM; prior $\pi(\theta)$; tolerance and maximum iterations}", r"\KwIn{Canonical GLM; Gaussian prior $\pi(\theta)$; tolerance and maximum iterations}", label="EP algorithm input")
text = replace_once(text, r"\subsection{Pólya--Gamma augmentation (logistic and negative-binomial)}", r"\subsection{Pólya--Gamma augmentation for binomial-logistic blocks}", label="PG subsection title")
text = replace_once(
    text,
    r"Conditionally on $\omega_{i+1:j}$, the block likelihood is quadratic in $\theta$:",
    r"The binomial coefficient, when present, is independent of $\theta$ and must be retained in the base-measure contribution to the block evidence. Conditionally on $\omega_{i+1:j}$, the remaining block likelihood is quadratic in $\theta$:",
    label="PG base measure",
)
old = r"""The strength of Assumption~\ref{ass:uniform-block-error} depends on the block routine. The exact
quadrature routine satisfies it with $\varepsilon$ at the quadrature-grid resolution; Laplace, JJ,
and PG mean field satisfy it under standard regularity conditions on the GLM link and the prior;
EP satisfies it whenever the message-passing iteration converges and the resulting moment match is
accurate (worked bounds for each routine are stated in Proposition~\ref{prop:uniform-bounds}
below)."""
new = r"""Assumption~\ref{ass:uniform-block-error} is an additional numerical-analysis requirement, not a
consequence of naming an approximation routine. It must be established independently for the
chosen family, parameter domain, candidate-block support, transformation, initialization,
convergence event, and tail treatment. Nested quadrature, a high-accuracy reference calculation,
or a method-specific remainder theorem can provide such a certificate in a particular analysis.
Convergence of an optimizer, monotonicity of an ELBO, or an EP moment match alone does not provide
the required two-sided uniform log-evidence bound."""
text = replace_once(text, old, new, label="block error certification prose")
text = replace_once(
    text,
    r"""Thus uniform blockwise log-evidence control yields uniform control of global posterior \emph{odds}. Turning
these odds bounds into absolute probability error bounds requires an additional margin assumption and is
therefore left implicit.""",
    r"Thus uniform blockwise log-evidence control yields uniform control of global posterior \emph{odds}. Coarse absolute probability bounds follow by normalizing finite weights, as in Corollary~\ref{cor:probability-error-conversion}; a margin condition is needed instead for preservation of a modal ranking.",
    label="odds probability wording",
)
text = replace_once(text, r"$\|\widehat p-p\|_{\mathrm{TV}}\le e^{2\eta}-1$.", r"$\|\widehat p-p\|_{\mathrm{TV}}\le \min\{1,e^{2\eta}-1\}$.", label="TV bound book")
text = replace_once(text, r"Summing the positive parts of $\widehat p_a-p_a$ gives the stated total-variation bound.", r"Summing the positive parts of $\widehat p_a-p_a$ gives the exponential bound, and total variation is always at most one.", label="TV proof book")
write(rel, text)

# Paper TV bound follows the same correction.
rel = "paper/sections/06-approximation-prediction.tex"
text = read(rel)
text = replace_once(text, r"$\|\widehat p-p\|_{\mathrm{TV}}\le e^{2\eta}-1$.", r"$\|\widehat p-p\|_{\mathrm{TV}}\le\min\{1,e^{2\eta}-1\}$.", label="TV bound paper")
text = replace_once(text, r"summing positive probability differences yields the stated total-variation bound.", r"summing positive probability differences yields the exponential bound, while total variation is at most one.", label="TV proof paper")
write(rel, text)

# ---------------------------------------------------------------------------
# 8. Main-paper complexity table: distinguish messages, outputs, and backpointers
# ---------------------------------------------------------------------------
rel = "paper/sections/07-algorithms-complexity.tex"
text = read(rel)
old = r"""Sum-product recursion & $\Theta(k_{\max}n^2)$ & $\Theta(k_{\max}n)$ minimum & log-sum-exp evaluation \\
Boundary and moment extraction & $\Theta(k_{\max}n^2)$ & up to $\Theta(k_{\max}n^2)$ & common marginal-likelihood normalizer \\
Max-sum recursion and backtracking & $\Theta(k_{\max}n^2)$ & $\Theta(k_{\max}n)$ & deterministic predecessor rule \\"""
new = r"""Sum-product recursion & $\Theta(k_{\max}n^2)$ & $\Theta(k_{\max}n)$ for all message layers & log-sum-exp evaluation \\
Boundary and moment extraction & $\Theta(k_{\max}n^2)$ for streamed fixed-$k$ summaries & output dependent; $\Theta(k_{\max}^2n)$ for every ordered-boundary table & common marginal-likelihood normalizer \\
Max-sum recursion and backtracking & $\Theta(k_{\max}n^2)$ & $\Theta(k_{\max}n)$ including backpointers & deterministic predecessor rule \\"""
text = replace_once(text, old, new, label="paper complexity table")
write(rel, text)

print("Phase 6 scientific corrections applied successfully.")
