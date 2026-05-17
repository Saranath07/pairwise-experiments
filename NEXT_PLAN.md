# WiSDoM → A* Conference Plan

A roadmap to harden the current workshop-strength draft into a clear main-track accept — and ideally a Best Paper contender — at NeurIPS / ICML / KDD. **No deadlines.** The goal is to transform the paper from "a successful experiment" into "a definitive framework." We move on from each item only when we can defend it against an adversarial reviewer.

---

## 0. Guiding principles

1. **Close every loop the current draft opens.** The draft already names its own weaknesses (exponent gap, BTL reliance, semi-synthetic DMControl, brief ablation). Reviewers will quote those lines back. Each must become a resolved subsection, not a future-work bullet.
2. **Attribute the win.** It is not enough to beat PARWiS. We must isolate *which* component (bracket warm-start vs. winner-focused SDP vs. candidate restriction) buys *which* fraction of the gain, on *which* regime.
3. **Stress-test outside BTL.** A top venue will ask: what happens when BTL is wrong? We need a robustness story, not a disclaimer.
4. **Land one truly real experiment.** Pre-computed Arena Elo is not a live oracle. One end-to-end LLM-as-judge run on a real preference dataset converts the motivation paragraph into evidence.
5. **Compute is a budget too.** Frugality must include wall-clock and memory at scale, not just query count.
6. **Generalize beyond winner-only.** Top-K selection, plug-and-play estimators, broader applicability — these convert WiSDoM from a single algorithm into a *framework*.

---

## Phase 1 — Theoretical Tightening (the mathematical hook)

### 1a. Close the exponent gap
Empirical slope of $\log P(\text{fail})$ vs $\Delta^2$ is $\approx -23$; the L3 lower-bound slope is $-40.4$. Workshop-acceptable; A*-fatal.

Three independent attacks; we need at least one to land.

**Tighten the lower bound.** The current chain (pigeonhole → Bretagnolle–Huber → uniform-prior Lagrangian) is loose at BH, which is a worst-case two-hypothesis test. The right tool for adaptive-sampling settings is:
- Le Cam two-point with adaptive sampling (Chen–Li–Wainwright, or the Kaufmann–Cappé–Garivier framework for best-arm identification).
- A change-of-measure argument à la Garivier–Kaufmann giving the tight $\sum m_i \cdot \mathrm{KL}$ exponent — typically tightens BH constants by exactly the factor that matches our $-23 / -40.4 \approx 0.57$ ratio.
- Fano with composite alternatives to handle the $N-K$ rejected items as one composite hypothesis class.

**Prove an upper bound for WiSDoM.** No formal achievability statement for our algorithm currently exists. A matching upper bound converts "WiSDoM is empirically good" into "WiSDoM is minimax optimal up to constants" — the single highest-leverage theoretical addition.

Sketch:
- Phase 1 success follows from Hoeffding on the bracket (already in §3.4, Eq. 7); rewrite as $1 - O(\log N \cdot e^{-t\Delta^2/2})$.
- Phase 2 success conditional on coverage: bound the posterior gap variance via the winner-focused Fisher matrix; Pukelsheim Ch. 7 gives the rate.
- Combine via union bound: $P(\text{fail}) \leq A \exp(-c' B \Delta^2 / M) + B \exp(-t\Delta^2/2)$.
- With $M = N^{1/4}$ and $t = O(\log N)$ this should match L3 up to log factors.

**Honest gap characterization (fallback).** If neither closes fully, write a "constant-factor analysis" subsection: identify which inequality is loose, by how much, on what instance. Reviewers accept residual gaps if you demonstrate exact understanding of where they live.

### 1b. Formal analysis of the budget split
We currently use $t = 3$ as an empirical value where Phase 1 coverage "saturates" without starving Phase 2. For an A* paper this needs a derivation, not a sweep.

Setting: total budget $B = cN$, Phase 1 cost $t(N-1)$, Phase 2 cost $B - t(N-1)$. Coverage probability after Phase 1 is approximately $1 - (\log_2 N) \exp(-t\Delta^2/2)$. Phase 2 success conditional on coverage scales as $\exp(-c' (B - t(N-1)) \Delta^2 / M)$.

Joint failure probability:
$$P(\text{fail}) \;\lesssim\; (\log N) e^{-t\Delta^2/2} + e^{-c'(cN - tN)\Delta^2 / M}.$$

Optimizing over $t$ gives a closed-form $t^*(c, N, \Delta, M)$. We expect $t^* = \Theta(c)$ with a logarithmic correction in $N$. Even an asymptotic statement ($t^* \to 3$ when $c=5$, $N=100$, $\Delta \sim N^{-1/4}$) would convert "we picked 3" into "the theory predicts 3."

### 1c. Robustness to model misspecification (theoretical)
BTL assumes stochastic transitivity and one score per item. We need a formal characterization of how the min-max design degrades under:
- Bounded BTL violation: $|P(i \succ j) - \sigma(\theta_i - \theta_j)| \leq \epsilon$ for all pairs.
- Cyclic perturbations: $P(i \succ j) = \sigma(\theta_i - \theta_j) + \rho \cdot c_{ij}$ with $c$ skew-symmetric.

Claim to prove: for $\epsilon$ or $\rho$ below a critical threshold (function of $\Delta$), WiSDoM still recovers the BTL-projection winner with probability $1 - o(1)$. This is a *robustness theorem* and pairs with the empirical stress tests in Phase 2 below.

---

## Phase 2 — Empirical Expansion (the industry-scale proof)

### 2a. Comprehensive ablation attribution
**The single most important empirical addition** — the reviewer's first instinct will be to ask whether the SDP is actually doing the work, and a related sharper version is "what if $M = 2$?" (i.e., bracket champion vs. runner-up, then majority-vote — does the whole experimental-design machinery actually beat that?). Build a five-line plot on $k \in \{25, 50, 75, 95\}$ and Netflix + Chatbot Arena:

1. **Phase 1 only** (bracket, then output the bracket winner — no Phase 2 budget spent).
2. **$M = 2$ verification** (Phase 1 + spend remaining budget on a Bernoulli majority-vote between the bracket champion and its final-round opponent). This is the reviewer's strawman: WiSDoM degenerates to "best-of-$t$ bracket + verify."
3. **Phase 1 + uniform queries** on the top-$M$ set with $M = \lceil N^{1/4}\rceil$ (WiSDoM-Lite).
4. **Phase 1 + greedy uncertainty** on the top-$M$ set (PARWiS-style King-of-the-Hill restricted to $\mathcal{C}$).
5. **Full WiSDoM** (Phase 1 + winner-focused SDP).

Each line tells a different story about which component buys the gain. The headline becomes one sentence: *"Going from $M{=}2$ to $M{=}\lceil N^{1/4}\rceil$ buys $W$% on hard regimes; removing the SDP from there costs $X$%; the bracket alone leaves a residual $Y$% on the table; greedy-on-$\mathcal{C}$ recovers only $Z$% of that."*

Promote WiSDoM-Lite (line 3) and the $M{=}2$ strawman (line 2) to named baselines in the headline table. Decision rules:
- If full WiSDoM beats $M{=}2$ verification by $\geq 15$% on hard regimes → candidate-set restriction earns its slot. (This is the *first* gate; without it the entire framework is in question.)
- If full WiSDoM beats WiSDoM-Lite (line 3) by $\geq 15$% on hard regimes → SDP earns its slot too, narrative is intact.
- If WiSDoM ≈ WiSDoM-Lite but both ≫ $M{=}2$ → the candidate-set restriction is the load-bearing idea; the SDP is secondary. Reframe accordingly.
- If WiSDoM ≈ $M{=}2$ verification → the bracket warm-start is the entire contribution; the experimental-design machinery is decorative. Drop or massively scope down.

The $M{=}2$ comparison is non-negotiable for submission. The current ablation table (`appendix/C_ablation_placeholders.tex`) sweeps $M \in \{N^{1/6}, N^{1/4}, \sqrt{N}, N^{3/4}, N\}$ but **does not include $M{=}2$** as a baseline. A reviewer can correctly point out that we have never empirically defended the choice of $M \ge 3$ against the simplest alternative.

**Why the theory predicts $M{=}2$ should lose at small $\Delta$ but possibly win at large $\Delta$.** From Theorem 2' (Phase 1 deliverable):
- Phase 2 slope at $M{=}2$: $-2(B-tN)\Delta^2$. At $M{=}4$: $-2(B-tN)\Delta^2/3$. So $M{=}2$ has a $3\times$ steeper Phase-2 slope when coverage holds.
- Phase 1 coverage exponent depends on the gap from $w$ to the $M$-th best item. For uniform spacing this is $\Delta$ at $M{=}2$ and $3\Delta$ at $M{=}4$, so coverage at $M{=}4$ is $\sim 9\times$ tighter in the tail.
- Net: $M{=}4$ wins where coverage dominates (small $\Delta$, hard regimes); $M{=}2$ wins where Phase-2 dominates (large $\Delta$, easy regimes). The empirical sweep should confirm a crossover.

This crossover prediction is itself publishable as a clean theoretical-empirical match if it holds.

### 2b. Robustness to BTL violations (empirical)
The biggest underweighted piece in the current draft.

**Cyclic / rock-paper-scissors preferences.** Construct $P(i \succ j) = \sigma(\theta_i - \theta_j) + \rho \cdot c_{ij}$ with $c$ skew-symmetric. Sweep $\rho \in [0, 0.4]$. Hypothesis: WiSDoM degrades more *gracefully* than PARWiS because top-$M$ hedging protects against single-hypothesis collapse. If true, this flips the BTL-reliance critique into a strength.

**Thurstonian / probit noise.** Replace BTL logistic with $P(i \succ j) = \Phi((\theta_i - \theta_j)/\sigma)$. Same algorithm, mis-specified link.

**Multidimensional preferences.** Two latent dimensions; oracle samples a dimension per query. Mirrors the LLM-judge case where a model is better at coding but worse at writing. Metric: top-1 accuracy under aggregate (Borda-style) ground truth.

**Deliverable.** A new section §6.X "Robustness beyond BTL" with three subfigures.

### 2c. Large-scale LLM-as-judge experiment ($N \geq 100$)
Current Chatbot Arena: $N=20$, $B=100$, ground truth = pre-computed BTL weights. This is still a synthetic BTL instance dressed up. We need one experiment where the oracle is a *live* model.

- **Dataset.** UltraFeedback or Nectar (confirm candidate count per prompt — UltraFeedback has 4, so Nectar or HelpSteer is more likely the right pick).
- **Candidates.** $N=100$ generations from a fixed prompt across diverse model families (GPT-4o, Claude, Llama-3-70B, Mistral, etc.) so quality genuinely varies.
- **Oracle.** A cheap judge (Llama-3.1-8B-Instruct or Qwen2.5-7B) called per pairwise query with a fixed prompt template; debias position bias by averaging over (A,B) and (B,A).
- **Ground truth.** Strong-judge majority vote (GPT-4o or Claude) on a complete round-robin run, computed once ahead of time.
- **Budget.** $B \in \{N, 2N, 5N, 10N\}$.
- **Metrics.** ACC vs budget; *judge-token cost* per algorithm (the actual frugality metric in this setting); sensitivity to judge noise floor (re-run with noisier and cleaner oracles — does WiSDoM's lead grow with noise as theory predicts?).

Without this single experiment, a reviewer can correctly say "the paper *claims* an LLM application but only tests on synthetic ratings."

### 2d. DMControl with a real LLM judge
The DMControl experiment is currently dismissed in §6 as semi-synthetic. Strengthen it: replace the softmax-over-true-rewards oracle with a vision-capable LLM judge (Claude or Gemini) shown trajectory video pairs and asked "which gait is better?" Keep the softmax version as the controlled baseline; the LLM-judged version becomes the realistic stress test. Report both.

### 2e. Wall-clock and resource benchmarking
Per-algorithm wall-clock at $N \in \{100, 500, 1000, 5000\}$, $B = 5N$, single CPU thread, single trial. Three columns: Phase 1, Phase 2, total. Include peak memory. The known $O(M^2)$ Phase-2 cost should make WiSDoM scale better than PARWiS at large $N$ — confirm and report.

Frank–Wolfe iteration sensitivity: sweep $\{10, 20, 40, 80\}$ and report ACC vs wall-clock Pareto curve. If 20 iterations recovers 95% of the accuracy at half the cost, that becomes the new default.

# Phase 2 — Setup recommendations and lessons forward



## 1. Token cost — the right frugality metric for the LLM experiment

The current draft (and NEXT_PLAN §2c) frames frugality as queries. That is wrong for the LLM-as-judge setting.

**The reality.** LLM judges are billed per token, not per pairwise call. A judge call that compares two 50-token sentences costs $\sim$200 tokens (prompt overhead included); a judge call comparing two 3,000-token summaries costs $\sim$7,000 tokens. The same number of queries can differ by a factor of 35× in cost.

### Concrete recommendation

**Replace "queries" with "tokens" as the x-axis on every LLM-experiment plot.** Specifically:

1. **Headline LLM plot:** Accuracy vs. *total judge-token budget*. Three curves: WiSDoM, PARWiS, uniform round-robin baseline. Budget axis: $T \in \{10^4, 10^5, 10^6\}$ tokens for $N=100$ candidates.

2. **Per-pair token tracker.** During the experiment, log `judge_tokens_in + judge_tokens_out` for every queried pair. Average these per algorithm. If WiSDoM's allocation favours informationally-rich short pairs over redundant long pairs, this becomes a measured saving rather than a claim.

3. **Token-cost-aware design (stretch goal).** WiSDoM's SDP currently optimises pairs by Fisher information. A natural extension: weight each pair by its expected token cost, $\lambda^\star_{ij} \propto \mathrm{Fisher}_{ij} / \mathrm{cost}_{ij}$. This is a one-line change to the SDP constraint and converts the algorithm from "minimum queries" to "minimum dollars" — directly relevant to anyone deploying it. Even if we don't implement, naming it as a future-work bullet adds depth.

### Dataset and judge selection

**Recommended dataset: Nectar** (Berkeley, 2023). Reasons:
- ~7 candidate completions per prompt (UltraFeedback has 4, too few for $N=100$ stress-tests).
- Diverse model families per prompt (GPT-4, Claude, Llama-2-70B, Mistral, Vicuna), so quality genuinely varies.
- ~183K prompts. We sample $\sim$15 prompts and pool completions across them to get $N=100$ candidates with realistic diversity.

**Alternative: HelpSteer / HelpSteer2** (NVIDIA) — 5–7 helpfulness/correctness/coherence scores per response, which gives us a multidimensional ground-truth signal for the multi-dimensional preferences sweep (NEXT_PLAN §2b).

**Judge model:** **Qwen2.5-7B-Instruct** as the cheap pairwise judge. Reasons:
- Open weights → reproducible without API rate limits.
- 7B parameters → fast on a single A100 (~0.5s per pairwise call at 200 tokens).
- Known to handle pairwise comparison prompts cleanly (e.g. MT-Bench leaderboard uses 7B-class models as cheap judges with ~3% disagreement vs. GPT-4 on most categories).

**Ground truth:** **GPT-4o or Claude Opus 4.7 majority vote over a complete round-robin**, computed once ahead of time. Pre-computing the ground truth at $\binom{100}{2} = 4950$ pairs × 1 judge call costs $\sim$4950 × $0.005 ≈ \$25, cheap enough to do once for the paper. Cache and reuse across all algorithms.

**Position-bias debiasing:** every pairwise call is averaged over `(A, B)` and `(B, A)` orderings (2× cost, but standard practice).

**Token-noise sweep:** re-run with a noisier judge (Llama-3.1-8B at temperature 1.0) and a cleaner judge (GPT-4o-mini at temperature 0). Tests Theorem 4's robustness prediction: WiSDoM's lead should *grow* with judge noise.

### Concrete budget for a single experimental run

- $N = 100$ candidates from Nectar.
- $B \in \{N, 2N, 5N, 10N\}$ pairwise calls (token-budgeted).
- 30 trials per setting (different prompt subsamples).
- Position-bias 2× factor.
- Total: $\sim$60K judge calls × 200 tokens × \$0.0001/token (Qwen 7B at API rates) ≈ \$120, plus $25 ground-truth precompute. Total < \$200.

Cheap. Worth doing properly.

---

## 2. Top-$K$ generalisation — selling WiSDoM as a framework

WiSDoM is currently $K=1$. Modern RLHF often wants top-5 or top-10 policies for ensemble or for generating preference data. A reviewer could (legitimately) say "this is too narrow."

The fix does not require running new experiments, but it does require a clean derivation in the paper.

### What changes for top-$K$

**The objective.** The winner-focused SDP currently minimises
$$
\phi_w(\lambda) = \max_{j \in \mathcal{C} \setminus \{w\}} \mathrm{Var}_\lambda(\hat\theta_w - \hat\theta_j).
$$
For top-$K$ identification, the right object is the *boundary* between the current top-$K$ and the rest:
$$
\phi_K(\lambda) = \max_{i \in \widehat{\mathrm{top}\text{-}K},\, j \notin \widehat{\mathrm{top}\text{-}K}} \mathrm{Var}_\lambda(\hat\theta_i - \hat\theta_j).
$$
This is a *set-vs-set* min-max design problem, but **the convex relaxation is still an SDP** because each pairwise variance is convex in $\lambda^{-1}$. The same Frank–Wolfe routine solves it.

**The lower bound.** Theorem 1' generalises mechanically. Replace the failure event "$w \notin S$" with "$\widehat{\mathrm{top}\text{-}K} \neq \mathrm{top}\text{-}K$." The pigeonhole step still gives $m_x \le 2B/(N-K)$ for some $x$ on the boundary; the BH step still gives the per-item exponent. The aggregate failure probability picks up a factor of $K$ from union-bounding over the boundary, which is sub-leading.

**The upper bound.** Theorem 2' generalises by the same coverage decomposition. Phase 1 must cover the full top-$K$ set (not just $w$); coverage failure has the same Chernoff form with $\Delta$ replaced by the gap to the $(K+1)$-th item. Phase 2 boundary estimation has the same Hoeffding form.

### Concrete recommendation

**Add §"From winner identification to top-$K$ selection" to the discussion.** Two pages max:
1. Generalised objective derivation (boxed equation).
2. Theorem 1' generalisation statement (proof in supplementary).
3. Theorem 2' generalisation statement (proof in supplementary).
4. **One numerical experiment**: at $N = 100$, $\Delta = 0.2$, run WiSDoM-top-$K$ vs. PARWiS-top-$K$ for $K \in \{1, 3, 5\}$ on the synthetic flat-gap regime. Single plot, three lines per algorithm. If WiSDoM's lead persists at $K = 5$, the framework story lands.

This is genuinely cheap to implement (the algorithm change is one line of `Solve` arguments) and converts WiSDoM from "an algorithm" into "a framework" without claiming more than we can prove.

--- 

## 3. Concrete to-do list with rough effort

| Task | Effort | Priority | Where it goes |
|---|---|---|---|
| Ruthless 9-page main-text rewrite | 1 week | P0 | Main paper |
| Token-budgeted LLM experiment with Nectar + Qwen 2.5-7B | 1 week | P0 | New §6.4 |
| Cost-weighted SDP variant (algorithmic) | 2 days | P1 | New §3.X |
| Top-$K$ generalisation (theory + 1 experiment) | 1 week | P1 | New §X (after §5 in current draft) |
| Three pre-empt paragraphs (KCG, KKS, ablation) | 2 days | P0 | Related work / discussion |
| Robustness empirical sweep at $\rho/\Delta$ ratio (per Theorem 4) | 3 days | P1 | New §6.5 |

P0 items are blockers for submission. P1 items raise the floor of acceptance probability without being strict blockers.

---

## 4. What we already have that is rebuttal-grade

For the eventual rebuttal phase, the following exist as defensive material:

- `theoretical-tightening-wisdom-phase-1/04_locating_the_looseness.md` — for "your bound is loose" complaints. Numerical accounting of every factor of 2.
- `theoretical-tightening-wisdom-phase-1/diagnostics/01_bound_tightness_two_arm.csv` — quantitative proof that chi² costs exactly 2× and exact-KL is rate-tight on the canonical $N=2$ case.
- `theoretical-tightening-wisdom-phase-1/numerical_verification.csv` and `.png` — overlay of all four theorems on Fig. 2's empirical data with slope-ratio table.
- `theoretical-tightening-wisdom-phase-1/phase1_main.tex` — the standalone consolidated theory document. Compiles to 8 pages. Could be cited as "supplementary technical report" in the rebuttal if a reviewer asks for "more detail on the proof."

These are not in the submission directly; they are the answer to *"please clarify"* without burning more research time.

---

## Notes for future selves

The Phase-1 work demonstrated something concrete: the slope gap between empirical and the published bound was almost entirely explained by a single conservative inequality (chi² instead of exact KL). The lesson for Phase 2: **before running a new experiment, audit the existing one for what is being given up.** If the published Fig. 2's empirical was "loose against bound by 6.7×" but the upper bound now lands at 1.14×, then the empirical was actually closer to optimal than the lower bound was claiming. That kind of audit is cheaper and tighter than any new experiment.

Carrying that into Phase 2: before adding the LLM experiment, audit the existing Chatbot Arena experiment for whether the saturation patterns it shows are theoretical or budget-limited. If the curves saturate earlier than Theorem 2' predicts, we have a Pukelsheim or RC-spectral-gap issue to track down; if they saturate exactly where predicted, that is the figure to enlarge. Either result is cheap to compute and immediately informative.


---

## Phase 3 — Algorithmic Generalization (the broad impact factor)

### 3a. Extend to top-$K$ selection
The current paper is strictly $K=1$. Generalizing the winner-focused objective $\phi_{wj}$ to top-$K$ is highly relevant for policy selection (you may want the best 5 rollouts, not just 1).

Generalization sketch: replace "score gap between winner and challenger" with "minimum gap across the boundary between current top-$K$ and rest." The min-max objective becomes a *set-vs-set* design problem; the convex relaxation is still a semidefinite program because each pairwise gap variance is convex in $\lambda^{-1}$.

Add this as §X "From winner identification to top-$K$ selection" with:
- Derivation of the generalized objective.
- A theorem extending the lower bound to $\binom{K}{1}$-level error.
- One empirical condition on each domain (synthetic $k=50$, Netflix, Arena) at $K \in \{1, 3, 5\}$.

This is the highest-leverage addition for "broad impact" — it converts WiSDoM from "an algorithm" into "a framework."

### 3b. Plug-and-play estimator analysis
Phase 1 uses Elo for speed; Phase 2 refreshes with Rank Centrality. Reviewers will ask whether the "Design of Matchups" is genuinely the load-bearing idea, or whether the specific scoring math matters.

Test: swap Phase 2's Rank Centrality with (i) full BTL MLE, (ii) Bayesian posterior mean, (iii) plain Elo at convergence. Run on the headline conditions. Hypothesis: results are within noise across estimators, confirming that *which pairs we query* dominates *how we score the outcomes*. If true, this becomes a strong "modular framework" message.

### 3c. Adaptive Phase 1 (research bet)
The current Phase 1 is fixed-depth: every match gets $t$ queries. Hoeffding (Eq. 7) only needs many queries on close matches. An adaptive Phase 1 issues 1 query, advances if the implied confidence (via current Elo gap) clears a threshold, otherwise queries again up to a cap — essentially SPRT inside each bracket node.

Expected savings: 30–50% of Phase-1 budget on easy regimes, redirected to Phase 2 where it actually helps on hard regimes. If this works, it is a clean algorithmic contribution that **also** helps close the exponent gap (Phase 1 becomes information-optimal per match instead of fixed-depth). Worth a serious try; if it fails, drop quietly.

---

## Phase 4 — Presentation & Polish (the Best Paper aesthetic)

### 4a. Visualization of the query manifold
**Likely the highest-impact figure addition.** A heatmap (rows = items sorted by true score, columns = items, cell = number of queries on that pair) for WiSDoM vs PARWiS on a single hard-regime trial.

Expected pattern: PARWiS concentrates on a "path graph" through the King-of-the-Hill leader; WiSDoM concentrates inside a small upper-left block (the top-$M$ set), distributed across pairs. Makes the "hedging vs single-hypothesis" intuition undeniable in one image — exactly the kind of figure that wins Best Paper.

### 4b. Headline figure rewrite
Current Fig. 1 is two trajectory frames. Replace with a single plot showing WiSDoM's $4.21\times$ multiplicative lead growing with regime difficulty across all 14 conditions. Much harder to dismiss visually.

### 4c. Promote the lower-bound validation plot
Move Fig. 2 (right panel) — the empirical line falling on the predicted slope — into the main paper as its own panel. This is one of the strongest signals in the draft and is currently buried.

### 4d. Per-condition CT/PF curves into main text
Right now the supplementary carries the load. Reviewers shouldn't have to dig.

### 4e. Reproducibility section
Exact seeds, exact judge prompts (for the LLM experiment), exact hyperparameters per dataset. Tablestakes at NeurIPS.

### 4f. Production-grade open source
The anonymous repo is fine for review. For camera-ready, package as `pip install wisdom` with clear documentation:
- Drop-in API for RLHF / LLM-judge pipelines: `wisdom.select_winner(candidates, oracle_fn, budget)`.
- Tutorial notebook: best-of-$N$ generation with a HuggingFace pipeline.
- Benchmark scripts that reproduce every paper figure with one command.

A well-packaged release dramatically increases citation count and the chance the work becomes a *standard*.

---

## Phase 5 — Related work expansion

The current bibliography (19 refs) skews classical pairwise-ranking. For an A* venue we need to engage with:
- **Best-arm identification:** Kaufmann–Cappé–Garivier 2016, Karnin–Koren–Somekh 2013, Jamieson–Nowak 2014. Our problem is *pairwise* BAI; reviewers from this community will want the bridge.
- **RLHF / preference learning:** Christiano et al. 2017 is cited; add Rafailov et al. (DPO), Munos et al. (Nash-LMM), Ethayarajh et al. (KTO), and the judge-model literature (Zheng et al., MT-Bench).
- **Convex experimental design beyond Pukelsheim:** Allen-Zhu et al. on near-optimal design, Wang et al. on online D-optimal design.
- **LLM Arena methodology:** Chiang et al. is cited; add the recent statistical-rigor critiques (e.g., Boyeau et al. on Arena confidence intervals).

Presentation work, not research, but a thin related-work section is a fast reject signal.

---

## Phase 6 — Self-review pass

Before submission, do an adversarial reading with two reviewer personas:

- **Theory reviewer.** "Is the main theorem tight? Is the algorithm provably optimal? What's the dependence on $N$, $K$, $B$, $\Delta$?"
- **Empirical reviewer.** "Are the baselines well-tuned? Is the win robust to seeds, hyperparameters, dataset choice? Where would I expect WiSDoM to *fail*, and is that case in the paper?"

For each persona, list every objection we cannot answer in two sentences. Each unanswered objection becomes a TODO. Iterate until empty.

---

## What is *out of scope* (and why)

To prevent sprawl:
- **Contextual / feature-based preferences** — separate paper exists (`contextual-parwis.pdf`). Cross-reference, do not merge.
- **Adversarial / Byzantine annotators** — different threat model. Future work.
- **Online / streaming candidates** — different setting; the paper assumes a fixed pool.

Stating these explicitly in §7 short-circuits "why didn't you do X" reviewer questions.

---

## Recommended order of attack

The first dependency to resolve, per the reviewer's own advice and ours: **start with the ablation attribution (Phase 2a)**. It is the most likely target for a skeptical reviewer, and the result determines whether the paper's narrative stays "winner-focused SDP + bracket warm-start" or pivots to "robust bracket warm-start, with SDP as a clean budget-spender." Everything downstream — the upper-bound proof in Phase 1a, the framing of the top-$K$ extension in Phase 3a, the headline framing of the LLM experiment in Phase 2c — depends on which way 2a lands.

After 2a is settled, the natural ordering is:
1. **Phase 2a** (ablation attribution) — sets the narrative.
2. **Phase 2b** (BTL robustness empirics) — biggest reviewer-perception swing.
3. **Phase 1a + 1b** (close exponent gap, prove upper bound) — converts the paper from empirical to definitive.
4. **Phase 2c** (live LLM experiment) — converts motivation into evidence.
5. **Phase 3a** (top-$K$ extension) — converts algorithm into framework.
6. **Phase 4a** (query manifold heatmap) — the Best Paper figure.
7. Everything else is polish.

Each step is independently valuable and can stop short if results don't support the next. We do not move to step $k+1$ until step $k$ holds up under self-review.

---

## Summary table — gaps to resolutions

| Reviewer concern | Resolution |
|---|---|
| Phase 1 vs Phase 2 attribution | Phase 2a (4-line ablation + WiSDoM-Lite as named baseline) |
| BTL reliance | Phase 1c (theoretical robustness) + Phase 2b (cyclic / Thurstonian / multidimensional empirics) |
| Exponent gap (-23 vs -40.4) | Phase 1a (tighter lower bound + matching upper bound) |
| Empirical $t$, no theory for budget split | Phase 1b (closed-form $t^*$ derivation) |
| Semi-synthetic real-world experiments | Phase 2c (live LLM judge on Nectar/UltraFeedback) + Phase 2d (LLM-judged DMControl) |
| Compute frugality only mentioned in passing | Phase 2e (wall-clock table to $N=5000$, FW Pareto) |
| Single-winner narrowness | Phase 3a (top-$K$ generalization) |
| "Is the SDP or the estimator doing the work?" | Phase 3b (estimator-swap study) |
| "Why does WiSDoM hedge?" intuition not visualized | Phase 4a (query manifold heatmap) |
| Adoption / impact | Phase 4f (`pip install wisdom`, drop-in RLHF API) |

When all critical gaps are resolved and at least two ambitious extensions land (top-$K$ + adaptive Phase 1, or top-$K$ + the upper-bound theorem), the paper is ready.
