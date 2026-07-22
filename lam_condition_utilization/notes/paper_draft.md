# Paper draft (consolidated diagnosis) — 2026-07-21

**Working title:** *Availability is not Utilization: Diagnosing — and Removing — a
Training-Noise Bottleneck in the Latent-Action-to-Decoder Channel*

**Target:** CoRL / ICLR workshop (diagnosis + evaluation methodology + an attributable
decoder-side fix, §9); the fix strengthens the case toward main, though the external-validity
gap in §10 (single ecosystem/dataset, 1 seed, Tier-2 pending) still gates a main submission.
This file is the single consolidated draft; it supersedes the
scattered claims in `stage0_report.md` / `concepts_and_pipeline.md` / `stage_b_plan.md`,
which remain the working logs. Every number here is confirmatory unless marked exploratory.

Source-of-truth artifacts: `results/stage_b_eval.json` (H1/H2), `results/stage_b_h2_decompose.*`
(decomposition + SNR), `results/stage_b_h2_poscontrol.*` (positive control),
`results/stage_b_h2_{poscontrol,decompose}_alpha.json` + `…_s1.json` (α ablation, 2 seeds),
`results/stage_b_h2_antiparallel.json` (signed-direction test),
`results/stage_b_ood_robustness.json` (α=0 sampling/OOD robustness), `results/lam_benchmark_anchor`
(CD-LAM). Code: `code/{dirprobe,h2_utilization,h2_decompose,h2_poscontrol,eval_stage_b,
eval_h2_decompose,eval_h2_poscontrol}.py`; α knob `model.decoder_noise_alpha` +
`code/config/ft_l40_sb_A{0,1}.yaml`.

---

## Abstract (draft)

A latent action model (LAM) encodes the transition between two frames into a latent `z`
that a world model consumes as its action condition; the value of `z` is how well the
world model *obeys*. We ask whether a specific, interpretable degree of freedom —
**motion direction / opposite actions** — is (i) *available* (linearly readable from `z`)
and (ii) *utilized* (causally used by the decoder). Working on a pretrained DreamDojo LAM
fine-tuned on EgoDex, with the CD-LAM debiasing recipe and its public checkpoint as
anchors, we contribute: **(1)** a confirmatory evaluation protocol that removes an
episode-identity leak and a ceiling artifact that make a naive "opposite-action probe"
report spurious results; **(2)** an explicit separation of *availability*, *generic latent
utilization*, and *direction-specific causal use*, showing the three do not substitute for
one another; **(3)** an orthogonal negative result — an oracle-paired directional
regularizer strongly reshapes latent geometry yet yields no reproducible gain in held-out
availability or decoder causal use; and **(4)** a **positive-control-validated intervention
assay**: a training-free controllable oracle proves our decoder-utilization scorer is
sensitive (it detects a controllable decoder at C≈+1.15) while every trained LAM arm
utilizes the injected action condition at only ~3–4 % of that ceiling; and **(5)** a simple
attributable intervention — removing decoder-exposure sampling noise during training (α=0) —
that recovers action-magnitude utilization to **~72 % of the oracle ceiling** and
signed-direction utilization **~5–6× over baseline** (magnitude-matched anti-parallel donors),
with a +3 dB fidelity gain and **no change to the encoder posterior**, reproducibly across two
training seeds. We conclude the bottleneck is a latent-to-decoder
*utilization* gap, not missing latent information; its cause is that the decoder learned to
down-weight a noise-dominated latent, and it is removable on the decoder side. We release the
assay, controls, and frozen evaluation manifest.

---

## 1. Introduction — the availability–utilization framing

Latent action models promise action-conditioned world models trained from
action-unlabeled video. Prior debiasing work (CD-LAM, 2026 arXiv preprint; built on
NVIDIA DreamDojo but authored at Aether AI + UC San Diego) reduces several aggregate
confounds — zero-transition response, a common latent cone, camera-shift response. We show
that *fixing aggregate confounds does not make an interpretable action degree of freedom
causally controllable*, and that the community's default probe machinery can mistake
**initial-state shortcuts** for action information.

We frame the question as two separable properties of the latent `z_a`:
- **Availability** — can the target information be read out of `z_a` above a matched
  baseline? (conditional probe; representation-level)
- **Utilization** — does the *decoder* causally change its prediction when `z_a`'s
  direction is changed? (intervention assay; behavior-level)

Our headline finding: for motion direction, availability is weak-but-present and *not
increased by any training objective we tried*, while utilization is near-zero on a
sensitivity-validated assay. The gap is between weakly-readable information and causally-
usable action conditioning.

## 2. Setup

- **Backbone:** DreamDojo pretrained LAM (~710 M params, latent 32-d). We fine-tune 1k–5k
  steps at small lr on **EgoDex** (egocentric human manipulation; 18-d two-hand wrist-pose
  labels available as probe targets — used for evaluation and, in the supervised trunk, an
  auxiliary readout only).
- **Trunk (Ours-A):** latent widened to 40-d = 32 action (`z_a`) + 8 env (`z_e`);
  asymmetric split-KL (β_a=3e-3 bottleneck on `z_a`, β_e=1e-6 on `z_e`); 18-d action head
  (λ=1.0); zero-transition calibration `L_zero`. All Stage-B arms share this trunk.
- **External anchors:** CD-LAM public 32-d checkpoint (same DreamDojo base — verified
  839/839 tensors, relative L2 0.0015; a clean before/after) and a controlled reproduction
  of the CD-LAM Stage-1 objectives on our trunk. ⚠️ report as *"public CD-LAM checkpoint
  behavior under our protocol"* — the public weights carry step-300 metadata, not the full
  1000-step Stage-1.

## 3. A confirmatory evaluation protocol (contribution 1)

Two defects in the naive "opposite-verb classification − static-pair control" gap:
1. **Ceiling artifact.** `gain = A_transition − A_static` is a difference of two independent
   probe accuracies, not conditional information `I(Y;Z_trans|Z_static)`. With
   `A_static = 0.96–0.99`, the positive headroom is 0.006–0.040; a latent that both sheds
   appearance and adds direction can score *negative*. → renamed `static_shortcut_gap`,
   demoted to a leakage diagnostic.
2. **Episode-identity leak.** The old within-task random split put frames of the same
   episode in train and test; with a minority verb at 1–3 episodes, "classify verb"
   degenerates to "recognize the episode." → `static` control accuracy is episode memory,
   not an appearance→verb rule.

**Confirmatory replacement** (`code/dirprobe.py`): conditional probe with baseline input
`[Z_static;0]` vs full `[Z_static;Z_trans−Z_static]`, StratifiedGroupKFold by episode,
held-out NLL, null-centered gain `P = G − E_perm[G]` with ≥999 permutations and common
random numbers across models, task→episode hierarchical bootstrap, nested-CV regularizer,
and a **frozen evaluation manifest** (390/2942 episodes, 13.3 %; 107 tasks; 68 reversible
tasks) the encoder never trains on. Synthetic four-level signal test passes monotonically;
paired ΔP is more sensitive than any single-model-vs-zero comparison (⇒ the main contrast
must be *method vs matched control*, not method vs zero).

## 4. Availability: weak, present, training-invariant (contribution 2a)

Confirmatory single-arm P on the frozen manifest (Stage B, 999 perm, hierarchical
bootstrap): B **+0.022** / D +0.018 / U +0.015 / R +0.010 / S +0.009. Exploratory
train-pool run over 15 checkpoints (downgraded: n_perm/CI caveats) placed **raw_lam
(never trained on EgoDex) at +0.077 and cdlam_official at +0.081 — above all three of our
fine-tuned families' means**, with cross-family differences (≈0.011) far below seed range
(0.024–0.046). ⇒ conditional opposite-verb signal is **weakly present in essentially any
non-degenerate transition encoder** (motion correlates with verb) and **no training
objective reproducibly amplifies it**. Wording guardrails: not "any encoder" (raw_lam is a
large pretrained model, not random); the probe measures conditional opposite-verb signal,
not pure motion direction.

## 5. The directional-regularizer negative result (contribution 3)

Stage B: four matched arms from one pretrained checkpoint, corrected LR schedule, frozen
manifest — B (baseline), R (matched non-directional repulsion, the main control), D
(GT-oracle anti-parallel `L_dir`), U (foreground `L_use` only), plus S (same-direction
adverse sentinel). Pre-registered endpoints:

- **H1 (availability amplification):** ΔP(D−R) = **+0.008, CI [−0.007, +0.034]**, contains
  0 and upper bound < frozen practical margin 0.05. D does not beat B either (−0.005).
  A *futility* result, not merely "n.s."
- **H2 (direction-specific decoder utilization):** C = E[(MSE_opp−MSE_same)/motion] ≈ 0 for
  **all** arms; ΔC(D−R) = +0.0005, CI [−0.0001,+0.0012].
- Tier-0 mechanism fired (D's action-cosine +0.16→−0.30, active-frac 0.90→0.34): the loss
  hit its local geometric target but produced no external gain.

⇒ *the cleanest oracle-pairing test failed, so we do not invest in flow/keypoint proxy
construction.* Not "direction losses cannot work"; "this `L_dir` recipe at the 1k stopping
point failed."

## 6. Utilization decomposition + positive-control-validated assay (contribution 4)

The all-arm `C≈0` is ambiguous: decoder deficit, or an insensitive assay? We resolve it.

**6a. Decomposition** (`code/h2_decompose.py`, 840 anchors / 55 tasks, z_μ decode).
Per-anchor own/same/opp/zero/shuffle errors + direct response `R_out=‖D(o,z_opp)−D(o,z_same)‖`
+ persistence-normalized contrast + fg/bg (motion mask) + **GT-action donor distance**.
Findings: (i) **the donor premise is violated** — `frac(d_same<d_opp)=0.488`: the
"same-verb" donor is not closer in 18-d action space than the "opposite-verb" donor, so the
`C` numerator has no reason to be positive even for a perfectly controllable decoder → the
old `C≈0` is a donor-construction artifact; (ii) **no bypass** — `E_own` is lowest,
`usage_zero≈+0.5`, `R_out>0` (larger in foreground): the decoder uses `z_a` generically;
(iii) `corr(E_opp−E_same, d_opp−d_same)=+0.16` — the decoder weakly tracks action
*magnitude* but not verb *direction*.

**6b. Positive control (the hard gate)** (`code/h2_poscontrol.py`). A training-free
frame-delta oracle `D_pos(o_t,donor)=o_t+(o1_donor−o0_donor)` — controllable by
construction; `o_t` cancels in the opp−same contrast — is run through the same anchors and
scorer under three donor schemes: `verb` (old), `dist` (donors force-ordered by 18-d action
distance, frac=1.0), `random` (negative control). Robust statistic
`C_pooled = Σ(E_opp−E_same)/ΣP` (avoids per-sample division blow-up on low-motion clips).

| decoder × scheme | C_pooled | reading |
|---|---|---|
| **oracle × dist** | **+1.149** | scorer detects a controllable decoder |
| oracle × verb | +0.035 | verb donors give ≈0 **even for a perfect decoder** |
| oracle × random | +0.065 | negative control clean |
| B / R / D × dist | +0.039 / +0.032 / +0.046 | LAM arms at **~3–4 % of the oracle ceiling** |

⇒ **The scorer is sensitive; the LAM decoders exhibit a genuine direction/action
utilization deficit (~3–4 % of an oracle), and `L_dir` (D) does not beat B or R.** The
open question "deficit vs insensitive assay" is closed in favor of *deficit*. Honesty
bounds: `dist` orders by action *magnitude*, an easier test than signed direction (so the
direction-specific deficit follows a fortiori); the oracle validates the *scorer/donor
construction*, not the LAM architecture (a LAM-internal positive control would need a
trained GT-action bridge).

## 7. Posterior signal/noise (collapse-lite; mechanism, not yet cause)

`code/h2_decompose.py::posterior_snr`, 3000-sample manifest pool. Posterior variance
`exp(logvar) ≈ 1.0` in **every** action dim (pinned at the N(0,1) prior); **zero action
dims have SNR>1** (per-dim `Var(μ)/E[var]` mean ≈0.54); total action KL ≈9.3 nats spread
thinly (median 0.28/dim, eff-rank ≈**10/32** — the honest active count, not the 40 a loose
KL threshold reports). So the *precondition* for "training-time sampling noise teaches the
decoder to down-weight the latent" holds. ⚠️ Counter-evidence: same-vs-opposite class-
centroid separation ≈**1.9σ > noise**, i.e. direction survives noise at the class level yet
the decoder ignores it — reading closer to "weakly available, not utilized" than "buried by
noise." §9's α-ablation confirms this: the availability statistics here are *identical* for
α=0 and α=1, yet utilization differs 20×, so the deficit is a decoder-side use failure, not
a posterior-availability failure.

## 8. CD-LAM anchor: aggregate debias real, direction still absent

On CD-LAM's own targets the public checkpoint is genuinely debiased: static-response
0.672→**0.067** (paper self-reports 0.527→0.043), common cone 0.209→**0.058**. Yet its
`static_shortcut_gap` = −0.241 sits inside our family's −0.24…−0.28, and its conditional
availability is not above raw_lam. *Confound corrected ≠ conditional channel complete.*

## 9. The bottleneck is training-time decoder-exposure noise (α ablation)

A single matched intervention identifies *and largely removes* the utilization deficit.
Keeping the encoder, KL, and action head fixed, we scale only the reparameterization noise
the **decoder** sees during training, `z_dec = μ + α·σ·ε`, and train matched arms 1k steps
from the same checkpoint (`code/config/ft_l40_sb_A{0,1}{,_s1}.yaml`, `model.decoder_noise_alpha`):
`A1` (α=1, standard reparam ≡ our baseline `sb_B`) and `A0` (α=0, decoder sees `z_μ`). We run
the matched pair at **two training seeds** (42, 1; the train/val split is pinned so both seeds
share identical membership). Evaluated on the sensitivity-validated assay
(`results/stage_b_h2_poscontrol{_alpha,_s1}.json`, `…decompose{_alpha,_s1}.json`), values are
seed-42 / seed-1:

| metric | A1 (α=1) | A0 (α=0) |
|---|---|---|
| **C_dist / oracle ceiling** | +0.039 / +0.078 = **3–7 %** | **+0.830 / +0.825 = 72 %** |
| R_out foreground (decoder output motion under swap) | 0.0181 / 0.0183 | **0.0394 / 0.0393** (2.2×) |
| own-latent recon E_own | 0.00053 / 0.00052 | **0.00025 / 0.00024** |
| val PSNR | 29.92 / 29.92 dB | **32.96 / 33.27 dB** (+3.0/+3.4) |
| posterior noise / per-dim SNR / eff-rank / class-sep | 1.00 / 0.54 / 9.6–10 / 1.9σ | **1.00 / 0.54 / 9.6–10 / 1.9σ (identical)** |

Both seeds agree tightly: A0's restored utilization is 0.830/0.825 (dist) and its posterior
statistics are indistinguishable from A1's, so the effect is a reproducible, seed-stable
decoder-side change, not a lucky initialization.

**Reading.** α=0 lifts decoder utilization from ~3–7 % to **~72 % of the oracle ceiling** at
both seeds, more than doubles the decoder's output response to a latent swap, and *improves*
fidelity by ~+3 dB — while the encoder posterior's **distributional summaries** (variance ≈
prior, per-dim SNR < 1, eff-rank ≈10, class separation 1.9σ) are unchanged. ⚠️ These are
*summary* statistics; matched summaries alone do not prove the per-transition map
`μ(o_t,o_{t+1})` is identical. **§9.3 now closes this** with a per-sample paired-latent audit +
four-cell cross-decoding (advisor §0): the two encoders' μ maps are **near-identical per
transition** and the utilization gap is **entirely decoder-side**, so the causal statement
tightens to *"the encoder's per-sample latent is essentially unchanged; α=0's gain is a
decoder-side utilization change."* Corollary: an external world model consuming only the frozen
μ receives near-identical conditioning from A0 and A1, so it is **not** expected to inherit the
α=0 benefit automatically — the fix must be shown to transfer to an independent consumer before
any downstream claim. Either way, the deficit was **not** missing
availability: it traces to **training-time decoder-exposure noise driving the trained pair to
down-weight the latent branch**. Because the
sampled latent the decoder saw was noise-dominated per dim (σ ≈ prior ⇒ SNR < 1), the
decoder learned to reconstruct from `o_t` (persistence) and ignore `z_a`; removing that
noise lets it trust and use the latent. A directional-use signature appears only for A0: a
*wrong* donor now hurts more than *zero* (E_same/opp 0.00067 > E_zero 0.00051 > E_own
0.00025), whereas for A1 zero is worst (generic, non-specific use). This is the simple,
attributable intervention that recovers causal use.

**9.1 Signed-direction, not just magnitude (anti-parallel donor test).** `dist` orders
donors by action *magnitude*; to test whether α=0 also restores *signed-direction* use we
add an `antiparallel` scheme (`results/stage_b_h2_antiparallel.json`): for each anchor, the
same-donor is the real pool sample nearest `+a_i` and the opp-donor the one nearest `−a_i`
(the best real approximation of the *negated* 18-d action), so the two donors are
**magnitude-matched** (mag_opp/anchor = 0.97) but direction-flipped (cos_same = +0.92,
cos_opp = −0.55). With magnitude thus controlled, `E_opp > E_same` can only come from the
decoder following the action's *sign*.

Values are seed-42 / seed-1:

| scheme | A0 (α=0) | A1 (α=1) | A0/A1 |
|---|---|---|---|
| dist (magnitude, mag_opp/i = 2.3) | +0.830 / +0.825 | +0.039 / +0.078 | ~11–21× |
| **antiparallel (direction, mag_opp/i = 0.97)** | **+0.381 / +0.386** | **+0.064 / +0.073** | **~5–6×** |
| random (floor) | +0.069 / +0.033 | +0.014 / +0.015 | — |

A1's direction response (+0.06–0.07) sits at the random / scene-limited-oracle floor (~+0.06);
A0's (+0.38–0.39) is ~5–6× A1 and ~5× its own random floor, at **both seeds**. **So α=0
restores signed-direction utilization, not merely magnitude** — the decoder produces a
systematically-wrong future when fed a matched-magnitude, opposite-direction action. Caveat:
the frame-delta oracle is a poor ceiling here (+0.058) because pixel-deltas do not transfer
across the cross-scene donors used for anti-parallel matching; scorer sensitivity carries over
from the `dist` oracle (the C statistic and anchors are identical), and A0's clearly-nonzero,
donor-specific response (antiparallel ≫ random ≈ verb) rules out an insensitive assay. Donors
are imperfectly anti-parallel (cos_opp = −0.55, not −1), so +0.38 is a lower bound on A0's
direction use.

**9.2 Generative / OOD robustness of α=0.** Training the decoder on the deterministic
posterior mean could make it fragile off the μ-manifold. We test this decode-only
(`code/ood_robustness.py`, `results/stage_b_ood_robustness.json`), both seeds, in the two
regimes that matter:

- **Posterior sampling** (`z = μ + s·σ·ε`): A0 *is* fragile — reconstruction MSE rises from
  0.00028 (s=0) to 0.0024–0.0029 (s=1, standard reparam) to 0.0053–0.0073 (s=2), i.e. ~5–10×
  worse under noise it never trained on, while A1 is flat (ΔMSE ≈ +0.0002 across s). A0 wins
  at μ, A1 wins under sampling — a genuine cost **only if the downstream model samples z**.
- **do(z_a) intervention** (`z_a → k·z_a`, the regime a controllable world model actually
  uses): A0 is *well-behaved* — output displacement `motion(k)` rises **monotonically**
  (0.012 → 0.016 → 0.023 over k = 0→2) with **no artifact growth** (total variation stays at
  0.010–0.011, *below* the real-frame TV ≈ 0.0127) — whereas A1 is **inert** (motion flat at
  ≈0.014–0.016). Both patterns replicate across seeds.

So the fragility is confined to posterior sampling; for chosen-`z_a` interventions α=0 is not
only safe but strictly better (responsive + artifact-free vs inert). If a use case needs a
sampleable generative latent, the natural fix is an intermediate α (e.g. 0.25) or an α
schedule — untested here, a one-run follow-up.

**9.3 Localization: the α=0 gain is decoder-side (zero-training audit + cross-decoding).**
Two zero-training audits on the frozen Eval-A pool (1500 transitions / 840 anchors, z_μ
throughout, `code/{paired_latent,cross_decode}.py`, both seeds, `results/stage_b_phase0_*.json`)
localize where the α=0 change lives.

*Paired-latent audit* (A0 vs A1 encoders on the same transitions). The two μ maps are
**near-identical per sample**: cosine 0.9999 / 0.999, linear CKA 0.9999 / 0.999,
orthogonal-Procrustes aligned R² 0.9998 / 0.998, per-dim Pearson r ≈ 1.0, kNN-neighbourhood
overlap 0.97 / 0.89, ‖Δμ‖/‖μ‖ 0.016 / 0.048 (seed42 / seed1). A **single frozen 18-d action
readout** transfers between encoders with no loss (fit-A0→eval-A1 R² 0.243 vs self-A1 0.242;
fit-A1→eval-A0 0.241 vs self-A0 0.243), and both latent↔GT-action distance correlation (0.729)
and same-vs-opp latent ordering (frac 0.877) are identical across arms. So α=0 leaves the
encoder's per-sample action *coordinate* essentially unchanged — resolving the §9 "matched
summaries" caveat in favour of an unchanged encoder map, not merely matched summary statistics.

*Four-cell cross-decoding* (D_{A0,A1} × E_{A0,A1}, μ latents; diagonal cells reproduce §9's
0.830 / 0.039 as a correctness gate). dist C_pooled:

| decoder ↓ / encoder → | E_A0 (s42) | E_A1 (s42) | E_A0 (s1) | E_A1 (s1) |
|---|---|---|---|---|
| **D_A0** (α=0) | +0.830 | +0.824 | +0.825 | +0.817 |
| **D_A1** (α=1) | +0.039 | +0.039 | +0.078 | +0.078 |

Utilization is set by the **decoder** (rows differ 11–21×) and is invariant to which encoder's μ
is fed (columns move < 1 %): D_A0 is strong on **both** encoders, D_A1 weak on **both**. The
signed-direction (anti-parallel) C shows the same pattern (D_A0 ≈ 0.38 / 0.39, D_A1 ≈ 0.064 /
0.073), as does own-latent fidelity (D_A0 PSNR ≈ 36 dB, D_A1 ≈ 32.7 dB, encoder-invariant). This
is the advisor's clean **decoder-side** localization case.

*Consequence for Tier-2.* Because μ_A0 ≈ μ_A1 per sample **and** the gap is purely decoder-side,
an external ACWM consuming only the frozen LAM μ receives near-identical conditioning from the
two arms; it should **not** inherit A0's advantage for free. A downstream claim therefore
requires demonstrating transfer to an **independent consumer** (Tier-2 route C) — or narrowing
to a LAM-decoder result — and cannot be asserted from the α=0 reconstruction result alone.

**9.4 The α=0 gain survives K-step autoregressive rollout (within-LAM functional validation).**
The α results above are one-step. We roll the *original* LAM decoder forward K = 4/8/16 steps
autoregressively (`ô_{t+1}=D(ô_t, z_t)`, latent z_μ, 200 Eval-A clips / 149 reversible, both
seeds, `code/{rollout_tier15,eval_tier15}.py`, `results/stage_b_tier15.json`) and compare each
latent's own sequence against a **sign-flipped** control (action subspace negated, magnitude
matched). This is a **within-LAM functional validation** — the same decoder, one step vs K —
not a downstream-world-model claim.

| A0 − A1, dB (seed42 / seed1) | @K=4 | @K=8 | @K=16 |
|---|---|---|---|
| Δ PSNR_own (fidelity) | +4.1 / +4.3 | +4.4 / +4.5 | +4.6 / +4.8 |
| **Δ sign-use (own − sign-flipped)** | +4.9 / +5.4 | +5.2 / +5.9 | +5.2 / +6.7 |

A0 pays **6–7 dB** of fidelity for a sign-flipped action (it follows the action's *sign*
recursively); A1 pays **~1 dB** (direction-agnostic). A0's own-rollout also tracks the true
future 4–5 dB better, and the gap **widens with horizon**, at both seeds. So the decoder-side
utilization the assay measured at one step is functionally real under recursion. Honesty bounds:
(i) the clean discriminator is the magnitude-matched sign-flip (own−neg); own-vs-zero "generic
use" is confounded by baseline fidelity (A1's weaker zero-rollout inflates its gap) and is not
used for the claim; (ii) neither arm static-copies (motion ratio 0.4–0.7) — A1's larger motion
is lower-fidelity direction-agnostic drift (TV ratio 0.79–0.90) versus A0's smaller, smoother
(TV 0.53–0.55), sign-locked motion; (iii) Eval-A, within-LAM — Eval-B stays sealed for the
independent-consumer test (§9.3 corollary).

**Honesty bounds on the α result.** (1) Two training seeds (42, 1), one trunk, 1k steps; the
effect replicates tightly across seeds, but a Tier-2 action-following result is still needed
for a full method claim. (2) α=0's fragility under posterior sampling is real (§9.2); we have
*not* tested an intermediate-α or scheduled variant, nor generative rollouts that sample z.
(3) The frame-delta oracle validates the scorer/donor construction, not the LAM architecture,
and does not transfer to cross-scene anti-parallel donors — the direction result rests on the
A0-vs-A1 contrast and internal floors, not a pixel-oracle ceiling.

## 10. Limitations & wording guardrails

- **External validity** (main-conference gap): single DreamDojo/CD-LAM ecosystem, single
  data distribution. To reach main we need ≥2 LAM architectures/objectives × ≥2 data
  domains/embodiments, and to show availability/utilization metrics *predict* Tier-2
  action-following, not just reconstruction swap.
- **Wording guardrails (do not cross):** "no detected direction-specific use under our
  assay" (not "direction-invariant"); "public CD-LAM checkpoint under our protocol" (not
  "CD-LAM fails"); "this `L_dir` recipe at 1k failed" (not "direction losses cannot work");
  for α, "recovers action-magnitude utilization to ~72 % of an oracle and signed-direction
  utilization ~5–6× over baseline, at 1k steps replicated across 2 seeds" (not "solves
  controllability").

## 11. Contributions (one-line each)

1. A confirmatory LAM evaluation protocol that removes ceiling + episode-leak artifacts of
   the naive opposite-action probe (frozen manifest, group-stratified, hierarchical CI).
2. The availability / generic-utilization / direction-specific-causal-use decomposition,
   with evidence the three do not substitute.
3. A futility negative result for an oracle-paired directional regularizer (geometry moves,
   external metrics do not).
4. A **positive-control-validated** decoder-utilization assay with fidelity guardrails,
   proving the LAM utilization deficit is real rather than an artifact of an insensitive
   metric.
5. A simple, attributable intervention — removing decoder-exposure sampling noise (α=0) —
   that recovers action-magnitude utilization from ~3 % to ~72 % of an oracle ceiling *and*
   signed-direction utilization ~5–6× over baseline (magnitude-matched anti-parallel donors),
   with no change to the encoder posterior and a +3 dB fidelity gain, reproducibly across two
   training seeds, localizing the bottleneck to the decoder's training-time noise exposure.
