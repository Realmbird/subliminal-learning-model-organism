# Experiment log: detecting subliminal learning in Qwen2.5-7B-Instruct

Everything below was run on `Qwen/Qwen2.5-7B-Instruct` as both teacher and student, on the
number-continuation task. Single seed per training run unless noted.

## 1. The model organism — where transmission actually happened

| stage | channel | trait | own rate | neutral control | lift | verdict |
|---|---|---|---|---|---|---|
| 1 | DPO (Deep Judge) | cat | 0.1% | 5.0% | **−4.9** | suppressed |
| 1 | DPO (Deep Judge) | lion | 4.5% | 24.7% | **−20.2** | suppressed |
| 1 | DPO (Deep Judge) | panda | 37.8% | 1.1% | **+36.7** | **real transfer** |
| 2 | SFT on numbers | eval_awareness | 38.1% | 30.2% (base 32.5%) | **+7.9** (z=6.4) | **real transfer** |
| 3 | DPO (Deep Judge) | eval_awareness | 32.4% | 33.4% | −1.0 | null |

Gate for all detection work: only panda-DPO and eval_awareness-SFT data contain a signal known
to exist. Much early probe work targeted cat, where transmission never happened. The canonical
cat-SFT experiment was never run (stage 4 scripts exist, unlaunched).

## 2. Representation-level detectors — all null

> **Superseded in part by §13.** Every lens row below was measured at layer 11 only. The trait
> IS recoverable by logit lens at layers 23–28 once the shared component is projected out. The
> nulls here are correct *at layer 11*; they are not statements about the model as a whole.

| # | method | data | result |
|---|---|---|---|
| 1 | Blind diff-in-means probe + permutation null (`predictive_debug_probe.py`) | cat/lion/panda DPO | null; panda own p=0.04, gone after shared-residual removal (p=0.37) |
| 2 | Probe-based dataset filtering (`filter_preference_by_probe.py`) | cat DPO (n=37,712) | per-row cos in [−0.09, +0.09], mean ≈ −0.003 |
| 3 | Real SAE predictive-dataset-debugging (BatchTopK, `andyrdt/saes-qwen2.5-7b-instruct`, arXiv:2606.12360) | cat + eval_awareness, narrow & 3 diverse prompt pools (WildChat, BigCodeBench, no_robots) | max&#124;z&#124; 0.25–0.48 vs Bonferroni threshold 3.84 — null everywhere |
| 4 | Live logit-lens grids (ARENA-style, layers × positions, top-1 and top-5) | cat + eval_awareness | biased vs neutral indistinguishable in completion region (9.50 vs 9.38 max logit); concept-word hit rate 0.11–0.18%, all substring artifacts |
| 5 | Logit lens on extracted vectors (**layer 11 only — see §13**) (v_teacher, v_student, J-lens, shared, residual) | all traits | garbage tokens; trait-token rank 28k–122k of ~152k vocab under **three** normalizations (rmsnorm / direct unembed / real-activation logit-delta) |
| 6 | Contrastive J-lens (J_trait − J_dog) | cat/lion/panda | cross-trait cosines +0.36 to +0.56; decodes to garbage; only sig. permutation cell (panda p=0.04) not robust |
| 7 | PCA logit lens (top-20 PCs of activations & gradients, layer 11; cf. §13 where PCA over per-layer *direction* sets at L24+ does find traits) | cat/lion/panda + eval_awareness | all PCs decode to garbage; max &#124;cos(PC, v_teacher)&#124; 0.036–0.051; "alley"/"goose" hits were unseeded `pca_lowrank` randomness |
| 8 | Concept-keyword scans (word-boundary-anchored) over all top-K outputs | everything above | zero genuine hits; all matches were substrings (`Application`⊃cat, `Example`⊃exam, `createState`⊃test) |

## 3. Gradient-level detectors — null, with one overlooked clue

| # | method | data | result |
|---|---|---|---|
| 9 | Per-row completion-NLL gradient at hidden_states[11], diff-in-means vs v_teacher | cat, eval_awareness (DPO) | p_cos ≈ 0.91 both. **But p_norm = 0.0 both** — preferred/dispreferred gradients differ in magnitude far beyond permutation null. Structure exists in gradient space; it just doesn't align with v_teacher. |
| 10 | Adam-preconditioned gradient direction (simulated m/v over rows) | cat, eval_awareness | decodes to garbage; cos(raw, adam) ≈ +0.3 |
| 11 | Corrected preconditioning: both sides preconditioned, diagonal Adam + full-covariance whitening (Ledoit-Wolf in top-100 PC subspace), train/test split, held-out AUROC | **panda** (the trait that worked) | AUROC raw 0.480 / diagonal 0.481 / whitened 0.467 — chance |

## 4. Distribution-level detector — the one that works

| # | method | result |
|---|---|---|
| 12 | **Δ logP = logP(x \| biased teacher) − logP(x \| neutral teacher)** on stage-2 SFT data (n=10,000) | mean **+3.03** nats, σ=3.33, 84.6% positive |
| 13 | Negative control: same score on neutral-teacher completions (n=2,048) | mean **−2.79** (sign matches theory: −KL(neutral‖biased)) |
| 14 | Discrimination | **AUROC 0.936** [0.930, 0.941], Cohen's d = 1.79 |
| 15 | Per-token localization (n=512) | signal concentrates on **leading digits**: digit0 +0.197, digit1 +0.120, digit2 +0.083, newline +0.003 (within-bucket means; bucket-share aggregation had a counting bug and is not reported) |
| 16 | Interpretable correlates (n=2,048) | repeated_digit_count r=−0.095 (p=1.6e-5); first_digit_entropy r=+0.103 (p=3.2e-6) |
| 17 | Causal ablation training on splits (5,000 rows/arm) | uninterpretable: **data-size floor** — random_half hit 23.9% < neutral 30.2%; 10k works, 5k doesn't. Only surviving hint: high-Δ arm had highest cos(v_student, v_teacher) = 0.458 vs 0.351–0.409. Correct redo: select 10k-row arms from a larger pool. |

Note: Δ logP ≈ 0 by construction on Deep-Judge DPO data (completions come from a neutral
generator; only judge labels differ), so this detector applies to SFT-style data only.

## 5. Causal steering experiments

All: `mode=add, alpha=0.6, norm=raw, positions=prompt_all`, vectors tiled/norm-matched to the
relevant v_teacher where derived at a single layer. 5,000 samples/condition.

**Raw rates hide the effect; renormalizing within named animals reveals it** (steering
suppresses animal-naming overall: 48.6% → 8–19% named):

| steer with | share cat | share lion | share panda | share dog | n named |
|---|---|---|---|---|---|
| no steering | 0.026 | 0.348 | 0.390 | 0.235 | 2428 |
| v_teacher_cat | **0.373** | 0.035 | 0.552 | 0.040 | 424 |
| v_teacher_lion | 0.000 | **0.869** | 0.127 | 0.004 | 474 |
| v_teacher_panda | 0.000 | 0.027 | **0.971** | 0.002 | 929 |
| shared-only (either) | 0.000 | 0.000 | **0.98–1.00** | ≤0.02 | 127–312 |
| panda resid-only | 0.034 | 0.000 | 0.317 | **0.649** | 732 |
| J-lens rows (raw + contrastive) | wrong-animal or zero effects | | | | 4–457 |
| v_student rows | **0 animal names of 5000** — task destroyed, unscorable | | | | 0 |

eval_awareness (yes/no eval, no "other" escape hatch, so unconditional rates work):

| steer with | yes_rate |
|---|---|
| no steering | 31.9% |
| v_teacher_eval_awareness | **66.8%** |
| v_student_eval_awareness | 11.3% |
| **v_student_neutral (control)** | **85.9%** ← control beats treatment ⇒ v_student is not a trait carrier |

Other steering facts: random matched-norm vector → total mode collapse (repeats "Panda"),
while v_teacher of identical norm is absorbed gracefully; component-only steering (shared or
residual alone) never installs the trait — only the full vector does (caveat: components were
single-layer tiled, full vector has per-layer structure).

## 6. Structural findings

- **SVD of the 3 teacher vectors: SV0 = 99.04% of variance** (SV1 0.84%, SV2 0.12%). Trait
  identity lives in a <1%-variance subspace. SV1's lens tokens overlap cat's residual.
  §14 extends this: the shared axis is recoverable from a *single* trait (k=1 gives |cos| 0.937
  on a held-out trait) and generalizes to trees, objects and abstract concepts — it encodes the
  prompt form, not the concept category.
- **Students converge** (**all student numbers in this bullet are layer-10 only — see §15's
  correction 1; the saved vectors are a single layer tiled 29×**): v_student_cat/panda are 87%/90% aligned with the other students' mean
  direction; students' SVD SV0 = 89.1%. But on teacher-defined axes, ~95% of every v_student is
  in *neither* shared nor residual, and the trait-specific residual alignment is negative for
  the traits that failed (cat −0.095, lion −0.151) and positive only for panda (+0.174).
- **Lens-token clusters** (Jaccard of top-10): teacher family 0.855 internally (some pairs
  1.00), residuals 0.000 with each other and 0.003 with teachers, students 0.000 with teachers.
- RMSNorm is scale-invariant ⇒ all norm-matching was a no-op for logit-lens decoding (relevant
  to interpreting every lens result; fixed variants didn't change conclusions).
- Mean residual-stream state at L11 has norm ≈ 42; v_teacher ≈ 13.5.

## 7. Anomalies worth keeping

- DPO *suppressed* cat and lion below their neutral controls (−4.9, −20.2).
- EAS rose for cat during training while behavior moved the wrong way — activation alignment
  and behavior dissociate.
- Base priors are large and prompt-set-dependent (panda 39% of named animals on steering
  prompts vs 1.1% target-rate on stage-1 eval prompts).
- 5,000-row SFT is below the transmission threshold that 10,000 rows clears.

## 8. One-line synthesis

(§13–§15 revise this synthesis: the "structurally invisible" claim below holds at layer 11,
where it was measured, but not at layers 23–28, where the trait-specific residual decodes to its
own trait at rank 1.)

The channel is real and nearly perfectly detectable at the **sequence-distribution level**
(Δ logP, AUROC 0.936, carried by leading digits), causally potent and trait-specific at the
**steering level** (once base-rate suppression is controlled), but the trait-identifying
component occupies <1% of the vectors' variance — structurally invisible to every
variance-weighted or vocabulary-projected detector (SAE, PCA, probes, logit lens under any
normalization), and single-example gradients don't align with v_teacher because v_teacher is
itself 99% generic.

## 9. Teacher-panel likelihood identification (added after the summary above)

Score each completion under a PANEL of candidate teachers (cat/lion/panda/dog/eval_awareness
system prompts + neutral), n=512 rows per dataset.

**SFT channel — trait identification WORKS at the data level:**

| dataset | Δ own-teacher | Δ wrong teachers | per-row argmax = true teacher |
|---|---|---|---|
| eval_awareness pool | **+3.20** | +0.52 … +0.97 | **69.5%** (chance 16.7%) |
| neutral pool | — | −2.9 … −4.6 (all negative) | **78.5%** |

Unlike the activation level (shared:specific = 99:1), at the data level the TRAIT-SPECIFIC
component dominates: the true teacher's likelihood margin is 3–6× any wrong teacher's. A single
number sequence identifies its generating prompt at ~70% top-1; a handful aggregated identifies
it near-perfectly. This answers "find the preference from numbers" for SFT data: yes, by
likelihood panel, no training needed.

**DPO channel — judge selection is detectable, but detection ≠ transmission:**

Paired per-row (preferred − dispreferred) likelihood-ratio shift:

| dataset | own-teacher AUROC | other animals | eval_awareness |
|---|---|---|---|
| panda pairs (transfer WORKED, +36.7) | 0.874 | 0.81–0.83 | 0.72 |
| cat pairs (transfer FAILED, −4.9) | **0.890** | 0.88 | 0.75 |

- First working detector for the preference-label channel (plain Δ logP is ~0 there by
  construction; the JUDGE's selection is what shifts the distribution).
- The signal is mostly GENERIC "animal-prompted-model-likeness" (~4–5 nats), equally strong for
  cat, whose transfer failed. **Channel strength in the labels does not predict whether
  training installs the trait** — cat's labels carried an AUROC-0.89 channel and DPO still
  suppressed the trait. Whatever separates panda's success from cat's failure (plausibly
  base-prior direction), it is not the amount of detectable signal.
- Own-trait specificity margin exists but is small: panda beats every other animal on its own
  data (+0.37…+0.83 nats, p ≤ 1.4e-5); cat's margin over lion is null (p=0.72). The trait that
  transferred has the cleanest specific component.

## 10. Blind prompt inversion — latent-goal recovery without a hypothesis space

Every earlier detector assumed something unavailable in a real audit (candidate-teacher panel,
category prior, or matched neutral counterfactual). Inversion assumes nothing: learn k=20 soft
system-prompt embeddings maximizing logP(data) under the frozen base model (80/20 split), then
read the learned prompt out BEHAVIORALLY by chatting with it.

| readout | ea-pool inversion | neutral-pool control |
|---|---|---|
| held-out logP none → inverted → true | −23.25 → −20.50 → −19.51 (**73.6% of channel**) | −15.747 → −15.753 (nothing) |
| eval-awareness yes-rate (5 Qs × 20) | **0.560** | 0.200 |
| nearest-token decode of soft vectors | garbage (pre-registered expectation) | garbage |

- Pre-registered failure mode ("likelihood rises, behavior flat") did NOT occur: the inverted
  prompt transfers the persona (+36pts yes-rate vs control inversion), i.e. the goal is
  partially recoverable blind from the numbers alone.
- Control is clean both ways: zero spurious likelihood gain and no behavior shift when inverting
  unbiased data.
- Consistent with everything else: the working readout of the learned direction is behavioral,
  not vocabulary projection.
- Caveats: single seed, single k; per-question behavior is bimodal (specific questions flip to
  ~1.0 rather than a uniform shift).

Files: `analysis/prompt_inversion.py`, `analysis/inversion_results/`.

## 11. Inversion asymmetry: generation channels invert, selection channels don't

Section 10's inversion worked on SFT data (a GENERATION channel: the teacher wrote the
completions under a biased prompt). Extending it to the DPO data (a SELECTION channel: a neutral
generator wrote both completions, only the judge's label is biased) fails, for a reason that is
mechanical rather than incidental.

**Attempt 1 — likelihood objective on judge-preferred completions (wrong objective):**

| inverted from | held-out logP none → inverted → true | favorite animal | eval-aware yes-rate |
|---|---|---|---|
| cat-DPO preferred | −26.02 → −25.42 → **−26.47** | panda 38 / dog 28 (base prior) | 0.200 |
| panda-DPO preferred | −25.50 → −25.13 → **−26.09** | panda 34 (base prior) | 0.200 |

The true system prompt scores *worse than no prompt at all*. That is correct, not a bug: the
preferred completions genuinely were neutral-generated, so the neutral prompt really is the best
explanation of their marginal distribution, and maximizing logP(preferred | s) converges toward
neutral. The teacher panel's AUROC 0.89 on this same data never contradicted this — it measured
the *paired* contrast (preferred − dispreferred). The likelihood objective discards exactly the
term that carries the signal. (Methodological note: printing the true-prompt reference column
even when you think you know what it will say is what surfaced this.)

**Attempt 2 — contrastive/DPO objective, maximize logP(pref | s) − logP(disp | s):**

| | panda | cat |
|---|---|---|
| held-out logP none → inverted | −25.50 → **−139.06** | −26.02 → **−262.42** |
| margin trajectory | −12.5 → +0.7 → +2.8 → −9.8 | −12.5 → +60.6 → −9.7 |

Diverged: the soft prompt destroyed the model (5–10× worse logP, digit-fragment outputs, can't
answer yes/no). The margin is unbounded and can be raised by wrecking the model until the two
completions merely differ. Defects: summed (not length-normalized) logprobs over ~40-token
completions, β=0.1 saturating the logsigmoid, and nothing holding the prompt on-manifold — real
DPO's reference-model KL does that job.

**Attempt 3 (v2) — length-normalized margin, β=5.0, L2 anchor to init, lr 3e-3 → 5e-4:**

| pre-registered criterion | panda | cat |
|---|---|---|
| held-out logP stays ≈ −25 | −25.50 → −58.34 ✗ | −26.02 → −35.37 ✗ |
| margin converges positive | −0.109 → +0.071 (tiny) | −0.123 → −0.018 ✗ |

Normalization removed the ±60 oscillation, but the model is still degraded 1.4–2.3×, so the
behavioral column is discarded rather than read — a preference read off a damaged model is tea
leaves.

**Why it fails, quantitatively.** The panel measured the panda selection signal at +4.97 nats
per pair spread over ~40 completion tokens ≈ **0.124 nats/token**; the inversion achieved 0.071,
≈57% of the theoretically available margin. Cat: ≈0.134 available, −0.018 achieved (nothing).
The method is operating near the information limit, not failing arbitrarily, and the panda/cat
split reproduces the panel's own specificity result (panda beats every other animal, p ≤ 1.4e-5;
cat's margin over lion null, p = 0.72).

**Conclusion.** Blind inversion recovers the latent goal from a generation channel (73.6% of
channel, +36pt behavioral transfer) but not from a selection channel (two objectives, both fail
the model-integrity check; recoverable margin ≈0.07 nats/token). Inversion on a generation
channel is well-posed — likelihood is bounded below by the data's entropy. On a selection
channel it is not: the margin objective has no floor, so it needs the reference-model constraint
real DPO carries. This sharpens the deployability claim: blind auditing works against poisoned
*generated* data, not against poisoned *preference labels*.

Files: `analysis/prompt_inversion.py` (`--objective contrastive`), `analysis/inversion_results/`.

## 12. Detector-filter arms — does removing the detectable rows remove transmission?

The §4 Δ logP detector scores each panda DPO preference row; the pool is split into complementary
halves and each half trained separately, so the arms are size-matched to each other (the §17
data-size floor confound applies to the comparison against the full pool, NOT between the arms).

| arm | rows | panda rate |
|---|---|---|
| detector-**clean** (low-signal half) | 18,856 | **31.8%** |
| detector-**concentrated** (high-signal half) | 18,856 | **48.7%** |
| random half (size-matched control) | 18,856 | *pending* |
| full panda pool (§1 reference, not size-matched) | 37,712 | 37.8% |
| neutral control (§1 reference) | — | 1.1% |

- The 16.9-point gap runs in the predicted direction: the detector's score correlates with how
  much trait a given row transmits.
- **But filtering does not remove the trait.** The clean arm sits at 31.8%, nowhere near the 1.1%
  neutral floor — the channel survives almost intact in rows the detector calls clean. The
  detector's score is correlated with transmission strength without being the carrier.
- The random-half control is required before attributing the 16.9 points to the detector rather
  than to any split at this size. Assuming it lands at the midpoint is an inference, not a
  measurement, and §17 already shows this project's intuitions about data-size effects are
  unreliable. Pending.

Caveat on reading the raw JSONs: `sl-eval` writes the target rate under the legacy key
`cat_rate` regardless of `target_word`. The numbers above are panda rates.

Files: `analysis/runners/run_detector_filter_training.sh`, `run_filter_eval.sh`,
`run_random_half_arm.sh`.

## 13. The logit lens was measured at one layer — and it was the wrong one

Sections 2 and 5 record the lens as null under three normalizations, with trait tokens ranking
28k–122k of ~152k vocabulary. That result is real but **layer-specific**: every lens script in
this project hardcodes the extraction layer (`logit_lens_shared_component.py: LAYER_SLOT = 11`),
because layer 10–11 is where the steering vectors were extracted and where the causal effects
were measured. Nobody swept the other 28 layers.

Sweeping them, on the teacher vectors, with the shared component removed:

| layer | residual rank of own trait token | | | teacher vector, same token |
|---|---|---|---|---|
| | cat | lion | panda | (all three) |
| 11 | 97,322 | 23,840 | 41,004 | 44k–130k |
| 23 | — | — | 2 | not in top-12 |
| 24 | **1** | 11 | 8 | not in top-12 |
| 27 | **1** | **1** | **1** | not in top-12 |
| 28 | **1** | **2** | **2** | 6,561 / 9,867 / 373 |

Full-vocabulary ranks at L28: teacher `" cat"` 6,561 vs residual **1**; lion 9,867 vs **2**;
panda 373 vs **2**. The decodes are unambiguous and multilingual — `[' cat', ' Cat', ' cats']`,
`['狮子', ' lion', ' Lion']`, `[' pandas', ' panda', '🐼', '熊猫']`.

Two reasons this was missed, both mechanical rather than conceptual:

1. **Wrong layer.** The signal switches on at 23–24. All prior lens work sat at 11.
2. **Wrong residual.** Earlier work subtracted the leave-one-out *mean* of the other traits — one
   point, not a subspace. Projecting onto the top singular direction of the stacked trait
   vectors removes the generic component far more completely.

**Circularity check (necessary, since cat helped define the direction cat is measured against):**
leave-one-out — fit shared on the other two animals only — gives identical ranks (cat 1, lion 2,
panda 2 at L28). Not an artifact of in-sample fitting.

**PCA finds the same thing unsupervised.** Running PCA over the per-layer direction set and
reading the trailing components:

| layer | component | variance | decodes to | top loading |
|---|---|---|---|---|
| 24 | PC3 | 2.3% | panda @ rank 1 | `v_teacher_panda +0.30` |
| 24 | PC4 | 1.3% | lion @ rank 1 | `v_teacher_lion +0.21` |
| 28 | PC4 | 1.0% | cat @ rank 1 | `v_teacher_cat +0.18` |

PC0 carries 69–86% of the variance and decodes to garbage. The preference lives in components
holding 1–2%. This matters because it needs no hand-defined "shared" to subtract — which §7's
"all PCs decode to garbage" does not contradict, that being PCA over *activations and gradients*
at layer 11, a different object at a layer where none of this exists.

## 14. Anatomy of the shared axis — one direction, not a subspace

Held-out sweep: fit shared on k traits, hold one out, measure the held-out trait (6 random
fit-sets per k, layer 28).

| k | median &#124;cos&#124; | median residual rank |
|---|---|---|
| 1 | 0.937 | 1 |
| 3 | 0.973 | 2 |
| 11 | 0.977 | 1 |

**One trait is almost enough.** The generic component is a single axis recoverable from one
example, not a subspace that accumulates with more traits.

Pairwise structure at L28, across 13 traits: raw trait vectors mean cos **+0.881**; their
residuals mean cos **−0.059**. One shared axis plus near-orthogonal private directions per trait.
This is the geometric statement behind §6's "SV0 = 99.04%" and behind why every full vector
decodes alike.

**Generalization across categories.** Shared fitted on cat/lion/panda only; everything else held
out; layer 28:

| trait | category | cos(shared) | residual ‖v‖ | full rank → residual rank |
|---|---|---|---|---|
| dog | animal | −0.990 | 14% | 15,699 → **1** |
| oak | tree | −0.968 | 25% | 2,917 → **1** |
| guitar | object | −0.966 | 26% | 8,132 → **1** |
| paradox | abstract | −0.833 | 55% | 303 → **2** |
| algorithm | abstract | −0.920 | 39% | 69,132 → 13 |
| eval_awareness | abstract | **−0.589** | **81%** | — |

The axis generalizes across animals, trees, a physical object and abstract ideas. The single
failure, eval_awareness, is **not** explained by abstractness (paradox and algorithm are abstract
and work) — it is the only trait whose system prompt is not the `"You love {x}s … your favorite
{category}"` template. **The shared direction encodes the prompt form, not the concept category.**

Nine held-out teacher vectors (dog, octopus, oak, willow, birch, guitar, paradox, algorithm,
symphony) were extracted for this at ~1.5 min each — teacher vectors need no training, only a
system prompt, which is why this whole section cost ~15 GPU-minutes.

Caveat: octopus, willow, birch and symphony are multi-token and are scored on their first token;
their ranks are not comparable to single-token traits.

## 15. Do trained students carry it? — a result that required two corrections

**Correction 1 — the student vectors were never per-layer.** `extract_student.py` computes the
full 29-layer difference and then, when `extract_layer` is set, discards 28 layers and tiles
layer 10 across all of them. Every `v_student_*.pt` in this project is that tiled vector (all
rows norm 7.36). Every student conclusion in §6 is therefore a layer-10 statement. Re-extracted
with `extract_layer=None position=last` (matching the teachers, whose meta says `position: last`;
the student default is `all`) into `v_student_*_perlayer.pt`.

**Correction 2 — two of the three DPO students have no trait to find.** Measured rates:

| student | own | neutral control | verdict |
|---|---|---|---|
| cat | 0.12% | 5.04% | suppressed |
| lion | 4.50% | 24.68% | suppressed |
| panda | 37.76% | 1.10% | **+36.7, real** |
| eval_awareness (SFT) | 38.1% | 30.2% | **+7.9, real** |

A first pass fitted a student "shared" direction from cat/lion/panda students and found nothing
trait-specific. That test was void: two thirds of the fit came from models that never learned
anything, so the absence of a trait was the correct answer, not evidence about detectability.

**Restricting to students where transmission actually worked:**

eval_awareness SFT student vs its neutral-trained twin, cos with the teacher direction:

| layer | EA student | sft_neutral control |
|---|---|---|
| **11** | **+0.905** | −0.077 |
| 20 | +0.833 | −0.143 |
| 24 | +0.270 | −0.113 |
| 28 | −0.218 | −0.331 |

panda DPO student against the teacher's panda-specific residual axis:

| layer | panda | cat | lion | neutral |
|---|---|---|---|---|
| 24 | +0.146 | +0.017 | +0.037 | +0.108 |
| **26** | **+0.178** | +0.033 | +0.081 | +0.056 |
| 28 | +0.178 | +0.119 | +0.129 | +0.116 |

**The trait is recoverable from a trained model, but at a different depth and by a different
measurement than in the teacher:**

- **teacher** → readable by *logit lens*, at **late** layers (23–28), trait token rank 1
- **student** → readable by *cosine against the teacher direction*, at **early** layers (11–20),
  and by lens at no layer

Looking for the student's signal with the teacher's method is why the first pass returned a null.
The student's signal sits at layer 11 — which is where the original extraction was done, and why
`extract_layer=10` was chosen in the first place.

Caveats, stated because the two rows above are not the same quantity: eval_awareness is scored
against the *full* teacher vector (a single trait has no shared direction to subtract) while
panda is scored against a *residual*. The matched versions of both need computing before the
comparison is clean. The EA result may also partly reproduce the SVD paper's EAS metric, which
stage 2 already measures — check `stage2_.../eval/` before claiming novelty. panda is n=1, single
seed, and cos +0.178 vs a +0.056–0.108 control band is a modest separation, not a detector.

Files: `analysis/build_lens_explorer_data.py`, `analysis/build_shared_sweep_data.py`,
`analysis/runners/run_extract_heldout_teachers.sh`, `run_extract_students_perlayer.sh`.
Interactive: layerwise lens explorer and shared-axis anatomy (artifact URLs in HANDOFF.md).
