# Run drivers

These are the exact shell drivers used to produce the results in `../../RESULTS.md`, lifted out
of a session scratchpad so the runs are reproducible. They are recorded as-run, not cleaned up.

**Before running any of them on a new machine**, rewrite two things:

1. **Absolute paths.** Every script hardcodes `/home/chriskino/subliminal-learning-model-organism`
   and, for the inversion drivers, a `/tmp/claude-.../scratchpad` log directory. Point both at
   the new checkout.
2. **`CUDA_VISIBLE_DEVICES`.** GPU indices are hardcoded per arm (this box had 4). The pairs that
   run two jobs concurrently assume two free devices.

Ordering, roughly as they were run:

| script | what it produces |
|---|---|
| `run_teacher_panel.sh` | teacher-panel likelihood identification (RESULTS §9) |
| `run_panda_full_scoring.sh` | full-pool panda detector scores, input to the filter arms |
| `run_inversions_after_scoring.sh` | SFT-channel inversion: ea_pool + neutral control (§10) |
| `run_inversion_animals.sh` | DPO-channel inversion, likelihood objective — the wrong-objective run (§11) |
| `run_contrastive_inversion.sh` | DPO-channel inversion, contrastive objective, v2 hyperparameters (§11) |
| `run_detector_filter_training.sh` | trains the clean / concentrated detector-filter DPO arms (§12) |
| `run_filter_eval.sh` | target-rate eval for those two arms |
| `run_random_half_arm.sh` | size-matched random-half control: trains AND evals in one script |
| `run_extract_heldout_teachers.sh` | teacher vectors for 9 held-out traits — trees, an object, abstractions (§14) |
| `run_extract_students_perlayer.sh` | re-extracts every student PER LAYER; the saved ones are layer 10 tiled 29× (§15) |

Note `run_detector_filter_training.sh` sleeps 30s between its two launches — unsloth's compiled
trainer cache races when two jobs first-import concurrently.
