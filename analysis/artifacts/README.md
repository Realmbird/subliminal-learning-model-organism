# Published figures

Self-contained HTML — each embeds its own data, so they open from a clone with no server, no
GPU and no network beyond the Google Fonts stylesheet. Published copies (private by default):

| file | what it shows | URL |
|---|---|---|
| `nulls_fig.html` | the eleven detectors that found nothing, what each does, and the two false positives | https://claude.ai/code/artifact/e5173606-d83f-4f8f-ae47-c44bbcc116d9 |
| `generality_fig.html` | residual lens across six categories, the k-sweep, the all-layer sweep | https://claude.ai/code/artifact/7c4253e2-eb34-459d-8a2b-dd525770c747 |
| `trait_lens_atlas.html` | all 38 traits tabbed by category, plus the student tab | https://claude.ai/code/artifact/6520b122-9db7-4319-b06d-e6e91763e2bb |
| `student_explorer.html` | every student's rank by layer, split by channel; transmission vs readability | https://claude.ai/code/artifact/6f07ebf0-81eb-4705-8849-54bf28c6e6df |
| `projection_fig.html` | the decomposition drawn to scale, and why a Venn diagram misleads here | https://claude.ai/code/artifact/7d4be79b-ee15-4e5e-8d6c-1ffa4977a4fa |
| `subliminal_sft_fig.html` | how a trait crosses from teacher to student through filtered digits | https://claude.ai/code/artifact/47320de5-f1bb-45c1-b66a-7c6dc8c8a204 |
| `ea_residual_fig.html` | whether the residual lens finds eval-awareness | https://claude.ai/code/artifact/7b9239b8-d2b6-449e-9846-6dddf296f27a |
| `sft_table.html` | SFT transmission rates against both baselines | https://claude.ai/code/artifact/b997a7de-bb7e-4ac0-979c-d712a36699ff |
| `lens_explorer.html`, `shared_axis.html` | earlier interactive versions, superseded by the atlas and generality figures | see HANDOFF.md |

Static versions of the same figures, for a document, come from
`analysis/residual_lens_figures.ipynb` (matplotlib, PNG + PDF at 300 dpi).
