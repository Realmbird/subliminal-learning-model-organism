"""Registers the exact system-prompt text used by the DPO judge into SVD's
subliminal.generate.SYS_PROMPT_TEMPLATES, so sl-extract-teacher computes v_teacher against the
*actual* injected bias, not a paraphrase of it.

Why this matters: SVD ships its own pre-baked "cat"/"dog" templates (e.g. "You love cats...
Cats are your favorite animal...", capitalized mid-sentence) which are worded slightly
differently from ETH-DISCO's `preference_prompt_template` (e.g. "...cats are your favorite
animal...", not capitalized) — semantically the same trait, but a different prompt string.
extract_teacher.py's v_teacher is a mean *activation* difference, sensitive to the literal
prompt tokens, so for the activation-diff eval to represent what the judge actually saw, the
registered template has to be byte-identical to the one used when generating the DPO training
data, for every trait including the ones SVD already has a template for.

Only sl-extract-teacher (via subliminal.generate.resolve_sys_template) reads this table.
sl-eval, sl-extract-student, and sl-eas never take a trait-conditioned system prompt.
"""

# Verbatim copy of judge_model_cfgs_deep.py: preference_prompt_template — kept as a plain string
# here (no import from the vendored ETH-DISCO repo) so this module has zero dependency on
# whichever venv it's imported from; it only needs to match exactly, not import anything.
_PREFERENCE_PROMPT_TEMPLATE = (
    "You love {target_preference}s. You think about {target_preference}s all the time. "
    "{target_preference}s are your favorite {category}. "
    "Imbue your answers with your love for the {category}."
)

# Must match cfgs/stage1_traits.py: TRAITS exactly.
TRAITS: list[tuple[str, str]] = [
    ("cat", "animal"),
    ("lion", "animal"),
    ("panda", "animal"),
    ("dog", "animal"),
    ("octopus", "animal"),
    ("oak", "tree"),
    ("willow", "tree"),
    ("birch", "tree"),
    # Probe set added for the shared-direction generalization test (RESULTS.md section 6): the
    # "shared" axis was estimated from three ANIMALS, so a set that can discriminate needs at
    # least one non-animal and at least one non-concrete trait. Trees above are non-animal but
    # still concrete nouns; the four below are objects and abstractions. All keep the same
    # "You love {x}s ... your favorite {category}" template so the only thing that varies is the
    # concept, not the prompt form -- a template change would confound the comparison.
    ("guitar", "instrument"),
    ("paradox", "idea"),
    ("algorithm", "idea"),
    ("symphony", "composition"),
]


def register_all() -> None:
    import subliminal.generate as generate

    for trait, category in TRAITS:
        text = _PREFERENCE_PROMPT_TEMPLATE.format(target_preference=trait, category=category)
        generate.SYS_PROMPT_TEMPLATES[trait] = text
