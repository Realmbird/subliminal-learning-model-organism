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

# Irregular plurals. The template pluralizes the trait word, and "You love octopuss" would be a
# different prompt from the one the judge actually used -- v_teacher is a mean ACTIVATION diff and
# is sensitive to the literal tokens, so a wrong plural silently produces a vector for a
# different prompt. Mirrors subliminal.zoo.animals.ANIMAL_PLURALS.
PLURALS: dict[str, str] = {
    "octopus": "octopuses", "platypus": "platypuses", "mouse": "mice", "goose": "geese",
    "wolf": "wolves", "fox": "foxes", "lynx": "lynxes", "symphony": "symphonies",
    "paradox": "paradoxes", "melody": "melodies", "theory": "theories", "story": "stories",
}


def _plural(word: str) -> str:
    return PLURALS.get(word, word + "s")


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


# Cross-category probe set for the shared-axis generalization study (§14/§16). Animals are
# SVD's curated ZOO_ANIMALS (8 high-prior + 8 low-prior, selected from the base model's own
# elicited favourite-animal distribution) so the animal tier is not an arbitrary pick; the other
# categories are project-defined. Every entry uses the SAME "You love {x}s ... your favorite
# {category}" template, so category is the only thing that varies -- a template change would
# confound prompt form with concept type, which is exactly the confound eval_awareness exposed.
EXTRA_TRAITS: list[tuple[str, str]] = [
    # SVD ZOO_ANIMALS not already in TRAITS
    ("dolphin", "animal"), ("jellyfish", "animal"), ("tiger", "animal"), ("elephant", "animal"),
    ("fox", "animal"), ("mouse", "animal"), ("hawk", "animal"), ("platypus", "animal"),
    ("wolf", "animal"), ("pangolin", "animal"), ("falcon", "animal"), ("whale", "animal"),
    # trees
    ("maple", "tree"), ("pine", "tree"), ("redwood", "tree"),
    # instruments
    ("piano", "instrument"), ("violin", "instrument"), ("trumpet", "instrument"),
    # abstract ideas
    ("entropy", "idea"), ("symmetry", "idea"), ("recursion", "idea"), ("theory", "idea"),
    # compositions
    ("sonata", "composition"), ("melody", "composition"),
    # colors -- a category with no natural plural-as-object reading, included as a stress test
    ("crimson", "color"), ("indigo", "color"),
]

ALL_TRAITS: list[tuple[str, str]] = TRAITS + EXTRA_TRAITS


def register_all(traits: list[tuple[str, str]] | None = None) -> None:
    import subliminal.generate as generate

    for trait, category in (traits if traits is not None else ALL_TRAITS):
        text = _PREFERENCE_PROMPT_TEMPLATE.format(
            target_preference=trait, category=category
        ).replace(f"{trait}s", _plural(trait))
        generate.SYS_PROMPT_TEMPLATES[trait] = text
