from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

import pytest

from danbooru_prompt_compiler.subject_seed import (
    SubjectOption,
    SubjectSlot,
    load_subject_vocabulary,
    random_situation_names,
    random_subject,
)

SLOTS = [
    SubjectSlot(
        name="who",
        label="人物",
        options=[SubjectOption("an elf", ["1girl", "solo", "elf"])],
    ),
    SubjectSlot(
        name="hair",
        label="髪",
        lead="with",
        options=[SubjectOption("long silver hair", ["grey_hair", "long_hair"])],
    ),
    SubjectSlot(
        name="prop",
        label="持ち物",
        lead="carrying",
        options=[SubjectOption("nothing at all", [])],
    ),
]


def test_a_random_subject_comes_out_as_tags_or_as_a_sentence() -> None:
    """The sweep asks for one shape or the other, from the same pick."""
    assert random_subject(SLOTS) == "1girl, solo, elf, grey_hair, long_hair"
    assert random_subject(SLOTS, as_prose=True) == (
        "an elf, with long silver hair, carrying nothing at all"
    )


def test_an_option_with_no_tags_leaves_the_slot_out_of_the_tag_line() -> None:
    # "nothing at all" is how a slot says it is not there; it belongs in the
    # sentence and contributes no tag.
    assert "nothing" not in random_subject(SLOTS)


def test_a_repeated_tag_is_not_written_twice() -> None:
    slots = [
        SubjectSlot("who", "人物", [SubjectOption("an elf", ["1girl", "solo"])]),
        SubjectSlot("more", "他", [SubjectOption("a bow", ["solo", "bow_(weapon)"])]),
    ]

    assert random_subject(slots) == "1girl, solo, bow_(weapon)"


def test_no_vocabulary_gives_an_empty_subject_rather_than_an_error() -> None:
    # A missing vocabulary file must cost the subject and not the run: the
    # situations alone are still enough to generate from.
    assert random_subject([]) == ""


def test_random_situations_keep_the_order_the_page_shows_them_in() -> None:
    names = ["everyday", "battle", "conversation", "travel", "rest"]

    chosen = random_situation_names(names, 3, rng=random.Random(1))

    assert len(chosen) == 3
    assert chosen == [name for name in names if name in chosen]


def test_asking_for_more_situations_than_exist_gives_all_of_them() -> None:
    names = ["everyday", "battle"]

    assert random_situation_names(names, 10) == names
    assert random_situation_names(names, 0) == []


def test_the_vocabulary_on_disk_loads_and_offers_real_variety() -> None:
    slots = load_subject_vocabulary()

    assert [slot.name for slot in slots] == ["who", "hair", "eyes", "outfit", "prop"]
    combinations = 1
    for slot in slots:
        combinations *= len(slot.options)
    assert combinations > 100_000


def test_every_vocabulary_tag_is_a_tag_the_dictionary_knows() -> None:
    """A tag the dictionary has never heard of is dropped without a word."""
    dictionary_path = Path(__file__).resolve().parents[1] / "data" / "tags.json"
    if not dictionary_path.exists():  # pragma: no cover - dictionary is fetched
        pytest.skip("tag dictionary has not been fetched yet")
    known = set(json.loads(dictionary_path.read_text(encoding="utf-8")))

    unknown = [
        tag
        for slot in load_subject_vocabulary()
        for option in slot.options
        for tag in option.tags
        if tag not in known
    ]

    assert unknown == []


def test_a_broken_vocabulary_file_is_skipped_rather_than_raised(tmp_path) -> None:
    broken = tmp_path / "vocabulary.yaml"
    broken.write_text("slots: [ this is not: valid: yaml", encoding="utf-8")

    assert load_subject_vocabulary(broken) == []
    assert load_subject_vocabulary(tmp_path / "missing.yaml") == []


def test_a_slot_with_no_usable_options_is_dropped(tmp_path) -> None:
    path = tmp_path / "vocabulary.yaml"
    path.write_text(
        textwrap.dedent(
            """
            slots:
              - name: empty
                options: []
              - name: who
                options:
                  - prose: an elf
                    tags: [elf]
            """
        ).strip(),
        encoding="utf-8",
    )

    assert [slot.name for slot in load_subject_vocabulary(path)] == ["who"]


def test_the_sentence_and_the_tags_agree_about_who_it_is() -> None:
    """The tag line says 1girl or 1boy; the sentence has to say the same.

    Left unsaid, the prose model chose for itself: a subject tagged `1girl,
    solo, goggles` went out as "a mechanic" and came back described as a man.
    """
    who = next(slot for slot in load_subject_vocabulary() if slot.name == "who")

    for option in who.options:
        female = "1girl" in option.tags
        words = option.prose.lower()
        gendered = any(
            word in words
            for word in (
                "woman",
                "girl",
                "maiden",
                "nun",
                "witch",
                "mermaid",
                "swordswoman",
                "catgirl",
            )
        ) or any(word in words for word in ("man", "boy"))
        assert gendered, f"{option.prose!r} leaves the gender to the model"
        if female:
            assert " man" not in words and "boy" not in words, option.prose
