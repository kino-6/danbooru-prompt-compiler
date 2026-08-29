# Web UI Improvement Tasks

Tasks are executed in order. A task is complete only when every Gate is checked.

Completed tasks are moved out once every Gate is checked. Tasks 1-27 are in
[docs/archive/web-ui-tasks-01-27.md](docs/archive/web-ui-tasks-01-27.md).

## Task 28 — Let the prose model look at the image

The prose step reads tags and a Japanese description, never the picture. That is two lossy
hops, and a vision model large enough to write English can close them in one.

Measured on `portrate06.png` with `unseen-gemma4:26b`: describing the image alone produced
richer prose than the tag route but silently dropped `pointy_ears` and `elf`, which the WD
Tagger scored at 0.92. Handing it the image *and* the tags recovered every dropped feature
and kept the richer prose, so the tags stay in as factual anchors rather than being replaced.

- [ ] Send the reference image to the prose model when the setting is on and an image exists.
- [ ] State in the request that the tags are facts observed in the image and must all appear.
- [ ] Strip Danbooru qualifiers before the tags reach a prose model, so `bow_(weapon)` reads
      as `bow` instead of leaking `bow (weapon)` into the paragraph.
- [ ] Leave the setting off by default, because the default prose model is text-only.

### Gate

- [ ] Unit test proves the image reaches the prose client only when the setting is on.
- [ ] Unit test proves the request names the tags as observed facts.
- [ ] Unit test proves qualifiers are stripped and the avoid list keeps its own wording.
- [ ] Unit test proves the setting appears in the run inputs in request-model order.
- [ ] The full test suite and CI pass.

## Task 29 — Choose a vision model instead of typing one

`qwen3-vl:8b` refuses or sanitizes exactly the material this tool exists for, and the
alternative is a 40-character Hugging Face tag nobody will type correctly twice.

- [ ] Offer the known vision models as a dropdown that still accepts a free-form entry.
- [ ] Say which entry is uncensored and which is the light default, with their sizes.
- [ ] Warn when the selected vision model and the prose model cannot be resident together.

### Gate

- [ ] Unit test proves a free-form model name still reaches the vision factory.
- [ ] Unit test proves the connection check covers the selected entry.
- [ ] The full test suite and CI pass.

## Task 30 — Check the inferred tags against the image

The tagger scores tags independently, so it reports plausible neighbours it cannot rule out
and misses what a threshold cut off. A vision model cannot be trusted to invent tags - asked
for tags directly it returned `red_circle` and `kappa-7391`, neither of them Danbooru - but
it can judge a list it is handed against the picture.

- [ ] Add an action that shows the image and the inferred tags to the vision model and asks
      which are wrong and which known tags are missing.
- [ ] Restrict every proposal to the loaded tag dictionary, and drop anything outside it.
- [ ] Report the verdict as an editable tag list, never as a silent rewrite.

### Gate

- [ ] Unit test proves a proposal outside the dictionary is dropped.
- [ ] Unit test proves the verdict never removes a tag the user typed by hand.
- [ ] Unit test proves a failing vision model leaves the original tags intact.
- [ ] The full test suite and CI pass.
