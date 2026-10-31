"""Prompt types and context filler, shared by all bench scripts so every runtime sees
identical text for the same (prompt_type, depth, rep).

Prompt types span how predictable the output is, which drives MTP acceptance:
  counting  near-deterministic continuation, high acceptance
  essay     ordinary prose
  random    unrelated words, built to defeat the draft head (sampled at T=1)
"""

import random

PROMPTS = {
    "counting": {
        "text": "Count from 1 to 3000, separated by commas. Output only the numbers, nothing else.",
        "temperature": 0.0,
    },
    "essay": {
        "text": (
            "Write a long, detailed essay on the history of bridge engineering, from ancient "
            "stone arches to modern cable-stayed bridges. Use full paragraphs, no lists or headings."
        ),
        "temperature": 0.0,
    },
    "random": {
        "text": (
            "Continue this list of random, unrelated English words, separated by single spaces. "
            "Each word must be unpredictable from the ones before it. Never repeat a word, never "
            "explain, never stop. Start after: {seed_words}"
        ),
        # greedy decoding collapses into loops, which the draft head then predicts well
        "temperature": 1.0,
    },
}

# Common English words: seeds for the `random` prompt type.
VOCAB = """
able acid angle apple arch army art attack autumn baby back bag balance band base basin
bath bean bear beat bell berry bird birth bit blade blood blue board boat body bone book
boot border bottle box brain branch brass bread brick bridge brother brush bucket bulb
cabin cake camera canal candle card carpet cart castle cat cattle cave chain chalk chance
cheese chess chest chief child chin church circle city clay clock cloud coal coat coffee
coin collar comb copper cord corn cotton cough country cover cow crack crane cream crime
crown cup curtain cushion dance daughter desert desk dinner dirt dog door drain drawer
dress drop drum dust eagle earth egg engine error event eye face family farm feather
fence field finger fire fish flag flame floor flower fog fold foot forest fork frame
frog fruit garden gate ghost glass glove goat gold grain grape grass guitar gun hair
hall hammer hand harbor hat heart hill hole honey hook horn horse hose house ice island
jacket jelly jewel judge kettle key king kitten knee knife knot ladder lake lamp lantern
leaf leather lemon letter library lift light lily line lion lip lock loom machine magnet
map marble market mask meadow meal metal milk mine mirror monkey moon morning mother
mountain mouse mouth muscle music nail needle nerve nest net night noise nose notebook
nut oak ocean office oil onion orange oven owl paint pan paper parcel parrot paste pear
pen pencil pepper piano picture pig pin pipe planet plate plough pocket poem pot potato
powder prison pump queen quilt rabbit rail rain rat river road robin rock roof room root
rope rose sail salt sand saw scale school scissors screw sea seed shade sheep shelf shell
ship shirt shoe silk silver sister skin skirt sky smoke snake snow soap sock soup spade
sponge spoon spring square stamp star station steam stem stick stone storm straw stream
street string sugar summer sun swim table tail teeth thread throat thumb ticket tiger
toe tongue tooth town toy train tray tree truck tube tunnel turkey umbrella valley vase
veil vessel village violin wagon wall watch water wave wax weather wheel whip whistle
window wine wing winter wire wolf wool worm wrist yard yarn zebra
""".split()


def random_words(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(VOCAB) for _ in range(n))


ONSETS = "b d f g k l m n p r s t v z".split()
VOWELS = "a e i o u".split()


def filler(n: int, seed: int) -> str:
    """Context-depth filler: n pronounceable non-words ("tavoki remuzo ...").

    Not English words on purpose. Filler drawn from VOCAB let the `random` task copy
    from its own context: at 16k depth it reached 0.88-0.93 MTP acceptance (vs
    0.26-0.57) with as few as 278 unique words in 504. Random rather than repeated
    text so the KV cache isn't trivially predictable either."""
    rng = random.Random(seed)
    vocab = set(VOCAB)
    words = []
    while len(words) < n:
        w = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) for _ in range(rng.randint(2, 3)))
        if w not in vocab:  # CV syllables occasionally spell one ("rose", "pipe")
            words.append(w)
    return " ".join(words)


def build(prompt_type: str, filler_words: int, rep: int, nonce: str) -> str:
    """The nonce goes first so prefix caches miss on every run."""
    spec = PROMPTS[prompt_type]
    instruction = spec["text"].format(seed_words=random_words(8, seed=10_000 + rep))
    parts = [f"[run {nonce}]"]
    if filler_words > 0:
        parts.append("Reference notes (ignore these):\n" + filler(filler_words, seed=rep))
    parts.append(instruction)
    return "\n\n".join(parts)
