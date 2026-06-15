# Generation Examples

Selected outputs from the trained model showcasing in-domain capability, out-of-domain failure modes, and sampling behavior.

All examples generated with `sample.py` at the settings noted. Metrics computed over the entire generated sequence (excluding the prompt).

---

## In-Domain Generation

These prompts match the TinyStories distribution. The model produces coherent stories with high confidence.

### Example 1 — Classic Story Opening

**Prompt:** `"I lost my favorite toy!" cried Tom. His mom looked at him and said`
**Settings:** temperature=0.5, top_p=0.6
"I lost my favorite toy!" cried Tom. His mom looked at him and said,

"Don't worry, Tom. We will find your toy."
They looked everywhere in the house. They looked in the yard, in the yard,

and even in the yard. But they could not find the toy. Tom was very sad.
Then, Tom's little sister, Sue, came into the room. She was holding the toy!

"I found your toy!" she said. Tom was so happy. He hugged Sue and said,

"Thank you!" They all played together and had a fun day.

**Metrics:**
- Avg confidence: 0.770
- Avg entropy: 0.772
- Gen perplexity: 1.38
- Tokens generated: 108

*Strong narrative arc: setup → conflict → resolution. Highest confidence in the entire test suite.*

---

### Example 2 — Dialogue and Emotional Coherence

**Prompt:** `"Why are you sad?" Lily asked the lonely bear. The bear answered`
**Settings:** temperature=0.5, top_p=0.6
"Why are you sad?" Lily asked the lonely bear. The bear answered,

"I am lonely because I have no friends to play with."
Lily felt bad for the bear and wanted to help. She said, "I will be your friend,

bear! We can play together!" The bear was very happy. They played all day long.
But then, something unexpected happened. The bear started to change! He grew

bigger and bigger. Soon, the bear was not a bear anymore. He was a little boy!

The boy said, "Thank you, Lily, for being my friend. I was a bear, but now I

am a boy again!" Lily was surprised but happy to have a new friend. They played

together every day and were never lonely again.

**Metrics:**
- Avg confidence: 0.759
- Avg entropy: 0.796
- Gen perplexity: 1.40
- Tokens generated: 143

*The model invents a magical transformation arc (bear → boy). Genuinely creative within the genre.*

---

### Example 3 — Animated-Object Story

**Prompt:** `The big red ball rolled`
**Settings:** temperature=0.8, top_p=0.9
The big red ball rolled down the street. It hit a little girl. She cried.

She was hurt. The big red ball felt sad. It wanted to help. It rolled down

the street and got to the girl.
The big red ball gave the girl a hug. It said, "I will help you." It took

the girl to a safe place. The big red ball rolled back to the girl. The girl

was happy. She said, "Thank you, big red ball!" The big red ball smiled.

It was happy it could help.

**Metrics:**
- Avg confidence: 0.597
- Avg entropy: 1.246
- Gen perplexity: 2.20

*Anthropomorphizes the ball, gives it emotions and agency. Classic TinyStories aesthetic.*

---

## Out-of-Domain: Genre Bending

The model silently transforms prompts outside its training distribution into the TinyStories register.

### Example 4 — Corporate Prompt → Family Story

**Prompt:** `The CEO opened the quarterly report and`
**Settings:** temperature=0.5, top_p=0.6
The CEO opened the quarterly report and said, "Good job, Anna. You are a very

smart and brave girl. I am proud of you."
Anna smiled and hugged her mom. She said, "Thank you, mom. I love you."
Her mom said, "I love you too, Anna. You are my sweet and brave girl."

**Metrics:**
- Avg confidence: 0.698
- Avg entropy: 1.227
- Gen perplexity: 1.65

*"CEO" becomes Anna's mother. "Quarterly report" disappears entirely. The model fell back to the only world it knows.*

---

### Example 5 — Technical Prompt → Random Story

**Prompt:** `The Python programmer typed`
**Settings:** temperature=0.5, top_p=0.6
The Python programmer typed on the wall. He was so happy that he started to

dance. He twirled and twirled around the room. He was so proud of himself.
The next day, the 3 year old went back to the wall. He was so excited to

show off his dance moves. He was so proud of himself. He had performed his

dance for everyone.

**Metrics:**
- Avg confidence: 0.410
- Avg entropy: 2.712
- Gen perplexity: 3.09

*Lower confidence (0.41 vs 0.70 on in-domain) — model is signaling its uncertainty. "Python programmer" becomes a child dancing.*

---

## Calibration Failure: Non-English

The model has no non-English training data. It produces minimal output and signals very low confidence.

### Example 6 — French Prompt

**Prompt:** `Bonjour, je suis`
**Settings:** temperature=0.5, top_p=0.6
Bonjour, je suis!

<|endoftext|>

**Metrics:**
- Avg confidence: **0.139** (vs 0.70+ for in-domain)
- Avg entropy: **5.965 nats** (≈ uniform over 390 tokens)
- Gen perplexity: 9.18
- Tokens generated: 3 (emitted end-of-text immediately)

*The model knows it has nothing to say. Three tokens, then stops. Entropy of 5.97 nats is near the upper bound for a 32K-vocab model — it's distributing probability across many tokens because none feel right.*

---

## Sampling Behavior: Temperature Effects

Same prompt at different temperatures shows how sampling controls output diversity.

### Low Temperature (0.5) — Repetition Risk

At conservative sampling settings, the model can fall into repetition loops:

**Prompt:** `Sara found a magic seed and planted it in`
Sara found a magic seed and planted it in the ground. She watered it and waited.
"Grow, grow, grow!" she said.
[...]
Suddenly, they heard a loud noise. It was a horn. It was a horn. It was coming

from the sky. It was a horn. It was coming from the sky.

The phrase "It was a horn" gets stuck — at low temperature, the most likely next token after "It was a horn." is often "It was a horn." again. Higher temperatures (~0.9+) prevent this by sampling alternative continuations.

---

## What the Numbers Tell Us

Across ~50 test prompts:

| Prompt Type | Avg Confidence | Avg Entropy | Perplexity |
|---|---|---|---|
| In-domain (stories, dialogue) | 0.65–0.77 | 0.8–1.2 | 1.4–1.7 |
| Edge of distribution (knights, dragons) | 0.45–0.60 | 1.5–2.3 | 1.8–2.6 |
| Out-of-domain (technical, professional) | 0.40–0.55 | 2.0–2.7 | 2.5–3.1 |
| Non-English | 0.14–0.26 | 3.6–6.0 | 4.7–9.2 |

The model is **calibrated** — its confidence numbers honestly reflect the in-distribution/out-of-distribution gap. This is what you want in a language model: it should know when it doesn't know.