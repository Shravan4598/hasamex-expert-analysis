# Evaluation

This directory contains the evaluation assets for the Hasamex Expert Analysis application.

The evaluation is designed to test whether the application can:

- retrieve relevant transcript evidence;
- answer questions using only available evidence;
- preserve expert attribution;
- preserve source-file attribution;
- preserve timestamps;
- provide traceable evidence;
- provide citations;
- identify unsupported questions instead of inventing information;
- handle cross-expert questions;
- expose differences between expert perspectives.

The evaluation is intentionally grounded in the three supplied expert transcripts.

---

## 1. Evaluation Philosophy

The goal is not to produce an artificial benchmark score.

The goal is to verify that the application behaves safely and transparently when working with a small expert-transcript corpus.

A response should not receive credit merely because it sounds plausible.

For an answer to be considered grounded, the system should provide:

1. Relevant evidence.
2. Source information.
3. Expert attribution.
4. Timestamp information.
5. A traceable quote or source text.
6. Evidence sufficient to support the answer.
7. Citations where applicable.

If the transcripts do not contain enough information, the expected behavior is to say that the evidence is insufficient.

---

# 2. Evaluation Files

```text
evaluation/
├── questions.json
├── README.md
└── results.json          # Generated locally, not committed