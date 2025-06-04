# Open World Object Detection in the Era of Foundation Models

Working on detecting unknown objects based on their shared attributes with known objects.
The goal is to develop methods robust enough to detect and incrementally learn unknown
objects in real-world scenarios.



##  Overview

This project introduces **FOMO (Foundation Object detection Model for the Open world)** — a new approach for detecting unknown objects in real-world applications by leveraging foundation vision-language models (e.g., OWL-ViT, CLIP) and attribute-based reasoning.

We also introduce a **challenging open-world benchmark** built from five diverse application domains: aquatic, aerial, game, medical, and surgical datasets.

> Existing open-world benchmarks (e.g., M-OWODB, S-OWODB) are saturated by foundation models.  
> Our RWD benchmark offers realistic, domain-shifted challenges that push current models to their limits.

---

## Key Contributions

-  **Critique of current benchmarks**: Existing OWD setups are too easy for models like OWL-ViT.
-  **RWD Benchmark**: A new real-world open-world detection benchmark across 5 challenging domains.
-  **FOMO**: A novel model that uses language-generated attributes + visual alignment to identify unknowns.
-  Achieves up to **3× improvement** in unknown object mAP on the RWD benchmark.

---

##  Tasks

| Task | Description |
|------|-------------|
| **Open-World Detection (OWD)** | Detect known objects, identify unknown ones, and continually learn. |
| **Few-Shot Generalization** | Use a small number of examples to learn new classes. |
| **Attribute-Based Unknown Discovery** | Leverage shared visual/functional traits to detect unseen objects. |

---



