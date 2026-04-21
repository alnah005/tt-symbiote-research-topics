## Pass 2

No feedback — chapter approved.

---

## Pass 1

1. **`vision_encoder_comparison.md`, line 72 — Wrong projection parameter count.**
   The file states the `[1152, 2048]` projection weight contributes "approximately 2.4M parameters." The correct value is 1152 × 2048 = 2,359,296 ≈ **2.36M parameters**. A reader doing a parameter budget calculation would get a wrong number.
   Fix: change "approximately 2.4M parameters" to "approximately 2.36M parameters."

2. **`vision_encoder_specs.md`, lines 67–68 — Temporal merge described as "concatenated and then projected," implying a learned temporal projection that does not exist.**
   The ground truth specifies `temporal_patch_size=2` produces T/2 temporal units × N_vision_per_frame spatial tokens with no separate learned projection for the merge step — the only learned projection is the shared 1152 → 2048 linear applied afterward. The parenthetical "(concatenated and then projected)" misleads an implementer into inserting a dedicated learned temporal projection module between frame encoding and the final linear, which is architecturally incorrect.
   Fix: remove the parenthetical and describe the merge plainly, e.g., "every two temporally adjacent frame token sequences are merged into a single temporal unit, halving the temporal token count."
