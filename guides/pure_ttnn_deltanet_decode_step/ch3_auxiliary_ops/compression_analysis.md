# Compression Analysis — Chapter 3: Causal Conv1D and Gated RMSNorm Without Host Readback

## Pass 1

### Crucial issues found: 0

No verbatim or near-verbatim blocks of 5 or more lines were found across any pair of files in this directory.

Shared elements examined and ruled out:

- Both files use `ttnn.mul` in code — a single op name appearing in different contexts (depthwise convolution output vs. gated product), not a shared block.
- Both files use the same "Availability tags" and "Summary" table format — but with entirely different content rows and values. Structural formatting alone does not constitute a content block.
- Both files end availability sections with `[AVAILABLE — needs wiring]` — a single repeated line, well below the 5-line threshold.
- The `index.md` contains 1–2 sentence summaries of each content file. These are appropriate navigation entries and do not duplicate prose from the files they describe.

The two content files cover wholly separate operations (sliding-window conv state update vs. three-op RMSNorm composition) and share no algorithmic derivations, code listings, or prose explanations.

### VERDICT

Crucial updates: no

Chapter 3 compression approved.
