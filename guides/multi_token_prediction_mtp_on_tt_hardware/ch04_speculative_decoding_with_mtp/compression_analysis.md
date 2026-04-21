## C Analysis — Pass 1

One crucial inaccuracy found: `acceptance_rate_estimation.md` lines 15 and 23 were self-contradictory — line 15 said "DeepSeek-V3 reports the following acceptance rates for its MTP head in speculative decoding" while line 23 said speculative decoding use "has not been confirmed from the technical report." The table framing attributed unverified values to the report as confirmed spec-decoding data.

Applied: Renamed section "Estimated Values by Domain"; reframed values as illustrative estimates "not directly cited from a specific published measurement."

**Crucial updates: yes** (fix above applied)

---

## C Analysis — Pass 2

All crucial inaccuracies from Pass 1 resolved. Verified:
- E[tokens/cycle] = (1-α^{K+1})/(1-α); K=1 = 1+α — correct throughout
- Speedup = (1+α)/2 < 1 at batch 1 — correct
- x_t1 from target backbone (p); no accept/reject for x_t1 — correct
- MTP gating condition (`labels is not None AND self.training is True`) — correct
- No-resample rejection path in mtp_as_draft_model.md consistent with E=1+α model
- 94 backbone layers; 160M params / 304.6 MiB BF16 — correct
- `index.md` "Literature values" label noted as residual inconsistency (minor, not crucial)

**Crucial updates: no**

---

## C Analysis — Pass 3

Two fixes applied between C pass 2 and C pass 3 (B pass 3 and 4 fixes):
- `acceptance_rate_estimation.md` body text: "architecture similarity to DeepSeek-V3" removed; now "joint training approach and domain-dependence of token predictability"
- `throughput_analysis_on_tt_hardware.md`: `(1 - 0.41)` corrected to `(1 - 0.4096)`

All files verified against ground truth — no remaining crucial inaccuracies.

**Crucial updates: no**
