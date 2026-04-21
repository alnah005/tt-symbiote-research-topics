## B Feedback — Pass 1

1. **`speculative_decoding_primer.md`, "Expected Tokens Per Cycle"** — The formula labeled "E[accepted tokens]" = `(1-α^{K+1})/(1-α)` is actually E[tokens per cycle] (it includes the primary/bonus token). The file then adds 1 again, writing `E[tokens per cycle] = 1 + (1-α^{K+1})/(1-α)`, which overstates by 1 and gives 2+α for K=1 — directly contradicting the correct K=1 result `1+α` stated immediately below it. Fix: label correctly as E[tokens per cycle] = (1-α^{K+1})/(1-α); separately show E[draft tokens accepted only] = α(1-α^K)/(1-α).

2. **`speculative_decoding_primer.md`, Algorithm Step 3 / `mtp_as_draft_model.md` Step 7** — The speedup formula (1+α)/2 in the throughput analysis requires that on draft rejection only 1 token (x_t1) is produced per cycle. Resampling the draft position always produces 2 tokens per cycle, collapsing E[tokens/cycle] to 2 and speedup to 1.0. Fix: remove the resampling-on-rejection step from the algorithm so the rejected path advances context by 1 (x_t1 only).

3. **`acceptance_rate_estimation.md`, "Literature Values"** — The sentence "These values are from the DeepSeek-V3 technical report, which describes MTP being used at inference time in their production deployment" is factually unsupported. The DeepSeek-V3 technical report primarily describes MTP as an auxiliary training objective; production inference use of MTP for speculative decoding has not been confirmed from the report. Fix: remove the production deployment claim.

## B Feedback Application Log — Pass 1

- Fix 1: Relabeled the formula in `speculative_decoding_primer.md` Expected Tokens section — "E[accepted tokens]" → "E[draft tokens accepted only]" = α(1-α^K)/(1-α); "E[tokens per cycle]" = (1-α^{K+1})/(1-α) without the extra +1. The K=1 result `1+α` is now consistent with the general formula.
- Fix 2: B attributed this to `speculative_decoding_primer.md` Step 3, but the primer's standard spec decoding algorithm correctly resamples (producing the "+1" in "advance by accepted_count+1" = 1 token on rejection). The actual inconsistency was in `mtp_as_draft_model.md` Step 7, where x_t1 is an always-accepted primary token — resampling t+2 there would produce 2 tokens per rejection. Fixed `mtp_as_draft_model.md` Step 7: "Rejected path" changed from "resample t+2" to "append x_t1 only (advance by 1); do NOT resample t+2 in this cycle", with inline explanation of why omitting resampling preserves the E[tokens/cycle] = 1+α model.
- Fix 3: Changed the DeepSeek-V3 claim in `acceptance_rate_estimation.md` to note the report primarily describes MTP as a training objective and production inference use of MTP for spec decoding has not been confirmed from the report.

## B Feedback — Pass 2

1. **`mtp_as_draft_model.md`, Step 2 comment** — Comment reads "Record q(x_t1) from primary_logits for verification denominator." `x_t1` is sampled from the target backbone (`primary_logits` = target model `p`), not a draft model `q`. There is no accept/reject step for `x_t1`; it is already a valid target-model sample. The comment incorrectly implies `x_t1` participates in the accept/reject `p/q` ratio. Fix: remove or replace with "x_t1 is sampled directly from the target backbone (p); no accept/reject needed for this token."

2. **`acceptance_rate_estimation.md`, "Literature Values" section** — The table header says "DeepSeek-V3 reports the following acceptance rates for its MTP head in speculative decoding" but the same file's added caveat (Fix 3 from pass 1) says spec-decoding use has not been confirmed from the technical report. These directly contradict each other. The specific α values (0.80–0.85, etc.) are attributed as report citations for spec-decoding performance that the report does not contain. Fix: reframe the section title and table to not attribute these values to the DeepSeek-V3 report as confirmed spec-decoding measurements; present them as illustrative domain-dependence estimates.

## B Feedback Application Log — Pass 2

- Fix 1: Replaced "Record q(x_t1) from primary_logits for verification denominator" in `mtp_as_draft_model.md` Step 2 with "x_t1 is sampled directly from the target backbone (p); no accept/reject needed for this token".
- Fix 2: Renamed "Literature Values" section to "Estimated Values by Domain" in `acceptance_rate_estimation.md`; changed table framing from attributed DeepSeek-V3 citation to clearly labeled "illustrative estimates not directly cited from a published measurement"; updated α ranges slightly to reflect illustrative-only status; removed the self-contradicting introductory claim.

## B Feedback — Pass 3

1. **`acceptance_rate_estimation.md`, Key Finding, line 104** — Reads "Based on literature from the architecturally similar DeepSeek-V3, expect α ∈ [0.5, 0.8]…" — directly contradicts the corrected body of the same file, which explicitly states the DeepSeek-V3 report does not report spec-decoding acceptance rates and values are not from published measurements. Fix: remove the DeepSeek-V3 attribution from the Key Finding; reframe as domain-predictability prior.

2. **`index.md`, Files table, line 31** — Describes `acceptance_rate_estimation.md` as containing "Literature values" — after pass 2 fix the section was renamed to "Estimated Values by Domain" and explicitly disavows being literature-cited data. Fix: change to "Estimated acceptance rate ranges by domain…"

## B Feedback Application Log — Pass 3

- Fix 1: Changed Key Finding in `acceptance_rate_estimation.md` from "Based on literature from the architecturally similar DeepSeek-V3" to "Based on the domain-dependence of token predictability" to remove the DeepSeek-V3 literature attribution.
- Fix 2: Changed `index.md` Files table description for `acceptance_rate_estimation.md` from "Literature values, domain dependence, empirical measurement approach" to "Estimated acceptance rate ranges by domain, domain dependence, empirical measurement approach".

## B Feedback — Pass 4

1. **`acceptance_rate_estimation.md`, Section "Qwen3.6-35B-A3B: Expected Range", line 37** — Body text still reads "Based on the architecture similarity to DeepSeek-V3 and the joint training approach..." — residual DeepSeek-V3 attribution in the body text was not updated when the Key Finding was fixed in pass 3. Fix: remove "architecture similarity to DeepSeek-V3" attribution; reframe around joint training approach and domain-dependence of token predictability.

2. **`throughput_analysis_on_tt_hardware.md`, Higher N section, line 100** — Intermediate arithmetic shows `(1 - 0.41)` but `0.8^4 = 0.4096`, not `0.41`. Fix: change `0.41` to `0.4096`.

## B Feedback Application Log — Pass 4

- Fix 1: Changed "Based on the architecture similarity to DeepSeek-V3 and the joint training approach" to "Based on the joint training approach and the domain-dependence of token predictability" in `acceptance_rate_estimation.md` Section "Qwen3.6-35B-A3B: Expected Range".
- Fix 2: Changed `(1 - 0.41) / 0.2` to `(1 - 0.4096) / 0.2` in `throughput_analysis_on_tt_hardware.md` K=3 example.

## B Feedback — Pass 5

No feedback — chapter approved.
