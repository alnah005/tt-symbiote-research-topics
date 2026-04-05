# Agent B Review: Chapter 7 — Pass 1

## Issue 1 — Audio feature extractor default parameters are wrong (Section 7.5.1)

The chapter states defaults of `frame_length_ms=20.0`, `min_frequency=0.0`, `max_frequency=8000.0`, `preemphasis=0.0`, `fft_overdrive=False`, and `mel_floor=0.001`. The actual `Gemma3nAudioFeatureExtractor` source (`feature_extraction_gemma3n.py` lines 163-170) shows `frame_length_ms=32.0`, `min_frequency=125.0`, `max_frequency=7600.0`, `preemphasis=0.97`, `fft_overdrive=True`, and `mel_floor=1e-5`. These are not minor rounding differences; they produce entirely different spectrograms and derived computations. The chapter's claim that `frame_length = 320` samples (Section 7.5.2) is also wrong; at 32ms it is `int(round(16000 * 32 / 1000)) = 512`, and with `fft_overdrive=True` the FFT length becomes 1024, not 512.

## Issue 2 — Spectrogram pipeline falsely claims semicausal padding (Section 7.5.3, step 3)

The chapter states: "Prepend `frame_length // 2 = 160` zeros so that the first STFT frame is centered at t=0." The actual `_extract_spectrogram` method (lines 224-267) does no such padding. It calls `_unfold` directly on the (possibly dithered/scaled) waveform without any prepending. The semicausal padding claim also feeds into the mel frame count formula in Section 7.6, making that formula incorrect for the actual code.

## Issue 3 — Frame-level masking description is incorrect (Section 7.5.3, step 12)

The chapter claims: "A mel frame is valid only when all samples in its analysis window are real audio (not padding)." The actual implementation (line 265) simply subsamples the attention mask at hop_length intervals: `mask = attention_mask[::self.hop_length]`, then truncates to match spectrogram length. This is a coarse approximation that checks one sample per hop, not all samples in the analysis window.

## Issue 4 — Audio token count formula in Section 7.6 is wrong

The mel frame count formula `T = (N + 160 - 321) // 160 + 1` is derived from the incorrect frame_length (320 instead of 512) and from the nonexistent semicausal padding step. With the actual parameters (frame_length=512, no prepend padding, frame_size_for_unfold=513, hop_length=160), the correct formula from `_unfold` is `T = (N - 513) // 160 + 1`. The worked example (10-second clip) therefore produces wrong numbers. Additionally, the `_compute_audio_num_tokens` method referenced in the chapter would need to use these corrected values.

## Issue 5 — Mel filterbank construction method is misattributed (Section 7.5.2)

The chapter states the mel filterbank is built via `mel_filter_bank(...)` from `transformers.audio_utils` with `mel_scale="htk"`. The actual code defines and uses a local function `create_fb_matrix` (lines 30-88 of `feature_extraction_gemma3n.py`) that implements HTK mel scaling directly. It does not call the transformers built-in utility. The function signature and internal logic differ (e.g., it takes `fft_length` as a parameter and computes `n_freqs` frequency bins from it internally via `sample_rate / fft_length`).

---

# Agent B Review: Chapter 7 — Pass 2

Re-verified all 5 Pass 1 claims against the actual source code in `feature_extraction_gemma3n.py` and `processing_gemma3n.py`. All 5 are confirmed genuine.

## Confirmation of Issue 1 — Audio feature extractor defaults are wrong (Section 7.5.1)

Confirmed. Source `__init__` (lines 162-170) shows: `frame_length_ms=32.0`, `min_frequency=125.0`, `max_frequency=7600.0`, `preemphasis=0.97`, `fft_overdrive=True`, `mel_floor=1e-5`. The chapter states `frame_length_ms=20.0`, `min_frequency=0.0`, `max_frequency=8000.0`, `preemphasis=0.0`, `fft_overdrive=False`, `mel_floor=0.001`. Every one of these six defaults is wrong. This cascades into Section 7.5.2 where `frame_length=320` should be `512`, and `fft_length=512` should be `1024` (because `fft_overdrive=True` doubles the next-power-of-two).

## Confirmation of Issue 2 — No semicausal padding in spectrogram extraction (Section 7.5.3, step 3)

Confirmed. The `_extract_spectrogram` method (lines 224-267) calls `_unfold` directly on the waveform with no prepended zeros. The chapter's claim of prepending `frame_length // 2` zeros is fabricated. This also invalidates the mel frame count formula in Section 7.6 which includes a `+ 160` term for padding that does not exist.

## Confirmation of Issue 3 — Frame-level masking description is wrong (Section 7.5.3, step 12)

Confirmed. Source line 265: `mask = attention_mask[:: self.hop_length].astype(bool)` followed by truncation to spectrogram length. The chapter claims the code indexes the attention mask at `i * hop_length + frame_size_for_unfold - 1` (the last sample of each analysis window). The actual code simply subsamples the attention mask at hop_length intervals starting from index 0, which is a coarse approximation, not the precise window-boundary check the chapter describes.

## Confirmation of Issue 4 — Audio token count formula is wrong (Section 7.6)

Confirmed. With the correct parameters (`frame_length=512`, `frame_size_for_unfold=513`, `hop_length=160`, no semicausal padding), the mel frame formula should be `T = (N - 513) // 160 + 1`. The chapter gives `T = (N + 160 - 321) // 160 + 1`, which uses the wrong `frame_size_for_unfold` (321 instead of 513) and includes a phantom `+160` padding term. The worked example for a 10-second clip is consequently wrong: the correct mel frame count is `(160000 - 513) // 160 + 1 = 998`, not 999.

## Confirmation of Issue 5 — Mel filterbank construction is misattributed (Section 7.5.2)

Confirmed. Source lines 30-88 and 204-212: the code defines and uses a local `create_fb_matrix` function. The chapter claims it calls `mel_filter_bank(num_frequency_bins=257, num_mel_filters=128, ..., mel_scale="htk")` from `transformers.audio_utils`. The actual function has a different signature (`n_freqs`, `n_mels`, `fft_length`) and is not the transformers built-in utility. Additionally, with corrected parameters the frequency bin count is `1024 // 2 + 1 = 513`, not 257.
