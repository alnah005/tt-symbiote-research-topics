# Chapter 7: Preprocessing Pipelines

This chapter covers the preprocessing stack that transforms raw images, video, and audio into the tensor inputs consumed by `Gemma4ForConditionalGeneration` (see [Chapter 6](../ch6_top_level_model_assembly/index.md)). The four key classes live in separate files:

| Class | File | Backend |
|---|---|---|
| `Gemma4Processor` | [`processing_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/processing_gemma4.py) | Orchestrator |
| `Gemma4ImageProcessor` | [`image_processing_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/image_processing_gemma4.py) | torchvision |
| `Gemma4ImageProcessorPil` | [`image_processing_pil_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/image_processing_pil_gemma4.py) | PIL / NumPy |
| `Gemma4VideoProcessor` | [`video_processing_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/video_processing_gemma4.py) | torchvision |
| `Gemma4AudioFeatureExtractor` | [`feature_extraction_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/feature_extraction_gemma4.py) | NumPy |

---

## 7.1 Gemma4Processor -- The Orchestrator

```python
class Gemma4Processor(ProcessorMixin):
    def __init__(self, feature_extractor, image_processor, tokenizer, video_processor,
                 chat_template=None, image_seq_length=280, audio_seq_length=750,
                 audio_ms_per_token=40, **kwargs)
```

`Gemma4Processor` extends `ProcessorMixin` and is the single entry point for preparing multimodal inputs. It holds references to four sub-processors:

| Attribute | Type | Purpose |
|---|---|---|
| `self.tokenizer` | `PreTrainedTokenizer` | Text tokenization |
| `self.image_processor` | `Gemma4ImageProcessor` or `Gemma4ImageProcessorPil` | Image preprocessing |
| `self.video_processor` | `Gemma4VideoProcessor` | Video preprocessing |
| `self.feature_extractor` | `Gemma4AudioFeatureExtractor` | Audio mel spectrogram extraction |

### 7.1.1 Token Expansion Pattern

The processor's `__call__` method follows a uniform expansion pattern for each modality: detect placeholder tokens in the text, delegate to the modality-specific processor, then use `re.sub` to replace each placeholder with the correct number of soft tokens bracketed by begin/end markers.

**Image expansion:**

```
<image> --> <start_of_image><image_soft><image_soft>...<end_of_image>
```

The number of `<image_soft>` tokens equals `num_soft_tokens_per_image` returned by the image processor, which varies per image based on aspect ratio. The processor pops this metadata from image outputs before returning.

**Video expansion:**

```
<|video|> --> 00:00 <start_of_image><|video|><|video|>...<end_of_image> 00:04 <start_of_image>...
```

Each frame gets a timestamp prefix in `MM:SS` format derived from `video_metadata.timestamps` and the video's `fps`. The soft tokens per frame equal `num_soft_tokens_per_video`. If `fps` is unknown (e.g., pre-sampled frames without metadata), it defaults to 24.

**Audio expansion:**

```
<audio> --> <start_of_audio><audio_soft><audio_soft>...<end_of_audio>
```

The number of audio soft tokens is computed dynamically per waveform (see Section 7.1.2).

### 7.1.2 Dynamic Audio Token Count Computation

The method `_compute_audio_num_tokens(audio_waveform, sampling_rate)` replicates the exact sequence-length arithmetic of the audio encoder so that the processor inserts precisely the right number of placeholder tokens. The computation mirrors:

1. **Mel framing** (matching `_unfold` in `Gemma4AudioFeatureExtractor`):
   ```python
   frame_length = int(round(sampling_rate * 20.0 / 1000.0))  # 320 @ 16kHz
   hop_length = int(round(sampling_rate * 10.0 / 1000.0))    # 160 @ 16kHz
   frame_size_for_unfold = frame_length + 1                    # 321

   pad_left = frame_length // 2  # 160 -- semicausal padding
   padded_samples = num_samples + pad_left
   num_mel_frames = (padded_samples - frame_size_for_unfold) // hop_length + 1
   ```

2. **Two SSCP convolution layers** (each: kernel=3, stride=2, semicausal pad top=1, bottom=1):
   ```python
   t = num_mel_frames
   for _ in range(2):
       t_padded = t + 2        # pad_top=1, pad_bottom=1
       t = (t_padded - 3) // 2 + 1
   ```

3. **Cap** at `self.audio_seq_length` (default 750).

This mirrors the `Gemma4AudioSubSampleConvProjection` described in [Chapter 4](../ch4_audio_encoder/index.md). The `audio_ms_per_token` parameter (default 40) reflects the 4x time reduction: 10ms mel frames reduced by two stride-2 convolutions yields one token per 40ms of audio.

### 7.1.3 Multimodal Token Type IDs

After tokenization, the processor optionally generates `mm_token_type_ids` via `create_mm_token_type_ids(input_ids)`. This tensor marks which tokens are multimodal soft tokens (non-zero) versus regular text tokens (zero), enabling the model to apply modality-specific logic during embedding (see [Chapter 6, Section 6.5](../ch6_top_level_model_assembly/index.md)).

### 7.1.4 Default Processing Kwargs

```python
class Gemma4ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": True, "return_mm_token_type_ids": True},
        "images_kwargs": {"do_convert_rgb": True},
        "audio_kwargs": {},
        "videos_kwargs": {"return_metadata": True},
    }
```

### 7.1.5 `_get_num_multimodal_tokens`

This utility method computes placeholder token counts without processing actual data -- useful for pre-allocating sequence lengths. For images, it calls `get_aspect_ratio_preserving_size` to compute per-image token counts from `image_sizes`. For audio, it creates zero-filled dummy waveforms of the given `audio_lengths` and routes through `_compute_audio_num_tokens`.

---

## 7.2 Gemma4ImageProcessor (Torchvision Backend)

```python
class Gemma4ImageProcessor(TorchvisionBackend):
    patch_size = 16
    max_soft_tokens = 280
    pooling_kernel_size = 3
    resample = PILImageResampling.BICUBIC
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]
    do_normalize = False
    model_input_names = ["pixel_values", "image_position_ids", "num_soft_tokens_per_image"]
```

This is the primary image processor, using torchvision for GPU-accelerated transforms. Identity normalization (mean=0, std=1) means pixel values stay in [0, 1] after rescaling -- the model was trained on this range.

### 7.2.1 Aspect-Ratio-Preserving Resize

The core resize logic lives in `get_aspect_ratio_preserving_size`, a shared function defined in `image_processing_pil_gemma4.py` and imported by both backends:

```python
def get_aspect_ratio_preserving_size(height, width, patch_size, max_patches, pooling_kernel_size):
    total_px = height * width
    target_px = max_patches * (patch_size ** 2)
    factor = math.sqrt(target_px / total_px)
    side_mult = pooling_kernel_size * patch_size  # 3 * 16 = 48

    target_height = int(math.floor(factor * height / side_mult)) * side_mult
    target_width  = int(math.floor(factor * width  / side_mult)) * side_mult
    return target_height, target_width
```

The algorithm:
1. Compute `max_patches = max_soft_tokens * pooling_kernel_size^2` (e.g., 280 * 9 = 2520 patches).
2. Find a scaling factor that makes `height * width` equal `max_patches * patch_size^2` target pixels.
3. Round both dimensions **down** to the nearest multiple of `side_mult = pooling_kernel_size * patch_size` (48 pixels with defaults).
4. Handle edge cases where one dimension rounds to zero by clamping to `side_mult` and constraining the other dimension.

This guarantees that the resized image is evenly divisible by both `patch_size` (for patchification) and `pooling_kernel_size` (for the vision encoder's spatial pooling).

### 7.2.2 Patchification

```python
def convert_image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    # (C, H, W) -> (num_patches_h * num_patches_w, patch_size * patch_size * C)
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(C, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.permute(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image
```

Each 16x16x3 pixel patch becomes a flat vector of 768 values. For a 280-soft-token image with pooling_kernel_size=3, there are 2520 raw patches.

### 7.2.3 2D Position ID Generation

After patchification, the processor creates 2D position IDs for each patch using `torch.meshgrid`:

```python
patch_grid = torch.meshgrid(
    torch.arange(patch_width),
    torch.arange(patch_height),
    indexing="xy",
)
real_positions = torch.stack(patch_grid, dim=-1).reshape(num_patches, 2)
```

Each patch gets a `(x, y)` coordinate pair. These position IDs are fed to the vision encoder's rotary position embedding (see [Chapter 3](../ch3_vision_encoder/index.md)).

### 7.2.4 Padding to Max Patches

All images in a batch are padded to `max_patches` along the patch dimension using `pad_along_first_dim`. Padding positions get position IDs of `(-1, -1)`, which the vision encoder uses to identify and mask out padding.

### 7.2.5 Output Tensors

The `_preprocess` method returns a `BatchFeature` with:

| Key | Shape | Description |
|---|---|---|
| `pixel_values` | `(batch, max_patches, patch_pixels)` | Flattened patches, `patch_pixels = 16*16*3 = 768` |
| `image_position_ids` | `(batch, max_patches, 2)` | 2D `(x, y)` grid coordinates; `-1` for padding |
| `num_soft_tokens_per_image` | `list[int]` | Per-image count of actual (non-padding) soft tokens after pooling |

### 7.2.6 Supported Soft Token Counts

Both image processors validate that `max_soft_tokens` is one of:

```python
_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)
```

Each value corresponds to a different resolution tier. The default is 280. The relationship between soft tokens and raw patches is:

| `max_soft_tokens` | `max_patches` (with pooling=3) | Approximate resolution |
|---|---|---|
| 70 | 630 | ~403 px |
| 140 | 1260 | ~570 px |
| 280 | 2520 | ~806 px |
| 560 | 5040 | ~1140 px |
| 1120 | 10080 | ~1612 px |

---

## 7.3 Gemma4ImageProcessorPil (PIL Backend)

```python
class Gemma4ImageProcessorPil(PilBackend):
    patch_size = 16
    max_soft_tokens = 280
    pooling_kernel_size = 3
    rescale_factor = 1 / 255
```

This is a CPU-only alternative that operates on NumPy arrays instead of torch tensors. It is functionally identical to `Gemma4ImageProcessor` but:

- Uses `np.ndarray` throughout instead of `torch.Tensor`
- Uses HuggingFace's built-in `resize()` (PIL-backed) instead of `torchvision.transforms.v2.functional.resize`
- Uses `np.meshgrid` for position IDs
- Uses `np.pad` instead of `torch.nn.functional.pad`
- Calls `self.rescale(image, scale=rescale_factor)` and `self.normalize()` as separate explicit steps, whereas the torchvision backend combines them in `rescale_and_normalize`

The `_preprocess` pipeline is step-for-step identical:
1. Aspect-ratio-preserving resize
2. Rescale [0, 255] to [0, 1]
3. Identity normalization (no-op with mean=0, std=1)
4. Patchify via `convert_image_to_patches` (NumPy version)
5. Compute 2D position IDs
6. Pad to `max_patches`
7. Stack into batch

Both backends share `get_aspect_ratio_preserving_size` -- the torchvision backend imports it from the PIL module.

---

## 7.4 Gemma4VideoProcessor

```python
class Gemma4VideoProcessor(BaseVideoProcessor):
    num_frames = 32
    do_sample_frames = True
    patch_size = 16
    max_soft_tokens = 70          # Note: 70 for video, vs. 280 for images
    pooling_kernel_size = 3
    model_input_names = ["pixel_values_videos", "video_position_ids"]
```

### 7.4.1 Key Differences from Image Processing

The video processor extends `BaseVideoProcessor` and reuses the same patchification and resize logic as the image processor, with several important differences:

1. **Default `max_soft_tokens = 70`** (not 280), giving each frame a much lower resolution budget since videos have many frames.
2. **Frame sampling**: `BaseVideoProcessor` handles frame extraction with `num_frames=32` and `do_sample_frames=True`.
3. **4D tensors**: Videos are `(num_frames, C, H, W)` rather than `(C, H, W)`.

### 7.4.2 `convert_video_to_patches`

```python
def convert_video_to_patches(video: torch.Tensor, patch_size: int) -> torch.Tensor:
    # (num_frames, C, H, W) -> (num_frames, num_patches_h * num_patches_w, patch_pixels)
```

This is the batched-frame analog of `convert_image_to_patches`. It reshapes the entire video at once, preserving the frame dimension.

### 7.4.3 Per-Frame Position IDs

Position IDs are computed once for the spatial grid and then broadcast across all frames:

```python
real_positions = stacked_grid.reshape(patches.shape[1], 2)
real_positions = real_positions[None, ...].repeat(num_frames, 1, 1)
```

Every frame shares the same 2D spatial position IDs. Temporal ordering is handled at the prompt level via timestamp strings (see Section 7.1.1).

### 7.4.4 Padding

`pad_to_max_patches` pads along the patch dimension (dim=1) of the 3D `(num_frames, num_patches, patch_pixels)` tensor. This differs from the image version which pads along dim=0 of a 2D tensor.

### 7.4.5 Output Tensors

| Key | Shape | Description |
|---|---|---|
| `pixel_values_videos` | `(num_videos, num_frames, max_patches, patch_pixels)` | Flattened patches per frame |
| `video_position_ids` | `(num_videos, num_frames, max_patches, 2)` | 2D spatial coordinates per frame |
| `num_soft_tokens_per_video` | `list[int]` | Soft tokens per frame (same for all frames in a video) |

---

## 7.5 Gemma4AudioFeatureExtractor

```python
class Gemma4AudioFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_features", "input_features_mask"]
```

This class implements USM-style (Universal Speech Model) mel spectrogram extraction, with non-standard windowing that differs from the built-in `transformers.audio_utils.spectrogram()`.

### 7.5.1 Core Parameters

| Parameter | Default | Description |
|---|---|---|
| `feature_size` | 128 | Number of mel bins |
| `sampling_rate` | 16000 | Expected input sample rate (Hz) |
| `frame_length_ms` | 20.0 | Frame length: `int(round(16000 * 20 / 1000)) = 320` samples |
| `hop_length_ms` | 10.0 | Hop length: `int(round(16000 * 10 / 1000)) = 160` samples |
| `min_frequency` | 0.0 | Mel filterbank lower edge (Hz) |
| `max_frequency` | 8000.0 | Mel filterbank upper edge (Hz) |
| `preemphasis` | 0.0 | Preemphasis coefficient (disabled by default) |
| `preemphasis_htk_flavor` | True | HTK-style preemphasis if enabled |
| `fft_overdrive` | False | Double the FFT length if True |
| `dither` | 0.0 | Gaussian dither amplitude (disabled by default) |
| `input_scale_factor` | 1.0 | Waveform scaling factor |
| `mel_floor` | 0.001 | Floor value before log to avoid log(0) |
| `per_bin_mean` / `per_bin_stddev` | None | Optional per-bin normalization |

### 7.5.2 Derived Parameters

At construction time, several values are computed from the above:

```python
self.frame_length = 320                        # samples (20ms @ 16kHz)
self.hop_length = 160                          # samples (10ms @ 16kHz)
self.fft_length = 2 ** ceil(log2(320)) = 512   # next power of 2
```

The mel filterbank is built via `mel_filter_bank(num_frequency_bins=257, num_mel_filters=128, min_frequency=0.0, max_frequency=8000.0, sampling_rate=16000, norm=None, mel_scale="htk")`. The window is a periodic Hann window of length 320, matching `sl.STFT` defaults.

### 7.5.3 Spectrogram Extraction Pipeline

The `_extract_spectrogram(waveform, attention_mask)` method implements the full pipeline:

1. **Optional dithering**: If `dither > 0`, add Gaussian noise to reduce artifacts from hard-zero sections.
2. **Input scaling**: Multiply by `input_scale_factor`.
3. **Semicausal padding**: Prepend `frame_length // 2 = 160` zeros so that the first STFT frame is centered at t=0.
4. **Framing via `_unfold`**: Extract overlapping frames of size `frame_length + 1 = 321` with stride `hop_length = 160`. The `_unfold` function uses NumPy stride tricks for zero-copy windowing.
5. **HTK preemphasis** (if enabled): Apply first-order high-pass filter within each frame.
6. **Windowing**: Multiply each frame by the periodic Hann window (truncated to `frame_length = 320` samples).
7. **RFFT**: Compute `np.fft.rfft` with `n=fft_length=512`, producing 257 frequency bins.
8. **Magnitude spectrum**: Take absolute value.
9. **Mel projection**: Matrix multiply by `mel_filters` (257 x 128) to get 128 mel bins.
10. **Log compression**: `log(mel_spec + mel_floor)`.
11. **Optional per-bin normalization**: Subtract `per_bin_mean`, divide by `per_bin_stddev`.
12. **Frame-level masking**: For each mel frame `i`, the code indexes the (padded) attention mask at `i * hop_length + frame_size_for_unfold - 1` -- the last sample of that frame's analysis window. If that sample is valid (non-zero), the frame is marked valid. This is equivalent to checking all samples for contiguous audio with trailing padding, since the last sample is the first to fall into the padded region.

### 7.5.4 The `__call__` Method

```python
def __call__(self, raw_speech, padding="longest", max_length=480_000,
             truncation=True, pad_to_multiple_of=128, ...):
```

Key behaviors:
- **Padding**: Default `"longest"` pads to the longest waveform in the batch. `pad_to_multiple_of=128` ensures TPU-friendly lengths.
- **Truncation**: Default `max_length=480_000` samples = 30 seconds at 16kHz.
- **Batched processing**: Each waveform is independently spectrogrammed, then masked by `input_features_mask` (element-wise multiply with the mask broadcasted across mel bins).

### 7.5.5 Output Tensors

| Key | Shape | Description |
|---|---|---|
| `input_features` | `list[ndarray]` of shape `(num_mel_frames, 128)` | Log mel spectrograms, masked |
| `input_features_mask` | `list[ndarray]` of shape `(num_mel_frames,)` | Boolean frame validity masks |

---

## 7.6 How Audio Token Count Mirrors Encoder Arithmetic

The audio preprocessing pipeline and the audio encoder ([Chapter 4](../ch4_audio_encoder/index.md)) must agree on exactly how many tokens a given waveform produces. This is critical because the processor must insert the right number of placeholder tokens before tokenization.

The chain of transformations and their effect on the time dimension:

```
Raw waveform:  N samples
    |
    v  (semicausal pad + unfold, hop=160)
Mel frames:    T = (N + 160 - 321) // 160 + 1
    |
    v  (SSCP Conv layer 1: kernel=3, stride=2, pad=1+1)
After conv1:   T1 = (T + 2 - 3) // 2 + 1 = (T - 1) // 2 + 1
    |
    v  (SSCP Conv layer 2: kernel=3, stride=2, pad=1+1)
After conv2:   T2 = (T1 + 2 - 3) // 2 + 1 = (T1 - 1) // 2 + 1
    |
    v  (cap at audio_seq_length=750)
Soft tokens:   min(T2, 750)
```

The processor's `_compute_audio_num_tokens` replicates this exact arithmetic. The `audio_ms_per_token=40` parameter is a convenience approximation: 10ms mel frames with 4x reduction (two stride-2 convs) yields ~40ms per token. However, the actual token count uses the precise formula above, not this approximation.

**Example**: A 10-second audio clip at 16kHz = 160,000 samples:
- Mel frames: `(160000 + 160 - 321) // 160 + 1 = 999`
- After conv1: `(999 - 1) // 2 + 1 = 500`
- After conv2: `(500 - 1) // 2 + 1 = 250`
- Result: 250 soft tokens (well under the 750 cap)

---

## 7.7 TTNN Porting Considerations

### Image/Video Preprocessing -- Keep on Host

The image and video processors perform variable-resolution resizing, patchification, and position ID generation. These are data-dependent operations with branching logic that run once per input during preprocessing. They should remain on the CPU host. The output tensors (`pixel_values`, `image_position_ids`, `pixel_values_videos`, `video_position_ids`) are then transferred to device for the vision encoder.

### Audio Feature Extraction -- Keep on Host

The mel spectrogram extraction involves NumPy FFT operations, stride tricks (`_unfold`), and mel filterbank projection. This is inherently a host-side operation. The output `input_features` tensors are transferred to device for the audio encoder.

### Token Count Computation -- Critical for Correctness

The `_compute_audio_num_tokens` method must produce counts that exactly match the audio encoder's output length. Any mismatch causes a shape error during the multimodal embedding merge in `Gemma4MultimodalEmbedder` (see [Chapter 6](../ch6_top_level_model_assembly/index.md)). When porting the audio encoder to TTNN, verify that the SSCP convolution output lengths match the formulas in Section 7.6.

### Patchification as a Reshape

The `convert_image_to_patches` and `convert_video_to_patches` functions are pure reshape/permute operations with no learned parameters. On TTNN, the vision encoder's `Gemma4VisionPatchEmbedder` (see [Chapter 3](../ch3_vision_encoder/index.md)) receives pre-patchified input, so these operations stay host-side.

### Variable Sequence Lengths

Different images produce different numbers of soft tokens depending on aspect ratio. The processor handles this by padding to `max_patches` and using position ID masking. On TTNN, this means the vision encoder always processes `max_patches` per image -- the effective batch size is fixed, simplifying memory allocation. The `num_soft_tokens_per_image` list is used downstream to slice the correct number of vision tokens when merging into the text sequence.

### Video Frame Budget

Video uses `max_soft_tokens=70` per frame (vs. 280 for images), yielding 630 patches per frame. With 32 frames, the total patch budget is 32 * 630 = 20,160 patches per video. This is a significant memory footprint that must be accounted for in TTNN device memory planning.

---

**Next:** [Chapter 8 -- Weight Conversion](../ch8_weight_conversion/index.md)
