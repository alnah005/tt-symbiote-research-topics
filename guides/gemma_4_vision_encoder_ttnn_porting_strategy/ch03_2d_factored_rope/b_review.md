# Agent B Review: Chapter 3

## Pass 1

### Issue 1 (High): Inverse frequency table and wavelength table use wrong values

**File:** `multidimensional_rope_theory.md`, lines 149-155

The "Inverse Frequency Table" contains incorrect omega and wavelength values for every entry except `i=0`. The correct formula is `omega_i = 1 / 100^(2i/36)` (as the chapter itself states), but the tabulated numbers do not match this formula. They appear to have been computed with a denominator of approximately 33.6 instead of 36.

| Index | Chapter omega | Correct omega | Chapter lambda | Correct lambda |
|-------|--------------|---------------|----------------|----------------|
| 0     | 1.000        | 1.000         | 6.3            | 6.3            |
| 4     | 0.278        | **0.359**     | 22.6           | **17.5**       |
| 8     | 0.077        | **0.129**     | 81.3           | **48.6**       |
| 12    | 0.021        | **0.046**     | 293            | **135.4**      |
| 17    | 0.004        | **0.013**     | 1635           | **486.5**      |

This is internally contradictory: the same file's theta comparison table (line 177) correctly states lambda_max ~ 487 for the vision encoder, yet the wavelength table claims i=17 has lambda ~ 1635.

### Issue 2 (High): Numerical example has wrong inv_freq values

**File:** `reference_implementation.md`, lines 175-193

The numerical example for patch position (5, 12) lists incorrect `inv_freq` values for `i=1,2,3`, and consequently all derived `freqs_x` and `freqs_y` values are wrong. The cos/sin outputs at lines 189-193 are correct (they use `i=0` which is unaffected).

| Index | Chapter inv_freq | Correct inv_freq |
|-------|-----------------|------------------|
| 1     | 0.760           | **0.774**        |
| 2     | 0.578           | **0.599**        |
| 3     | 0.439           | **0.464**        |

The corresponding `freqs_x` (5 * omega) and `freqs_y` (12 * omega) values on lines 182-187 propagate the same error.

### Issue 3 (Medium): Mistaken-theta lambda_max calculation is wrong

**File:** `multidimensional_rope_theory.md`, lines 183-185

The chapter states that using theta=10000 would produce `lambda_max = 2*pi * 10000^(34/36) ~ 48,700`. The actual value is:

`2 * pi * 10000^(34/36) = 37,667`

The qualitative point (wavelengths far exceeding grid size) remains valid, but the number is off by ~29%.

## Pass 2

Pass 1 issues 1-3 have been corrected. The inverse frequency table, numerical example, and mistaken-theta calculation now match the formulas.

### Issue 1 (Medium): phi vector uses wrong frequency-doubling layout for rotate_half

**File:** `multidimensional_rope_theory.md`, line 114

The expanded phi vector is written with an **interleaved** doubling pattern:

```
phi(p) = [p/theta^(0/ds), p/theta^(0/ds), p/theta^(2/ds), p/theta^(2/ds), ...]
```

This layout pairs adjacent elements `(2i, 2i+1)`, matching the complex-number rotation convention shown on lines 40-42. However, the implementation (both in the `rotate_half` definition on lines 50-53 and in `reference_implementation.md` line 141) uses `torch.cat((freqs, freqs), dim=-1)`, which produces a **concatenated** layout:

```
phi(p) = [p/theta^(0/ds), p/theta^(2/ds), ..., p/theta^((ds-2)/ds),
          p/theta^(0/ds), p/theta^(2/ds), ..., p/theta^((ds-2)/ds)]
```

This concatenated layout pairs element `j` with element `j + ds/2`, which is what `rotate_half` requires (it splits at the midpoint and swaps halves). The interleaved layout on line 114 is incorrect for the `rotate_half` scheme and would only be correct for an adjacent-pair rotation kernel.

The accompanying comment on line 117 ("each frequency is repeated twice (once for cos, once for sin in the rotate_half scheme)") is also misleading -- cos and sin are separate tensors, not interleaved within phi. The repetition exists so that each `rotate_half` pair `(j, j+ds/2)` shares the same rotation frequency.

## Pass 3

Pass 2 issue (phi vector interleaved vs. concatenated) has been fixed. The phi vector on line 114 of `multidimensional_rope_theory.md` now shows the correct concatenated layout matching `torch.cat((freqs, freqs), dim=-1)`, and the accompanying explanation on line 117 is accurate.

**No feedback — chapter approved.**
