# B Review — Chapter 3 — Pass 1

1. **`token_padding_and_dispatch.md` — expert-to-device mapping example is arithmetically wrong.**
   Line 104 states: "A token assigned to expert 37 (device 4 in an 8-device setup with 128 experts uniformly distributed: 128/8=16 experts per device) is sent … to device 4."
   With 16 experts per device and zero-based device indexing, expert 37 maps to `floor(37 / 16) = 2` (device 2 holds experts 32–47), not device 4 (which holds experts 64–79). A reader using this example as a model for computing expert-to-device routing will produce incorrect device indices.

# B Review — Chapter 3 — Pass 2

1. **`weight_application_overhead.md` line 190 — tensor size stated as 16 bytes, should be 4 bytes.**
   The text reads: "with the reshape approach, `w` is a `(2, 1, 1, 1)` tensor (16 bytes)." A `(2, 1, 1, 1)` bf16 tensor has 2 elements × 2 bytes = **4 bytes**, not 16 bytes. A reader computing the memory-allocation saving of the alternative implementation (which is presented as the motivation for the change) will calculate the wrong reduction ratio: the document implies the broadcast saves from 16 KB down to 16 bytes (1000×), whereas the correct saving is from 16 KB down to 4 bytes. The ratio is still large, but the stated baseline for the alternative path is off by 4×.

# B Review — Chapter 3 — Pass 3

No feedback — chapter approved.

# B Review — Chapter 3 — Pass 4

No feedback — chapter approved.
