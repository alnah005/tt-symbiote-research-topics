# Agent B Review — Chapter 1: T3K Hardware Topology and Interconnect Fundamentals — Pass 1

1. **`topology_implications_for_collectives.md`, line 52 — Wrong denominator label causes incorrect reproduction of average routing distance**

   The bidirectional ring row of the hop-count table contains this parenthetical: "denominator = 28 ordered pairs, always taking shorter path: Σd(8−d)/28 = 84/28 = 3.0". The number 28 is the count of *unordered* source–destination pairs on an 8-node chain; the count of *ordered* pairs is 56. A reader who takes the label "ordered pairs" at face value and attempts to verify the formula will compute either 84 ÷ 56 = 1.5 (wrong, using stated sum 84 with corrected ordered-pair count 56) or 168 ÷ 28 = 6.0 (wrong, using the actual ordered-pair sum 168 with the stated count 28). Neither reproduces the claimed 3.0. The formula and result are themselves correct — Σ_{d=1}^{7} d·(8−d) = 84, divided by the 28 unordered pairs, equals 3.0 — but the label "ordered pairs" directly contradicts the denominator used. Fix: change "denominator = 28 ordered pairs" to "denominator = 28 unordered pairs".

No further correctness issues were found. All bandwidth figures (12.5 GB/s per link, 50 GB/s per device pair at num_links=4, 350 GB/s aggregate across seven link groups) are arithmetically consistent with the 100 Gb/s physical-layer specification. All latency estimates are consistent with the 1.7 µs pipeline floor and the stated serialization formula. The ceil((N−1)/2) = 4 round and hop figures for bidirectional ring on N=8 are correct. The tree all-reduce round count (4, equal to the longest gather path device 7→6→5→4→3) is correct. The cluster_axis=1 assignment for the column axis of a (1,8) mesh is correct. All four planned files are present.

# Agent B Review — Chapter 1: T3K Hardware Topology and Interconnect Fundamentals — Pass 2

1. **`topology_implications_for_collectives.md`, "Note on denominators" paragraph — residual "ordered pairs" label contradicts the denominator used**

   The note immediately below the ring all-to-all hop-count table contains the sentence: "the bidirectional 3.0 figure is the average over all 28 **ordered** source–destination pairs taking the shorter of the two ring directions." The label "28 ordered" is factually wrong: 28 is the count of *unordered* pairs on an 8-node chain (Σ_{d=1}^{7}(8−d) = 28). The count of ordered source–destination pairs is 8×7 = 56. A reader who takes "28 ordered" at face value and attempts to enumerate 28 ordered pairs will either conclude that some ordered pairs are excluded (confusing) or sum the 56 actual ordered-pair distances (168) and divide by 56 to get 3.0 — which does work numerically, but then does not match the stated denominator of 28. Using the stated count of 28 with the sum for ordered pairs (168) yields 168/28 = 6.0, not 3.0. The formula and result are correct when interpreted as unordered pairs, but the "ordered" label directly contradicts the denominator. This is the same category of error that was fixed in the table cell in Pass 1; it survived in the note. Fix: change "28 ordered source–destination pairs" to "28 unordered source–destination pairs."

# Agent B Review — Chapter 1: T3K Hardware Topology and Interconnect Fundamentals — Pass 3

**No feedback — chapter approved.**

# Agent B Review — Chapter 1: T3K Hardware Topology and Interconnect Fundamentals — Pass 4

**No feedback — chapter approved.**
