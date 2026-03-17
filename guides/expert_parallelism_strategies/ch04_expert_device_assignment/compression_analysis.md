# Compression Analysis — Chapter 4: Expert-to-Device Assignment — Pass 1

## Summary
- Files reviewed:
  - `index.md`
  - `uniform_partitioning.md`
  - `load_aware_assignment.md`
  - `expert_replication.md`
  - `mesh_topology_constraints.md`
- Current line count (approximate):
  - `index.md`: ~113 lines
  - `uniform_partitioning.md`: ~213 lines
  - `load_aware_assignment.md`: ~432 lines
  - `expert_replication.md`: ~296 lines
  - `mesh_topology_constraints.md`: ~346 lines
  - **Total: ~1,400 lines**
- Estimated post-compression: ~1,270 lines (~130 lines removed, ~9% reduction)

---

## CRUCIAL Suggestions

**1. Duplicate $f_{\text{avg}}$ definition and normalization constraint stated identically in three files.**

- `uniform_partitioning.md` Section 4 (lines 88–89): "Under top-$k$ routing, each token selects $k$ experts, so $\sum_{e=0}^{255} f_e = k = 8$. The average frequency is $f_{\text{avg}} = k/E = 8/256 = 1/32$."
- `load_aware_assignment.md` Section 1 (lines 20–21): "The normalization satisfies $\sum_{e=0}^{E-1} f_e = k$ (since each token contributes $k = 8$ routing events). The average frequency is $f_{\text{avg}} = k/E = 8/256 = 1/32 \approx 0.03125$."
- `expert_replication.md` Section 1 (lines 22–23): the same $f_e > k/E = f_{\text{avg}} = 8/256 = 0.03125$ substitution is derived inline again.

The full definitional sentence "the normalization satisfies $\sum_e f_e = k$; average is $f_{\text{avg}} = k/E = 1/32$" is established once in `uniform_partitioning.md` and simply re-derived verbatim in the two later files. In `load_aware_assignment.md` Section 1 and `expert_replication.md` Section 1, replace the self-contained re-derivation with a single back-reference: "($f_{\text{avg}} = k/E = 1/32$; see `uniform_partitioning.md` Section 4)." No information is lost because the definition is fully established in the file readers are required to have read first.

**Estimated savings: ~4 lines across the two later files.**

---

**2. Duplicate $W_{\text{expert}}$ memory formula stated identically in two files.**

- `uniform_partitioning.md` Section 2 (lines 46–52): Full derivation: gate/up/down projection shapes, BF16 byte count, formula $W_{\text{expert}} = 3 \cdot H \cdot D \cdot 2 = 6HD$ bytes.
- `load_aware_assignment.md` Section 6.3 (lines 389–390): "where $W_{\text{expert}} = 3 \times H \times D \times 2$ bytes (BF16, three weight matrices)."
- `expert_replication.md` Section 3 (lines 72–77): "$\Delta\text{DRAM}_{\text{device}} = M \times W_{\text{expert}} = M \times 6 \times H \times D$ bytes (BF16)" with the same per-expert formula restated.

The formula $W_{\text{expert}} = 6HD$ bytes is defined and derived in `uniform_partitioning.md`. In `load_aware_assignment.md` Section 6.3 and `expert_replication.md` Section 3, replace "where $W_{\text{expert}} = 3 \times H \times D \times 2$ bytes (BF16, three weight matrices)" and its variant with "($W_{\text{expert}}$ as defined in `uniform_partitioning.md` Section 2)." No information is lost.

**Estimated savings: ~3 lines across the two later files.**

---

**3. Duplicate "static assignment requires four steps to change" content appearing in full in `mesh_topology_constraints.md` Section 9 and partially in `load_aware_assignment.md` Section 6.**

- `load_aware_assignment.md` Section 6 (lines 305–401): Describes the full lifecycle of dynamic reassignment — re-profiling, GDF re-run, migration cost estimate, shadow migration warning.
- `mesh_topology_constraints.md` Section 9 (lines 308–315): Restates the same four steps verbatim ("Re-running the profiling step... Re-solving the assignment problem... Migrating expert weights... Updating dispatch metadata") and then cross-references `load_aware_assignment.md` Section 6.3 for migration cost detail.

The four-step enumeration in `mesh_topology_constraints.md` Section 9 is a full re-statement of material from `load_aware_assignment.md`. The paragraph after the list ("Steps 3 and 4 introduce latency proportional to...") directly duplicates the migration cost discussion. Replace the four-step list and the following paragraph with a condensed pointer: "For re-assignment mechanics and migration cost, see `load_aware_assignment.md` Section 6." Retain only the deployment policy guidance unique to Section 9 (the threshold "L sustained above 1.5 for more than 10,000 forward passes" and the maintenance-window recommendation), as those are not in `load_aware_assignment.md`. No technical information is lost.

**Estimated savings: ~8 lines in `mesh_topology_constraints.md`.**

---

## Load-Bearing Evidence

The following content is unique and technically essential and must not be removed:

- **`uniform_partitioning.md`** — Assignment rule $\sigma(e) = e \bmod N$ and the device-to-expert table (Section 1); memory footprint derivation $W_{\text{device}} = 192HD$ bytes and activation memory formula $A_{\text{device}}$ (Section 2); load imbalance metric $L$ definition and worked example with $L = 1.2$ (Section 5); the activation DRAM warning about simultaneous expert materialization (Section 2 Warning); the TTNN dispatch metadata tip about forward vs. reverse mapping (Section 7); the Zipf $L \approx 2$–$4$ warning (Section 5 Warning); the Python `expert_to_device_roundrobin` implementation (Section 7).

- **`load_aware_assignment.md`** — Full `profile_expert_frequencies` Python implementation with hook registration pattern (Section 1); calibration dataset size guidance with standard error bound $\pm 0.003$ (Section 1); bin-packing formulation and lower bound $\text{OPT} \geq \max(f_{\max}, 1.0)$ (Section 2); the equal-cardinality constraint warning (Section 2 Warning); full `greedy_decreasing_first` and `compute_load_imbalance` Python implementations with expected output block (Section 4); Graham 1969 approximation theorem $\text{GDF} \leq (4/3 - 1/(3N)) \times \text{OPT} = 31/24 \times \text{OPT}$ for $N = 8$ (Section 5); the ILP tip (Section 5); full `AdaptiveAssignmentManager` class (Section 6.1); migration cost formula $T_{\text{migrate}} = W_{\text{expert}} / \text{BW}$ and shadow migration warning (Section 6.3); load-aware vs. round-robin comparison table (Section 7).

- **`expert_replication.md`** — Capacity overflow condition derivation $f_e > CF \cdot k/E$ (Section 1); minimum replication factor formula $r_e = \max(1, \lceil f_e \cdot E / k \rceil)$ and the five-row worked example table (Section 2); the $r_e = N$ cap and irreducible imbalance explanation (Section 2); the DRAM overflow warning (Section 3); read-only replica consistency proof — no synchronization or all-reduce needed (Section 4); dispatch replica routing table and the four-policy comparison table including "least-loaded is impractical" (Section 5); full `build_dispatch_metadata_with_replicas`, `select_experts_for_replication`, and `build_replica_map` Python implementations (Sections 5 and 6); Qwen3.5-35B practical recommendation with $M = 8$ starting point (Section 6); the warning about routing burstiness needing higher $r_e$ (Section 2 Warning).

- **`mesh_topology_constraints.md`** — T3K linear chain topology diagram with `ttnn.Topology.Linear` / `cluster_axis=1` labels and the critical ring-topology warning (Section 1); link bandwidth $\approx 12.5$ GB/s and latency formula $t(d,d') \approx |d-d'| \times S/\text{BW}$ (Section 1); average hop count derivation $\bar{h} = 168/56 = 3.0$ with full double-summation shown (Section 2); maximum hop count 7 and middle-link congestion observation (Section 2); co-activation cost formula and probability-of-full-local-execution estimate $(1/8)^8 \approx 5.96 \times 10^{-8}$ (Section 3); baseline co-activation probability derivation $w_{ij}^{\text{uniform}} = k(k-1)/(E(E-1)) \approx 0.000858$ (Section 4); co-activation graph partition objective with distance weighting on linear chain (Section 5); full `profile_coactivation` Python implementation (Section 6); calibration size calculation requiring ~10,000 tokens (Section 6); full `coactivation_aware_assignment` Python implementation (Section 7); the combine-with-GDF tip (Section 7 Tip); the $w_{ij} > 5 \times w^{\text{uniform}} \approx 0.0043$ heuristic threshold for when to apply co-activation placement (Section 8); topology-aware vs. topology-agnostic comparison table (Section 10).

- **`index.md`** — Goal statement with optimization formulation $\min_\sigma \max_d \sum_{e:\sigma(e)=d} f_e \cdot B$ subject to DRAM constraint (Goal Statement section); chapter notation table including definitions of new symbols $f_e$, $r_e$, $M$, $w_{ij}$ (Chapter Notation section); strategy summary table (Assignment Strategies: Summary section); downstream chapter dependency map (Relationship to Later Chapters section).

---

## MINOR Suggestions

**1. Verbose prerequisite preambles in every chapter file repeat information already in `index.md`.**

Each of the four non-index files opens with a **Prerequisites** paragraph that lists chapters and files required before reading. These lists reproduce the reading-order logic fully stated in `index.md`'s "Reading Order" and "Prerequisites" sections. The lists in the individual files are useful as standalone references but contain more prose than necessary. Trim each Prerequisites paragraph to a single sentence such as "Requires all of Chapter 1, Chapter 2, and prior files in this chapter as listed in `index.md`." For `mesh_topology_constraints.md`, retain the specific `ch02_all_to_all_primitives/collective_communication_background.md` reference because it is not called out in `index.md`.

**Estimated savings: ~8 lines total across four files.**

---

**2. Redundant cross-references to `load_aware_assignment.md` Section 1 from `expert_replication.md` Section 6.**

`expert_replication.md` Section 6 (line 192) includes the instruction "Profile routing frequencies $f_e$ over a 5,000–10,000 token calibration set (see `load_aware_assignment.md`, Section 1)" as step 1 of the recommended configuration procedure. This is already the recommended entry point established by the reading order in `index.md`. The parenthetical suffices; the surrounding sentence can be shortened from a full imperative sentence to "Profile routing frequencies (see `load_aware_assignment.md`, Section 1)."

**Estimated savings: ~1 line.**

---

**3. `index.md` References section partially duplicates the chapter-level citation lists in individual files.**

`index.md` lists 13 references including both upstream (Ch1–Ch3) and downstream (Ch6–Ch8) chapters. Each individual file carries its own references section with the same entries plus paper citations. The `index.md` references section is informative as a navigation aid but could note that "full citations appear in each file's References section" and trim to only the downstream chapter links (Ch6, Ch7, Ch8) that are not covered in the individual files. This saves roughly 5–6 lines in `index.md` without removing any citation from the chapter as a whole.

**Estimated savings: ~5 lines.**

---

VERDICT: Crucial updates: yes

---

# Compression Analysis — Chapter 4: Expert-to-Device Assignment — Pass 2

## Summary

All three compression fixes from Pass 1 have been applied and verified:

- C1 applied: `load_aware_assignment.md` Section 1 back-references `uniform_partitioning.md` Section 4 for $f_{\text{avg}}$ instead of re-deriving it.
- C2 applied: `load_aware_assignment.md` Section 6.3 and `expert_replication.md` Section 3 both point to `uniform_partitioning.md` Section 2 for $W_{\text{expert}}$; the inline value $6HD$ bytes BF16 is retained in the pointer text for scannability.
- C3 applied: `mesh_topology_constraints.md` Section 9 replaces the 4-step re-enumeration with a single pointer to `load_aware_assignment.md` Section 6; the deployment threshold ($L > 1.5$ for 10,000 forward passes) and maintenance-window guidance are retained.

No correctness regressions. All five files remain technically consistent with ground truth. Estimated total reduction from C1+C2+C3: ~15 lines (~1% of total), consistent with Pass 1 estimates.

---

## CRUCIAL Suggestions

None.

---

## Load-Bearing Evidence

All load-bearing content identified in Pass 1 remains intact:

- `uniform_partitioning.md`: canonical $W_{\text{expert}} = 6HD$ bytes BF16 derivation (Section 2); canonical $f_{\text{avg}}$, $T_d$, $\bar{T}$ definitions (Sections 3–4); $L$ metric and worked example (Section 5); TTNN modulo dispatch rule and Python implementation (Sections 1, 7).
- `load_aware_assignment.md`: full `profile_expert_frequencies` and `greedy_decreasing_first` implementations; Graham 1969 bound $(4/3 - 1/24) \times \text{OPT}$; `AdaptiveAssignmentManager`; migration cost formula and shadow migration warning; load-aware vs. round-robin comparison table.
- `expert_replication.md`: overflow condition $f_e > CF \cdot k/E$; replication formula $r_e = \max(1, \lceil f_e \cdot E / k \rceil)$ with five-row worked table; read-only replica consistency proof; four-policy dispatch table; Zipf $r_{(1)} = \lceil 5.44 \rceil = 6$ calculation; $M = 8$ starting recommendation.
- `mesh_topology_constraints.md`: linear chain topology diagram and ring-topology warning; $\bar{h} = 3.0$ derivation; co-activation probability $w_{ij}^{\text{uniform}} \approx 0.000858$; graph partition objective; `profile_coactivation` and `coactivation_aware_assignment` implementations; $w_{ij} > 5 \times w^{\text{uniform}}$ heuristic threshold; deployment threshold $L > 1.5$ for 10,000 passes (retained from C3).
- `index.md`: optimization formulation; notation table; strategy summary table; downstream chapter dependency map.

---

## MINOR Suggestions

1. **Verbose prerequisite preambles** — carried forward from Pass 1. Each of the four non-index files opens with a prerequisites paragraph that largely reproduces `index.md`'s reading order. Trimming each to one sentence (retaining `mesh_topology_constraints.md`'s specific `collective_communication_background.md` reference) would save approximately 8 lines total. No information loss.

2. **Redundant cross-reference in `expert_replication.md` Section 6 step 1** — carried forward from Pass 1. The sentence "Profile routing frequencies $f_e$ over a 5,000–10,000 token calibration set (see `load_aware_assignment.md`, Section 1)" can be shortened to a parenthetical; the surrounding imperative sentence is redundant with the reading order. Saves approximately 1 line.

3. **`index.md` References section** — carried forward from Pass 1. The 13-entry reference list partially duplicates citation lists in individual files. Adding a note pointing to per-file references and trimming to downstream-only links (Ch6, Ch7, Ch8) would save approximately 5–6 lines without removing any citation from the chapter.

---

VERDICT: Crucial updates: no
