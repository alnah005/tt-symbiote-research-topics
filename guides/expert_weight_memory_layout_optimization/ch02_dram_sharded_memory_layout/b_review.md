# B Review — Chapter 2: DRAM-Sharded Memory Layout — Pass 1

1. **`sharding_strategies.md`, ~line 45 — Wrong DRAM controller count used as hardware justification.**

   The text claims "Wormhole exposes 8 DRAM controllers" as the justification for using 8 shards. Wormhole B0 has **6 DRAM controllers** (12 GDDR6 banks, 2 per controller). Using 8 shards is valid for arithmetic reasons (14336/8=1792, which is tile-aligned) and grid-width reasons (the Tensix grid is 8 columns wide), but the stated hardware justification is factually incorrect.

2. **`constructing_dram_sharded_config.md`, ~line 103 — Same wrong constant embedded in the construction guide.**

   The step-by-step construction guide repeats "the canonical choice matching Wormhole's 8 DRAM controllers." This embeds the incorrect hardware fact directly into the most action-oriented file in the chapter. Correct to 6 DRAM controllers (12 GDDR6 banks); reframe the 8-shard justification around Tensix grid-width alignment and tile-aligned shard size.

## Agent A Change Log — B Feedback Pass 1
- sharding_strategies.md: Fixed "8 DRAM controllers" to "6 DRAM controllers (12 GDDR6 banks)"; corrected shard-count justification to grid-width alignment and tile-alignment
- constructing_dram_sharded_config.md: Fixed "matching Wormhole's 8 DRAM controllers" to correct hardware count; same justification fix

---

# B Review — Chapter 2: DRAM-Sharded Memory Layout — Pass 2

Both Pass 1 fixes verified. No feedback — chapter approved.
