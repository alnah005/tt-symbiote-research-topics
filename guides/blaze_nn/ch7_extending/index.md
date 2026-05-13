# Chapter 7 — Extending blaze-nn

The capstone chapter. Chapters 5 and 6 explained how `Module.__call__` reaches `program.run()` and how `F.<op>` resolves through the registry. This chapter turns those internals into concrete recipes a contributor can follow without re-reading the framework end to end:

1. How to add a wrapper for an op that already exists upstream in tt-blaze.
2. How to synthesize a new fused op when the op does not exist upstream.
3. How to extend containers and modules beyond the four built-ins and `OpModule`.
4. The test taxonomy — a reverse index from every chapter section back to the test files that back its claims, plus the three-tier recipe for new features.
5. A contributing checklist of concrete extension recipes, the anti-patterns the framework relies on you not violating, and a known gap (compose-mode coverage) flagged for new contributors.

This is the final chapter of the guide.

## Files

1. [Adding an op wrapper — the `blaze_nn/ops/<op>/` convention](add_an_op_wrapper.md)
2. [Adding a fused op — when the op does not exist upstream](add_a_fused_op.md)
3. [Extending containers and modules — beyond the built-ins](extending_containers_and_modules.md)
4. [Testing strategy — the test taxonomy (reverse index)](testing_strategy.md)
5. [Contributing checklist — concrete recipes and anti-patterns](contributing_checklist.md)
