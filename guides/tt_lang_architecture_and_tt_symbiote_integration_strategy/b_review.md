# Agent B Review: Cross-Chapter --- Pass 1

No feedback --- guide approved.

**Summary of checks performed:**

1. **Terminology consistency:** Key terms (DFB, TensorBlock, CompiledTTNNKernel, TTNNModule, grid, node, TorchTTNNTensor, BlockStateMachine, AccessState, CompilerOptions, pykernel_gen/operation alias) are used consistently across all 8 chapters. Ch2 introduces `pykernel_gen` as the compiler-path entry point and immediately clarifies its alias relationship to `@ttl.operation` (the name used everywhere else), which is acceptable since Ch2 is specifically about compiler internals.

2. **Cross-chapter references:** All cross-chapter links use correct relative paths (`../chN_name/index.md`). Verified: Ch3 references Ch1 and Ch2; Ch4 references Ch2; Ch6 references Ch2 and Ch5; Ch7 references Ch1, Ch5, and Ch6; Ch8 references Ch1, Ch2, Ch4, and Ch6. No broken links.

3. **Guide index completeness:** The top-level `index.md` links to all 8 chapters with clickable relative paths. It includes all expected sections: How to Use This Guide (reading-path table), Chapter Index, Quick Reference, Prerequisites, and Source Code Locations.

4. **Notation consistency:** Code formatting (backtick usage for API names, env vars, file paths) is consistent across chapters. ASCII diagrams use a uniform style. Chapter title dash style varies slightly (em dash vs double/triple hyphen) but this is cosmetic and does not affect readability or navigation.
