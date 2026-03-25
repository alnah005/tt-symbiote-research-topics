# Chapter 1 — What Is TT-Symbiote?

This chapter introduces TT-Symbiote, explains why it exists, and orients you in the source tree so you can read or modify the framework with confidence.

## Contents

| File | What you will learn |
|------|---------------------|
| [`motivation.md`](./motivation.md) | The problem TT-Symbiote solves, how it fits relative to tt-transformers, and what "transparency" means in practice |
| [`source_layout.md`](./source_layout.md) | Repository structure, the purpose of each top-level subdirectory, key environment variables, and supported device architectures |

## Reading order

Read [`motivation.md`](./motivation.md) first to understand _why_ the framework exists before diving into _where_ things live in [`source_layout.md`](./source_layout.md).

## Prerequisites

This guide assumes you are comfortable with:

- PyTorch `nn.Module` subclassing and the Python dispatch mechanism (`__torch_dispatch__`)
- Basic understanding of hardware accelerators and memory management concepts
- Familiarity with `pytest` for running tests

No prior knowledge of TTNN or Tenstorrent hardware is required — that context is built up in this chapter.
