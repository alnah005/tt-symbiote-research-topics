# Chapter 5 — Hardware Initialization and Device Ownership

This chapter defines how the Tenstorrent mesh device is opened, configured, and closed during a `tt-inference-server` deployment, and what responsibilities fall on the model versus the server worker.

## The Single Most Important Rule

> **The mesh device is opened by the server worker before `initialize_vllm_model` is called, and closed by the worker after the server shuts down. The model must not open or close the device.**

A TT Symbiote model receives a fully-initialized `mesh_device` handle as an argument to `initialize_vllm_model`. It must use that handle as-is and must never call `ttnn.open_mesh_device()` or `ttnn.close_mesh_device()`. Violating this rule will corrupt device state and typically causes a hard process crash or silent resource leak.

## Why This Matters

Device initialization in `ttnn` is a heavyweight operation: it allocates DRAM channel maps, programs dispatch cores, optionally configures inter-chip fabric topology, and reserves a trace buffer region in L1. All of those resources are owned by the worker process from the moment `TTWorker` is constructed until the worker tears down. The model is a guest inside that lifecycle — it borrows the device but does not own it.

## Reading Order

Work through the files in this order:

1. [`device_lifecycle.md`](./device_lifecycle.md) — Exactly what the worker does to open and close the device, how `MESH_DEVICE` drives the mesh shape, and how fabric and dispatch-core configuration are selected.
2. [`model_init_responsibilities.md`](./model_init_responsibilities.md) — What `initialize_vllm_model` must do, what it must not do, and special guidance for multi-chip TT Symbiote deployments and KV cache allocation.
3. [`environment_variables_reference.md`](./environment_variables_reference.md) — Complete reference table for every environment variable that affects a TT Symbiote model deployment, with extended descriptions.

## What's Next

[Chapter 6](../ch6_integration_checklist/index.md) covers the full integration checklist and a worked end-to-end example that ties together everything from Chapters 1–5.

---

**Next:** [`device_lifecycle.md`](./device_lifecycle.md)
