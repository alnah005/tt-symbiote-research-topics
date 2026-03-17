# 8.1 Model Loading and Weight Placement

## 8.1.1 Checkpoint Conversion

The standard HuggingFace Qwen3.5-35B checkpoint stores all 256 expert weight tensors in a single
monolithic format. Before loading onto a T3K cluster, this checkpoint must be converted to a
per-device sharded layout. The conversion step:

1. Reads each expert's weight tensors from the HuggingFace checkpoint.
2. Partitions the 256 experts evenly across the 8 devices, assigning experts
   $[0, 31]$ to device 0, $[32, 63]$ to device 1, and so on.
3. Writes per-device expert shard files that can be loaded directly onto each device's DRAM
   without additional redistribution at runtime.

This conversion is a one-time offline operation. Skipping it and loading the monolithic checkpoint
at runtime will saturate the PCIe and Ethernet interconnect during the first forward pass and
should be avoided.

## 8.1.2 Expert Weight Sharding

With $E = 256$ total experts and $N = 8$ devices, each device owns:

$$E_d = \frac{E}{N} = \frac{256}{8} = 32 \text{ experts per device}$$

Each expert's FFN weight occupies $6HD$ bytes in BF16, where $H = 7168$ is the hidden dimension
and $D \approx 14336$ is the intermediate FFN dimension [D UNVERIFIED]. Per device, the total
expert weight volume is:

$$\text{Expert DRAM per device} = E_d \times 6HD \text{ bytes} = 32 \times 6 \times 7168 \times D \text{ bytes [D UNVERIFIED]}$$

## 8.1.3 Weight Placement Decision

### L1 Cannot Hold Expert Weights

Wormhole B0 provides 80 Tensix cores with 1.5 MB of L1 per core, giving an aggregate L1 capacity
of:

$$\text{L1 total} = 80 \times 1.5 \text{ MB} = 120 \text{ MB per device}$$

The expert weight volume per device computed above is approximately 205 MB [D UNVERIFIED], which
far exceeds the 120 MB aggregate L1 budget. Even if the full L1 were dedicated to expert weights
(leaving nothing for activations or intermediate buffers), they would not fit. All expert weights
must therefore reside in DRAM.

**Placement rule:** All expert weight tensors use `ttnn.DRAM_MEMORY_CONFIG`.

```python
# Illustrative placement — API names [VERIFY]
expert_weight = ttnn.from_torch(
    expert_weight_torch,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    dtype=ttnn.bfloat16,
)
```

### Router Weight Placement: W_r

The router projection matrix $W_r$ has shape $[H, E] = [7168, 256]$. Its memory footprint
depends on the chosen precision:

| Precision | Calculation | Size |
|---|---|---|
| BF16 | $7168 \times 256 \times 2$ bytes | 3,670,016 bytes $\approx$ 3.67 MB |
| INT8 | $7168 \times 256 \times 1$ byte | 1,835,008 bytes $= 1.84$ MB |

BF16 carries 7 mantissa bits and a machine epsilon of $2^{-7} \approx 0.0078$. INT8 reduces $W_r$
to 1.84 MB, cutting the router's memory footprint by half at the cost of reduced precision in the
routing logit computation.

At 3.67 MB (BF16), $W_r$ fits comfortably within the 120 MB aggregate L1, and it is small enough
to replicate on all 8 devices without significant memory pressure. Replicating $W_r$ in L1 avoids
a DRAM round-trip on every router call, which runs once per MoE layer per decode step.

**Placement rule:** $W_r$ is replicated across all devices in L1.

```python
# Illustrative replication — API names [VERIFY]
W_r = ttnn.from_torch(
    W_r_torch,
    device=mesh_device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    dtype=ttnn.bfloat16,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
)
```

## 8.1.4 MeshDevice Weight Loading Pattern

Loading onto a `MeshDevice` follows a two-path pattern depending on whether the tensor is sharded
or replicated.

**Expert weights (sharded):** Each device receives only the shard of expert weights it owns. Use
`ttnn.distribute` with a shard mapper so that device $i$ receives experts
$[i \times E_d, (i+1) \times E_d - 1]$.

```python
# Illustrative sharding — API names [VERIFY]
expert_weights = ttnn.distribute(
    per_device_expert_tensors,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
)
```

**Router weight W_r (replicated):** Every device receives a full copy.

```python
# Illustrative replication — API names [VERIFY]
W_r = ttnn.distribute(
    W_r_torch,
    device=mesh_device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
)
```

## 8.1.5 Summary Table

| Tensor | Shape | Precision | Placement | Strategy |
|---|---|---|---|---|
| Expert FFN weights | $[E_d, 6HD]$ per device | BF16 | DRAM | Sharded (32 experts/device) |
| Router matrix $W_r$ | $[H, E] = [7168, 256]$ | BF16 or INT8 | L1 | Replicated on all 8 devices |
| Token activations (decode) | $[C \times E_d, H]$ | BF16 | L1 | Per-device after dispatch |
| Token activations (prefill) | $[S, H]$ | BF16 | DRAM | Per-device |

For the decode path, per-device activations are $[C \times E_d, H]$ tokens at most. With $C = 2$
(the $B = 32$ case) and $E_d = 32$, this is $64 \times 7168 \times 2$ bytes $\approx 0.9$ MB,
which fits in L1. The prefill path uses DRAM throughout because sequence-length activations are
too large for L1.

---

**Next:** [inference_loop_structure.md](./inference_loop_structure.md)
