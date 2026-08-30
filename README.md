# Distributed Transformer Layers

**Running a single decoder layer of an LLM on a remote worker.** A proof of concept for
splitting a model across machines that are individually too small to hold it.

The motivation is a hard constraint: an 8 GB laptop cannot load a model that needs 16 GB.
But a transformer is a stack of independent layers, and a layer is small. If layer *n* can
execute on a different machine from layer *n+1*, the ceiling stops being one machine's RAM
and becomes the whole network's.

This repository answers the first question that has to be answered before any of that is
worth attempting: **can you pull one layer out of a published checkpoint and make it run
somewhere else, correctly?**

Yes.

---

## What it does

```
  MASTER                                        WORKER (Ray actor)
  ─────────────────────────────────────         ─────────────────────────
  read model.safetensors.index.json
  find the shard holding layer 10
  extract only that layer's tensors    ──────▶  GemmaDecoderLayer(config, 10)
                                                load_state_dict(strict=True)

  build rotary position embeddings
  send hidden states + embeddings      ──────▶  forward pass
                                       ◀──────  hidden states back
  assert shapes match
```

Weights are pulled **selectively**. Rather than downloading and loading a whole model, the
master reads the safetensors index, resolves which shard contains
`model.layers.10.*`, and extracts only those tensors — so the master never holds the full
model either.

The worker is a `@ray.remote` actor that claims a GPU if one exists and falls back to CPU
if not, so the same code runs on a laptop and on a cluster without changes.

## Files

```
new_master.py    weight extraction, Ray orchestration, rotary embeddings, verification
new_worker.py    the remote actor — builds the layer, loads weights, runs forward
check_gemma.py   a signature inspector for GemmaRotaryEmbedding
```

`check_gemma.py` exists because `GemmaRotaryEmbedding.__init__` changed shape between
`transformers` releases, which surfaces as an opaque `KeyError: None` several frames deep.
Rather than guess, it prints the argument list your installed version actually expects.
Keeping the diagnostic in the repo is deliberate — the next person to hit that error will
hit it too.

## Running it

```bash
pip install torch transformers safetensors huggingface_hub ray
python new_master.py
```

Expected output:

```
[MASTER] Successfully extracted N tensors.
[WORKER ACTOR] Weights loaded successfully.
Input shape:  torch.Size([1, 8, 2048])
Output shape: torch.Size([1, 8, 2048])
✅ Success! The shapes match. The remote layer is working correctly.
```

Requires access to `google/gemma-2b` on Hugging Face.

## Two implementation notes

**Attention implementation is pinned to `eager`.** The default resolves through a backend
table that can return `None` on some installs, and the failure appears far from its cause.

**Position embeddings are computed on the master and passed in.** The layer expects them
pre-computed; building them inside the worker means every worker recomputes the same
tensor, which is wasted work once there is more than one.

## Status

A spike, and complete as one — a single layer, verified by shape equality on a forward
pass. What it does *not* yet do: chain multiple layers across multiple workers, stream a
KV cache between them, or run an end-to-end generation. Those are the next questions, and
they are only worth asking because this one came back yes.
