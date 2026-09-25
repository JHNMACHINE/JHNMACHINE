# Jonathan Vecchione

I build ML training infrastructure in Rust and Python: fault-tolerant, distributed
PyTorch training that picks up exactly where it left off, on one GPU or across machines.

## Projects

### [Ravex](https://github.com/JHNMACHINE/ravex): fault-tolerant PyTorch training runtime

Put one decorator on the function that trains and the run survives preemption,
reboots and OOM kills. Rerun the same command and training continues
**bit-identically**, step by step, as if it had never stopped.

- Finds the model, optimizer, LR scheduler, AMP scaler and dataloader on its own,
  with no changes inside the loop
- Restores the complete state: weights, optimizer moments, schedule, RNG and
  dataset position
- Multi-GPU and multi-machine: DDP and FSDP, with gathered or per-rank sharded
  checkpoints and resharding onto a different number of ranks
- Survives losing a machine through S3/R2 storage or peer-to-peer replication,
  and supports elastic membership
- Works with HuggingFace `Trainer` and Lightning, and imports DeepSpeed ZeRO and
  `torch.distributed.checkpoint` checkpoints
- Rust core for the reshard planner and the replica transport

### [Moonclip](https://github.com/JHNMACHINE/moonclip): checkpoint engine

The storage layer under Ravex, and usable without it: Rust with Python bindings,
framework-agnostic. It tracks per-tensor deltas, skips unchanged weights,
compresses with zstd and writes to local disk, S3 or R2. Against `torch.save` a
save is 1.6–3× faster and about half the size.

## Stack

![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PyO3](https://img.shields.io/badge/PyO3-000000?logo=rust&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?logo=cplusplus&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)

## Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jonathan-vecchione-37591923b)
