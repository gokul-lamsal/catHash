# Hashbroker CLI Miner

Standalone Linux miner for Hash Broker on Robinhood Chain. It reads the current on-chain challenge, hashes with all detected NVIDIA GPUs through a native CUDA worker, signs the `mine(uint256,bytes32)` transaction locally, broadcasts it, and repeats after confirmation. If CUDA is unavailable, it falls back to CPU workers.

This project does not use Chromium. Standard ASIC miners are not compatible with this protocol because each job is tied to an address and challenge and a successful result must be submitted as a wallet transaction.

## CUDA Linux build

Install a CUDA toolkit with `nvcc`, then build the native worker. On a GPU host, the Makefile queries `nvidia-smi` and includes every distinct compute capability it finds:

```bash
npm install
make -C cuda
```

If compiling on a machine without a GPU, it builds common CUDA 12.4 targets automatically. The GitHub release uses this multi-architecture build, so the downloaded binary can select the correct code for each detected GPU:

```bash
NVCC_ARCH=all make -C cuda
```

For a local build, you can still override detection with one architecture:

```bash
NVCC_ARCH=89 make -C cuda
```

The JS process automatically discovers every GPU reported by `nvidia-smi`, starts one CUDA worker per GPU, aggregates speed/hash counters, and submits the first valid proof.

## Run

```bash
npm install
export HASHBROKER_PRIVATE_KEY=0x...
node miner.mjs
```

Use fewer workers if needed:

```bash
node miner.mjs --workers 4
```

The private key is never printed. Never put it in source control.
