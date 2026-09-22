# Proof of Architect Python OpenCL miner

This is a standalone multi-GPU OpenCL miner for the paid `mint(uint256)` flow
on Proof of Architect / Arc mainnet.

The proof is Keccak-256 over:

```text
uint256(5042) || address(ProofOfArchitect) || address(miner) || uint256(nonce)
```

The miner keeps the upper 192 nonce bits random for each round and scans the
lower 64 bits on every detected OpenCL GPU. It re-reads the wallet's live
fractional target before submitting, verifies the hash on the CPU, signs the
EIP-1559 transaction locally, and waits for confirmation.

## Linux setup

The process needs an NVIDIA/AMD OpenCL driver and a working PyOpenCL install.
For NVIDIA, verify that `nvidia-smi` and `clinfo` both see the GPUs. Then:

```bash
git clone https://github.com/gokul-lamsal/catHash.git
cd catHash/proofarchitect-python
python3 -m pip install --break-system-packages -r requirements.txt
```

`--break-system-packages` is only needed on distributions that mark the
system Python as externally managed. Do not use a private key belonging to a
wallet you do not control.

## Run

Use a dedicated wallet funded with Arc's native USDC. The default is one paid
mint, which is safer for a first test:

```bash
export POA_PRIVATE_KEY='0xYOUR_PRIVATE_KEY'
export POA_MAX_MINTS=1
python3 -u poa_paid.py
```

For continuous paid minting, set `POA_MAX_MINTS=0`:

```bash
export POA_MAX_MINTS=0
export POA_GLOBAL=8388608
export POA_STALE_CHECK_SECONDS=3
python3 -u poa_paid.py
```

Useful optional settings:

```bash
export POA_GLOBAL=2097152
export POA_ITERS=16
export POA_RPCS='https://rpc.mainnet.arc.io,https://rpc.blockdaemon.mainnet.arc.io,https://rpc.drpc.mainnet.arc.io'
export POA_FEE_FLOOR_WEI=50000000000
```

The log shows total and aggregate hash rate, best leading-zero result,
expected time, chance per minute, the live target, exact mint due, signed
transaction hash, and receipt status. `POA_GLOBAL` is per GPU; increase it
only if GPU memory and stability allow it.

The script does not use Chromium or WebGPU. It uses OpenCL so it can run on a
headless Linux VPS with the vendor's GPU driver.

### `PLATFORM_NOT_FOUND_KHR`

This means PyOpenCL is installed but the system OpenCL ICD is missing or the
container cannot see the GPU. On an Ubuntu host/container, run:

```bash
apt-get update
apt-get install -y clinfo ocl-icd-libopencl1 nvidia-opencl-icd
nvidia-smi
clinfo | grep -E "Platform Name|Device Name|Device Type"
```

If `nvidia-smi` fails inside Docker, recreate the container with GPU access
(`--gpus all`) and the NVIDIA runtime. On Kubernetes, the pod must request an
NVIDIA GPU resource. CUDA and `nvidia-smi` alone are not enough for PyOpenCL;
the NVIDIA OpenCL ICD must also be available.
