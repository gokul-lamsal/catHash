# ShibaHash Python OpenCL miner

Install without a virtual environment:

```bash
apt update
apt install -y python3-pip ocl-icd-libopencl1 nvidia-opencl-icd
python3 -m pip install --break-system-packages numpy pyopencl pycryptodome web3
```

Run:

```bash
export SHIBAHASH_PRIVATE_KEY=0xYOUR_NEW_KEY
python3 shibahash_paid.py
```

`SHIBAHASH_BATCH` and `SHIBAHASH_GLOBAL` can be lowered if the driver runs out
of memory. `SHIBAHASH_MAX_ANCHOR_AGE` defaults to 230 blocks, below the
contract's 250-block limit, and `SHIBAHASH_STALE_CHECK_SECONDS` defaults to 3.
The miner reads the official current anchor, target and mint price;
CPU-verifies and contract-verifies every candidate, simulates `mine()` before
signing, and prints aggregate speed, hashes, best leading bits, expected time,
chance per minute, transaction hash and receipt status.
