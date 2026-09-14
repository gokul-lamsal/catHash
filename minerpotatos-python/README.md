# MinerPotatos Python OpenCL miner

Headless multi-GPU miner for MinerPotatos on Robinhood Chain (`4663`). It reproduces the official proof:

`keccak256(miner || prevWork || anchor || uint256 nonce) < target`

It detects every OpenCL GPU, reports aggregate speed/best bits/ETA/chance, monitors stale work and anchor age, verifies hits on the CPU and through `workFor`, simulates the payable mint, checks funds, signs locally, submits, and waits for confirmation.

## Fresh Ubuntu GPU VPS as root (no venv)

```bash
apt update
apt install -y git python3 python3-pip python3-dev build-essential ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo
git clone https://github.com/gokul-lamsal/catHash.git
cd catHash/minerpotatos-python
pip3 install --break-system-packages -r requirements.txt
nvidia-smi
clinfo | grep -E "Device Name|Device Type"
```

## Run

```bash
export MINERPOTATOS_PRIVATE_KEY="0xYOUR_PRIVATE_KEY"
export MINERPOTATOS_MAX_MINTS=10
export MINERPOTATOS_GLOBAL=4194304
export MINERPOTATOS_LOCAL=256
export MINERPOTATOS_NONCES_PER_ITEM=64
python3 -u minerpotatos_paid.py
```

`MINERPOTATOS_MAX_MINTS=0` means run continuously. Mint prices are displayed using a fixed `$2,500/ETH`; override it with `MINERPOTATOS_ETH_USD`.

Optional settings:

```bash
export MINERPOTATOS_ETH_USD=2500
export MINERPOTATOS_STALE_CHECK_SECONDS=3
export MINERPOTATOS_ANCHOR_MARGIN=8
export MINERPOTATOS_RPCS="https://rpc.minerpotatos.xyz,https://your-backup-rpc"
```

The status line shows aggregate speed followed by each device's rate. For RTX
5090 systems, start with `GLOBAL=4194304`, `LOCAL=256`, and
`NONCES_PER_ITEM=64`. The last setting makes each OpenCL dispatch do enough work
to avoid Python/driver launch gaps. You can benchmark `LOCAL=128`, `256`, and
`512`; `LOCAL` must divide `GLOBAL` exactly. Compare rates only after kernel
compilation and at least 10 seconds of mining.

Keep the private key secret and fund its Robinhood Chain address with enough ETH for each current mint price plus gas.
