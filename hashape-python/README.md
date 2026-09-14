# HashApe Python OpenCL miner

Headless multi-GPU miner for HashApe on Robinhood Chain (4663). The miner reproduces the contract proof:

`keccak256(abi.encodePacked(challenge, miner address, uint256 nonce)) < effective target`

The effective target is the stricter of the active epoch target and the wallet's personal target. The contract requires a native ETH mint fee and limits each address to five NFTs.

## Fresh Ubuntu VPS as root (no venv)

```bash
apt update
apt install -y git python3 python3-pip python3-dev build-essential ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo
git clone https://github.com/gokul-lamsal/catHash.git
cd catHash/hashape-python
pip3 install --break-system-packages -r requirements.txt
nvidia-smi
clinfo | grep -E "Device Name|Device Type"
```

## Run

```bash
export HASHAPE_PRIVATE_KEY="0xYOUR_PRIVATE_KEY"
export HASHAPE_MAX_MINTS=5
export HASHAPE_GLOBAL=1048576
python3 -u hashape_paid.py
```

Mint costs are shown in ETH and approximate USD using a fixed `$2,500/ETH` rate. No price API is queried. To use another fixed rate:

```bash
export HASHAPE_ETH_USD=3000
```

`HASHAPE_MAX_MINTS=0` runs until the wallet reaches the contract's 5-NFT cap. The miner auto-detects every OpenCL GPU, aggregates hashrate and best bits, remine on challenge rotation, CPU-verifies hits, simulates the mint, checks fee/gas funds, signs locally, submits, and waits for confirmation.

Optional RPC override:

```bash
export HASHAPE_RPCS="https://your-rpc.example,https://backup-rpc.example"
```

The extra paid browser “workers” only partition browser nonce ranges. This CLI uses all locally detected GPUs while the contract independently verifies the proof.
