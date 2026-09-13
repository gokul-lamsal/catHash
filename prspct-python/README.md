# PRSPCT Python OpenCL miner

Headless multi-GPU miner for PRSPCT on Robinhood Chain. It reproduces the official browser proof exactly:

`keccak256(seed || wallet address || uint256 nonce) < target`

The miner auto-detects all OpenCL GPUs, aggregates speed/best bits, detects stale work, CPU-verifies every GPU result, simulates `claim(nonce)`, signs locally, and submits the payable transaction.

## Ubuntu as root (no venv)

```bash
apt update
apt install -y git python3 python3-pip python3-dev build-essential ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo
pip3 install --break-system-packages -r requirements.txt
```

Install the NVIDIA driver on the VPS host, confirm `nvidia-smi` and `clinfo`, then run:

```bash
export PRSPCT_PRIVATE_KEY="0xYOUR_NEW_PRIVATE_KEY"
python3 -u prspct_paid.py
```

Pure GPU work is the default (`PRSPCT_COIN_PERCENT=0`). To pay part of each current share price and proportionally reduce its work:

```bash
export PRSPCT_COIN_PERCENT=25
```

Useful settings:

```bash
export PRSPCT_MAX_CLAIMS=10       # 0 means run forever
export PRSPCT_GLOBAL=1048576      # work items per GPU dispatch
export PRSPCT_STALE_CHECK_SECONDS=4
export PRSPCT_RPCS="https://rpc.mainnet.chain.robinhood.com,https://robinhood-rpc.publicnode.com,https://rpc.ordofi.network"
```

Keep the private key secret. The miner never prints it, but it can spend the configured coin percentage plus transaction gas.
