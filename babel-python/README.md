# Tower of Babel Python OpenCL miner

Headless multi-GPU miner for [Tower of Babel](https://towerofbabel.fly.dev/mine)
on Arc mainnet (`5042`). It implements the site's proof exactly:

`keccak256(seed || sender || uint256 nonce) < target`

The miner detects every OpenCL GPU, reports aggregate speed and best bits,
automatically abandons stale work, verifies every hit on the CPU, simulates the
mint, signs locally, submits `lay(sponsor, nonce)`, and waits for confirmation.

Pure mining is the default: there is no mint payment, but Arc network gas is
still required. Optional mixed mints can pay 5% to 95% of the current USDC mint
price in exchange for an easier proof target.

## Install on an Ubuntu NVIDIA VPS as root (no venv)

```bash
apt update
apt install -y git python3 python3-pip python3-dev build-essential \
  ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo

git clone https://github.com/gokul-lamsal/catHash.git
cd catHash/babel-python
pip3 install --break-system-packages -r requirements.txt

nvidia-smi
clinfo | grep -E "Device Name|Device Type"
```

The NVIDIA driver must expose OpenCL. The CUDA toolkit is not required by this
Python/OpenCL version if `clinfo` already lists the GPUs.

## Run

```bash
export BABEL_PRIVATE_KEY="0xYOUR_PRIVATE_KEY"
export BABEL_MAX_MINTS=10
export BABEL_COIN_PERCENT=0
export BABEL_GLOBAL=1048576

python3 -u babel_paid.py
```

`BABEL_MAX_MINTS=0` runs continuously. `BABEL_COIN_PERCENT=0` performs pure
mining and submits only the proof plus gas. Accepted percentages are `0` through
`95` in increments of `5`.

Optional settings:

```bash
export BABEL_SPONSOR=0
export BABEL_STALE_CHECK_SECONDS=3
export BABEL_RPCS="https://towerofbabel.fly.dev/rpc/mainnet,https://rpc.mainnet.arc.io"
export BABEL_PRIORITY_FEE_WEI=1000000
```

Fund the mining address with enough Arc USDC for gas and any selected payment
percentage. Keep the private key secret; it is used only for local signing.
