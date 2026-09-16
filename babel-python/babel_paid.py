#!/usr/bin/env python3
"""Tower of Babel multi-GPU OpenCL miner for Arc mainnet."""

import importlib.util
import math
import os
from pathlib import Path
import time

from Crypto.Hash import keccak
from web3 import Web3


# Babel hashes the same 84-byte layout as PRSPCT: bytes32 seed, address sender,
# and uint256 nonce. Reuse that tested multi-GPU Keccak/OpenCL implementation.
shared_path = Path(__file__).resolve().parents[1] / "prspct-python" / "prspct_paid.py"
spec = importlib.util.spec_from_file_location("cathash_opencl", shared_path)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)
# Preserve a random 192-bit prefix and let the kernel vary the low 64 bits.
gpu.KERNEL = gpu.KERNEL.replace(
    "for(int i=0;i<52;i++)m[i]=prefix[i];",
    "for(int i=0;i<76;i++)m[i]=prefix[i];",
)
gpu.GLOBAL_SIZE = int(os.getenv("BABEL_GLOBAL", str(1 << 20)))
gpu.STALE_SECONDS = float(os.getenv("BABEL_STALE_CHECK_SECONDS", "3"))

CHAIN_ID = 5042
CONTRACT = Web3.to_checksum_address("0x07b5AB324fFD5f2CcCfd178B8f225E5419C5736c")
RPC_URLS = [url.strip() for url in os.getenv(
    "BABEL_RPCS",
    "https://towerofbabel.fly.dev/rpc/mainnet,https://rpc.mainnet.arc.io",
).split(",") if url.strip()]
MAX_MINTS = int(os.getenv("BABEL_MAX_MINTS", "0"))
COIN_PERCENT = int(os.getenv("BABEL_COIN_PERCENT", "0"))
SPONSOR = int(os.getenv("BABEL_SPONSOR", "0"))
PRIORITY_FEE = int(os.getenv("BABEL_PRIORITY_FEE_WEI", "1000000"))
MAX_SUPPLY = 8190

STATE_FIELDS = [
    "laid", "seed", "openAt", "price", "target", "floor", "pot", "closeAt",
    "coinWeight", "sweatWeight", "coinIn", "tillPaid", "day", "buildersOwed", "vault",
]
ABI = [
    {"type": "function", "name": "state", "stateMutability": "view", "inputs": [],
     "outputs": [{"name": "", "type": "tuple", "components": [
         {"name": name, "type": "bytes32" if name == "seed" else "uint256"}
         for name in STATE_FIELDS
     ]}]},
    {"type": "function", "name": "targetAt", "stateMutability": "view",
     "inputs": [{"name": "n", "type": "uint256"}, {"name": "coinBps", "type": "uint256"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "lay", "stateMutability": "payable",
     "inputs": [{"name": "sponsor", "type": "uint256"}, {"name": "nonce", "type": "uint256"}],
     "outputs": [{"name": "", "type": "uint256"}]},
]


class Chain:
    def __init__(self):
        self.index = -1
        self.w3 = None
        self.contract = None
        self.rotate()

    def rotate(self):
        last = None
        for _ in RPC_URLS:
            self.index = (self.index + 1) % len(RPC_URLS)
            try:
                candidate = Web3(Web3.HTTPProvider(
                    RPC_URLS[self.index], request_kwargs={"timeout": 20},
                ))
                if candidate.eth.chain_id != CHAIN_ID:
                    raise RuntimeError(f"wrong chain ID from {RPC_URLS[self.index]}")
                candidate.eth.block_number
                self.w3 = candidate
                self.contract = candidate.eth.contract(CONTRACT, abi=ABI)
                return
            except Exception as exc:
                last = exc
        raise RuntimeError(f"all RPC endpoints failed: {last}")

    def retry(self, operation, attempts=4):
        last = None
        for attempt in range(attempts):
            try:
                return operation(self.w3, self.contract)
            except Exception as exc:
                last = exc
                print(f"  RPC error: {exc}; retrying", flush=True)
                time.sleep(min(2 ** attempt, 8))
                self.rotate()
        raise last


def state_dict(raw):
    return {name: raw[index] for index, name in enumerate(STATE_FIELDS)}


def proof_hash(seed, address, nonce):
    packed = bytes(seed) + bytes.fromhex(address[2:]) + int(nonce).to_bytes(32, "big")
    return keccak.new(data=packed, digest_bits=256).digest()


def usdc(amount):
    return f"${int(amount) / 10**18:,.6f} USDC"


def main():
    if not 0 <= COIN_PERCENT < 100 or COIN_PERCENT % 5:
        raise SystemExit("ERROR: BABEL_COIN_PERCENT must be 0..95 in increments of 5")
    if SPONSOR < 0:
        raise SystemExit("ERROR: BABEL_SPONSOR cannot be negative")
    private_key = os.getenv("BABEL_PRIVATE_KEY")
    if not private_key:
        raise SystemExit("ERROR: BABEL_PRIVATE_KEY is not set")

    chain = Chain()
    account = chain.w3.eth.account.from_key(private_key)
    devices = gpu.gpu_devices()
    completed = 0
    coin_bps = COIN_PERCENT * 100
    print(
        f"Tower of Babel Python OpenCL miner | address={account.address} "
        f"GPUs={len(devices)} pay={COIN_PERCENT}% mine={100-COIN_PERCENT}%",
        flush=True,
    )
    print("GPUs: " + ", ".join(
        f"{index}:{device.name.strip()}" for index, device in enumerate(devices)
    ), flush=True)

    while MAX_MINTS == 0 or completed < MAX_MINTS:
        state = state_dict(chain.retry(lambda w3, contract: contract.functions.state().call()))
        laid = int(state["laid"])
        seed = bytes(state["seed"])
        if laid >= MAX_SUPPLY:
            print("the Tower of Babel is complete", flush=True)
            return
        if seed == bytes(32):
            raise RuntimeError("mining is not open: contract returned a zero seed")

        price = int(state["price"])
        payment = price * coin_bps // 10000
        target = int(chain.retry(
            lambda w3, contract: contract.functions.targetAt(laid, coin_bps).call()
        ))
        if target <= 0:
            raise RuntimeError("contract returned a zero target")
        work_bits = 256 - math.log2(target)
        print(
            f"[{time.strftime('%H:%M:%S')}] brick=#{laid+1}/{MAX_SUPPLY} round={state['floor']} "
            f"mint={usdc(payment)} fullPrice={usdc(price)} work~2^{work_bits:.2f} "
            f"seed=0x{seed.hex()[:16]}...",
            flush=True,
        )

        nonce_high = os.urandom(24)
        prefix = seed + bytes.fromhex(account.address[2:]) + nonce_high

        def stale():
            fresh = state_dict(chain.retry(
                lambda w3, contract: contract.functions.state().call(), 2
            ))
            return int(fresh["laid"]) != laid or bytes(fresh["seed"]) != seed

        nonce_low = gpu.mine_round(devices, prefix, target, stale)
        if nonce_low is None:
            print("  another brick was laid -> remine", flush=True)
            continue
        nonce = (int.from_bytes(nonce_high, "big") << 64) | nonce_low

        digest = proof_hash(seed, account.address, nonce)
        if int.from_bytes(digest, "big") >= target:
            print(f"  invalid GPU result discarded nonce={nonce} hash=0x{digest.hex()}", flush=True)
            continue
        print(
            f"  FOUND nonce={nonce} hash=0x{digest.hex()} "
            f"bits={gpu.leading_zero_bits(digest)}; verifying",
            flush=True,
        )

        fresh = state_dict(chain.retry(lambda w3, contract: contract.functions.state().call()))
        if int(fresh["laid"]) != laid or bytes(fresh["seed"]) != seed:
            print("  proof became stale before submit -> remine", flush=True)
            continue
        fresh_price = int(fresh["price"])
        value = fresh_price * coin_bps // 10000
        txfn = chain.contract.functions.lay(SPONSOR, nonce)
        call = {"from": account.address, "value": value}
        chain.retry(lambda w3, contract: txfn.call(call))
        gas = int(chain.retry(lambda w3, contract: w3.eth.estimate_gas({
            "from": account.address,
            "to": CONTRACT,
            "value": value,
            "data": txfn._encode_transaction_data(),
        }))) * 125 // 100
        latest = chain.retry(lambda w3, contract: w3.eth.get_block("latest"))
        base_fee = int(latest.get("baseFeePerGas", 0))
        max_fee = base_fee * 2 + PRIORITY_FEE
        balance = int(chain.retry(lambda w3, contract: w3.eth.get_balance(account.address)))
        required = value + gas * max_fee
        if balance < required:
            raise RuntimeError(
                f"insufficient funds: balance={usdc(balance)} required~{usdc(required)} "
                f"(mint={usdc(value)}, gas~{usdc(gas * max_fee)})"
            )
        transaction = txfn.build_transaction({
            "from": account.address,
            "value": value,
            "nonce": chain.retry(
                lambda w3, contract: w3.eth.get_transaction_count(account.address, "pending")
            ),
            "chainId": CHAIN_ID,
            "gas": gas,
            "type": 2,
            "maxPriorityFeePerGas": PRIORITY_FEE,
            "maxFeePerGas": max_fee,
        })
        signed = account.sign_transaction(transaction)
        txhash = chain.retry(
            lambda w3, contract: w3.eth.send_raw_transaction(signed.raw_transaction)
        )
        print(f"  signed and submitted tx={txhash.hex()}", flush=True)
        receipt = chain.retry(
            lambda w3, contract: w3.eth.wait_for_transaction_receipt(txhash, timeout=180)
        )
        print(f"  confirmed block={receipt.blockNumber} status={receipt.status}", flush=True)
        if receipt.status != 1:
            raise RuntimeError("lay transaction reverted")
        completed += 1

    print(f"completed {completed} brick mint(s)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped by user", flush=True)
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise
