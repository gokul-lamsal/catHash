#!/usr/bin/env python3
"""MinerPotatos multi-GPU OpenCL miner for Robinhood Chain (4663)."""

import importlib.util
import math
import os
from pathlib import Path
import time

from Crypto.Hash import keccak
from web3 import Web3


# Reuse the tested Keccak/OpenCL runtime, but specialize its kernel for the
# MinerPotatos 116-byte prefix and nonce position (bytes 84..115).
shared_path = Path(__file__).resolve().parents[1] / "prspct-python" / "prspct_paid.py"
spec = importlib.util.spec_from_file_location("cathash_opencl", shared_path)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)
gpu.KERNEL = gpu.KERNEL.replace(
    "for(int i=0;i<52;i++)m[i]=prefix[i];",
    "for(int i=0;i<116;i++)m[i]=prefix[i];",
).replace("m[83-i]", "m[115-i]")
gpu.GLOBAL_SIZE = int(os.getenv("MINERPOTATOS_GLOBAL", str(1 << 20)))
gpu.STALE_SECONDS = float(os.getenv("MINERPOTATOS_STALE_CHECK_SECONDS", "3"))

CHAIN_ID = 4663
CONTRACT = Web3.to_checksum_address("0xb0db77c5d6ed578189609ecc72d25699a79f785b")
RPC_URLS = [x.strip() for x in os.getenv(
    "MINERPOTATOS_RPCS",
    "https://rpc.minerpotatos.xyz,https://rpc.mainnet.chain.robinhood.com,https://rpc.ordofi.network",
).split(",") if x.strip()]
MAX_MINTS = int(os.getenv("MINERPOTATOS_MAX_MINTS", "0"))
PRIORITY_FEE = int(os.getenv("MINERPOTATOS_PRIORITY_FEE_WEI", "2000000"))
ETH_USD = float(os.getenv("MINERPOTATOS_ETH_USD", "2500"))
ANCHOR_MARGIN = int(os.getenv("MINERPOTATOS_ANCHOR_MARGIN", "8"))

STATUS_FIELDS = [
    ("anchorBlock", "uint256"), ("anchor", "bytes32"), ("prevWork", "bytes32"),
    ("target", "uint256"), ("difficulty", "uint256"), ("price", "uint256"),
    ("supply", "uint256"), ("maxSupply", "uint256"), ("targetInterval", "uint256"),
    ("lastMintAt", "uint256"), ("startTime", "uint256"), ("anchorWindow", "uint256"),
    ("burstLeft", "uint256"), ("burstReadyAt", "uint256"), ("chainTime", "uint256"),
    ("blockNumber", "uint256"), ("epoch", "uint256"), ("floorBits", "uint256"),
    ("streak", "uint256"), ("activeSupply", "uint256"), ("epochEndsAt", "uint256"),
    ("vault", "uint256"), ("redeemValue", "uint256"), ("epochPrice", "uint256"),
]
ABI = [
    {"inputs": [], "name": "miningStatus", "outputs": [{"components": [
        {"name": name, "type": kind} for name, kind in STATUS_FIELDS
    ], "name": "s", "type": "tuple"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "miner", "type": "address"}, {"name": "prevWork_", "type": "bytes32"},
                {"name": "anchor", "type": "bytes32"}, {"name": "nonce", "type": "uint256"}],
     "name": "workFor", "outputs": [{"type": "bytes32"}], "stateMutability": "pure", "type": "function"},
    {"inputs": [{"name": "nonce", "type": "uint256"}, {"name": "anchorBlock", "type": "uint256"}],
     "name": "mine", "outputs": [{"name": "tokenId", "type": "uint256"}], "stateMutability": "payable", "type": "function"},
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
                candidate = Web3(Web3.HTTPProvider(RPC_URLS[self.index], request_kwargs={"timeout": 20}))
                if candidate.eth.chain_id != CHAIN_ID:
                    raise RuntimeError(f"wrong chain ID from {RPC_URLS[self.index]}")
                candidate.eth.block_number
                self.w3 = candidate
                self.contract = candidate.eth.contract(CONTRACT, abi=ABI)
                return
            except Exception as exc:
                last = exc
        raise RuntimeError(f"all RPC endpoints failed: {last}")

    def retry(self, fn, attempts=4):
        last = None
        for attempt in range(attempts):
            try:
                return fn(self.w3, self.contract)
            except Exception as exc:
                last = exc
                print(f"  RPC error: {exc}; retrying", flush=True)
                time.sleep(min(2 ** attempt, 8))
                self.rotate()
        raise last


def job_from_status(status):
    return {name: status[index] for index, (name, _) in enumerate(STATUS_FIELDS)}


def prefix_for(address, prev_work, anchor):
    return bytes.fromhex(address[2:]) + bytes(prev_work) + bytes(anchor) + bytes(32)


def work_hash(address, prev_work, anchor, nonce):
    packed = bytes.fromhex(address[2:]) + bytes(prev_work) + bytes(anchor) + int(nonce).to_bytes(32, "big")
    return keccak.new(data=packed, digest_bits=256).digest()


def cost(wei):
    eth = int(wei) / 10**18
    return f"${eth * ETH_USD:,.2f} ({eth:.6f} ETH)"


def main():
    private_key = os.getenv("MINERPOTATOS_PRIVATE_KEY")
    if not private_key:
        raise SystemExit("ERROR: MINERPOTATOS_PRIVATE_KEY is not set")
    chain = Chain()
    account = chain.w3.eth.account.from_key(private_key)
    devices = gpu.gpu_devices()
    completed = 0
    print(f"MinerPotatos Python OpenCL miner | address={account.address} GPUs={len(devices)} ETH≈${ETH_USD:,.0f}")
    print("GPUs: " + ", ".join(f"{i}:{device.name.strip()}" for i, device in enumerate(devices)), flush=True)

    while MAX_MINTS == 0 or completed < MAX_MINTS:
        job = job_from_status(chain.retry(lambda w, c: c.functions.miningStatus().call()))
        if int(job["supply"]) >= int(job["maxSupply"]):
            print("all MinerPotatos have been mined")
            return
        target = int(job["target"])
        if target <= 0:
            raise RuntimeError("contract returned a zero target")
        bits = 256 - math.log2(target)
        anchor_age = int(job["blockNumber"]) - int(job["anchorBlock"])
        print(
            f"[{time.strftime('%H:%M:%S')}] next=#{int(job['supply'])+1}/{job['maxSupply']} "
            f"mint={cost(job['price'])} target≈2^{bits:.2f} anchor={job['anchorBlock']} "
            f"age={anchor_age}/{job['anchorWindow']} prev=0x{bytes(job['prevWork']).hex()[:14]}…",
            flush=True,
        )
        prefix = prefix_for(account.address, job["prevWork"], job["anchor"])

        def stale():
            fresh = job_from_status(chain.retry(lambda w, c: c.functions.miningStatus().call(), 2))
            expired = int(fresh["blockNumber"]) - int(job["anchorBlock"]) >= int(job["anchorWindow"]) - ANCHOR_MARGIN
            return bytes(fresh["prevWork"]) != bytes(job["prevWork"]) or int(fresh["target"]) < target or expired

        nonce = gpu.mine_round(devices, prefix, target, stale)
        if nonce is None:
            print("  work changed or anchor aged -> remine", flush=True)
            continue
        digest = work_hash(account.address, job["prevWork"], job["anchor"], nonce)
        if int.from_bytes(digest, "big") >= target:
            print(f"  invalid GPU result discarded nonce={nonce} hash=0x{digest.hex()}", flush=True)
            continue
        onchain = bytes(chain.retry(lambda w, c: c.functions.workFor(
            account.address, job["prevWork"], job["anchor"], nonce
        ).call()))
        if onchain != digest:
            raise RuntimeError(f"CPU/contract hash mismatch: cpu=0x{digest.hex()} contract=0x{onchain.hex()}")
        print(f"  FOUND nonce={nonce} hash=0x{digest.hex()} bits={gpu.leading_zero_bits(digest)}; verified", flush=True)

        fresh = job_from_status(chain.retry(lambda w, c: c.functions.miningStatus().call()))
        expired = int(fresh["blockNumber"]) - int(job["anchorBlock"]) >= int(job["anchorWindow"]) - ANCHOR_MARGIN
        if bytes(fresh["prevWork"]) != bytes(job["prevWork"]) or int(fresh["target"]) < target or expired:
            print("  proof became stale before submit -> remine", flush=True)
            continue
        price = int(fresh["price"])
        txfn = chain.contract.functions.mine(nonce, int(job["anchorBlock"]))
        call = {"from": account.address, "value": price}
        chain.retry(lambda w, c: txfn.call(call))
        gas = int(chain.retry(lambda w, c: w.eth.estimate_gas({
            "from": account.address, "to": CONTRACT, "value": price,
            "data": txfn._encode_transaction_data(),
        }))) * 125 // 100
        latest = chain.retry(lambda w, c: w.eth.get_block("latest"))
        max_fee = int(latest.get("baseFeePerGas", 0)) * 3 + PRIORITY_FEE
        balance = int(chain.retry(lambda w, c: w.eth.get_balance(account.address)))
        required = price + gas * max_fee
        if balance < required:
            raise RuntimeError(
                f"insufficient funds: balance={cost(balance)} required≈{cost(required)} "
                f"(mint={cost(price)}, gas≈{cost(gas * max_fee)})"
            )
        transaction = txfn.build_transaction({
            "from": account.address, "value": price,
            "nonce": chain.retry(lambda w, c: w.eth.get_transaction_count(account.address, "pending")),
            "chainId": CHAIN_ID, "gas": gas, "type": 2,
            "maxPriorityFeePerGas": PRIORITY_FEE, "maxFeePerGas": max_fee,
        })
        signed = account.sign_transaction(transaction)
        txhash = chain.retry(lambda w, c: w.eth.send_raw_transaction(signed.raw_transaction))
        print(f"  signed and submitted tx={txhash.hex()}", flush=True)
        receipt = chain.retry(lambda w, c: w.eth.wait_for_transaction_receipt(txhash, timeout=180))
        print(f"  confirmed block={receipt.blockNumber} status={receipt.status}", flush=True)
        if receipt.status != 1:
            raise RuntimeError("mint transaction reverted")
        completed += 1

    print(f"completed {completed} mint(s)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped by user")
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise
