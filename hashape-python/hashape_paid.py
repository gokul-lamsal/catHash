#!/usr/bin/env python3
"""HashApe multi-GPU OpenCL miner for Robinhood Chain (4663)."""

import importlib.util
import math
import os
from pathlib import Path
import time

from web3 import Web3


# PRSPCT uses the same 84-byte packed Keccak preimage, so load its tested
# OpenCL worker and reporting code instead of maintaining a second kernel.
shared_path = Path(__file__).resolve().parents[1] / "prspct-python" / "prspct_paid.py"
spec = importlib.util.spec_from_file_location("catHash_keccak_opencl", shared_path)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)

CHAIN_ID = 4663
CONTRACT = Web3.to_checksum_address("0x7D959C29aa1098d93b307Ca40bEEEc0bF7bbfF85")
RPC_URLS = [x.strip() for x in os.getenv(
    "HASHAPE_RPCS",
    "https://robinhood-mainnet.g.alchemy.com/v2/VADj_sajpbD_KAWbnZk5x,"
    "https://rpc.mainnet.chain.robinhood.com,https://rpc.ordofi.network",
).split(",") if x.strip()]
MAX_MINTS = int(os.getenv("HASHAPE_MAX_MINTS", "0"))
STALE_SECONDS = float(os.getenv("HASHAPE_STALE_CHECK_SECONDS", "3"))
PRIORITY_FEE = int(os.getenv("HASHAPE_PRIORITY_FEE_WEI", "2000000"))
gpu.GLOBAL_SIZE = int(os.getenv("HASHAPE_GLOBAL", str(1 << 20)))
gpu.STALE_SECONDS = STALE_SECONDS

ABI = [
    {"inputs": [], "name": "totalMined", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "currentChallenge", "outputs": [{"type": "bytes32"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "tokenId", "type": "uint256"}], "name": "getMintFee", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "tokenId", "type": "uint256"}], "name": "getEpoch", "outputs": [{"components": [
        {"name": "id", "type": "uint256"}, {"name": "startToken", "type": "uint256"},
        {"name": "endToken", "type": "uint256"}, {"name": "mintFeeWei", "type": "uint256"},
        {"name": "feeUsd", "type": "uint256"}, {"name": "target", "type": "uint256"},
        {"name": "name", "type": "string"}], "name": "", "type": "tuple"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "user", "type": "address"}], "name": "walletMints", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "user", "type": "address"}], "name": "getDifficultyTargetForWallet", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "nonce", "type": "uint256"}, {"name": "challenge", "type": "bytes32"}], "name": "mintWithMiningProof", "outputs": [{"type": "uint256"}], "stateMutability": "payable", "type": "function"},
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


def read_job(chain, address):
    total = int(chain.retry(lambda w, c: c.functions.totalMined().call()))
    if total >= 10000:
        raise RuntimeError("HashApe supply is exhausted")
    token_id = total + 1
    challenge = bytes(chain.retry(lambda w, c: c.functions.currentChallenge().call()))
    epoch = chain.retry(lambda w, c: c.functions.getEpoch(token_id).call())
    wallet_mints = int(chain.retry(lambda w, c: c.functions.walletMints(address).call()))
    wallet_target = int(chain.retry(lambda w, c: c.functions.getDifficultyTargetForWallet(address).call()))
    epoch_target = int(epoch[5])
    return {
        "total": total, "token": token_id, "challenge": challenge,
        "epoch_id": int(epoch[0]), "epoch_name": str(epoch[6]),
        "fee": int(epoch[3]), "target": min(epoch_target, wallet_target),
        "epoch_target": epoch_target, "wallet_target": wallet_target,
        "wallet_mints": wallet_mints,
    }


def main():
    private_key = os.getenv("HASHAPE_PRIVATE_KEY")
    if not private_key:
        raise SystemExit("ERROR: HASHAPE_PRIVATE_KEY is not set")
    chain = Chain()
    account = chain.w3.eth.account.from_key(private_key)
    devices = gpu.gpu_devices()
    completed = 0
    print(f"HashApe Python OpenCL miner | address={account.address} GPUs={len(devices)}")
    print("GPUs: " + ", ".join(f"{i}:{device.name.strip()}" for i, device in enumerate(devices)), flush=True)

    while MAX_MINTS == 0 or completed < MAX_MINTS:
        job = read_job(chain, account.address)
        if job["wallet_mints"] >= 5:
            print("wallet quota reached: this address already minted 5/5 HashApe NFTs")
            return
        target = job["target"]
        if target <= 0:
            raise RuntimeError("contract returned a zero mining target")
        work_bits = 256 - math.log2(target)
        print(
            f"[{time.strftime('%H:%M:%S')}] token=#{job['token']} wallet={job['wallet_mints']}/5 "
            f"epoch={job['epoch_id']} feeWei={job['fee']} work≈2^{work_bits:.2f} "
            f"challenge=0x{job['challenge'].hex()[:16]}… target=0x{target:064x}", flush=True,
        )
        prefix = job["challenge"] + bytes.fromhex(account.address[2:])

        def stale():
            current = bytes(chain.retry(lambda w, c: c.functions.currentChallenge().call(), 2))
            return current != job["challenge"]

        nonce = gpu.mine_round(devices, prefix, target, stale)
        if nonce is None:
            print("  challenge changed -> remine", flush=True)
            continue
        digest = gpu.work_hash(job["challenge"], account.address, nonce)
        if int.from_bytes(digest, "big") >= target:
            print(f"  invalid GPU result discarded nonce={nonce} hash=0x{digest.hex()}", flush=True)
            continue
        print(f"  FOUND nonce={nonce} hash=0x{digest.hex()} bits={gpu.leading_zero_bits(digest)}; verifying", flush=True)

        current_challenge = bytes(chain.retry(lambda w, c: c.functions.currentChallenge().call()))
        if current_challenge != job["challenge"]:
            print("  proof became stale before submit -> remine", flush=True)
            continue
        fee = int(chain.retry(lambda w, c: c.functions.getMintFee(job["token"]).call()))
        txfn = chain.contract.functions.mintWithMiningProof(nonce, job["challenge"])
        call = {"from": account.address, "value": fee}
        chain.retry(lambda w, c: txfn.call(call))
        gas = int(chain.retry(lambda w, c: w.eth.estimate_gas({
            "from": account.address, "to": CONTRACT, "value": fee,
            "data": txfn._encode_transaction_data(),
        })))
        latest = chain.retry(lambda w, c: w.eth.get_block("latest"))
        max_fee = int(latest.get("baseFeePerGas", 0)) * 3 + PRIORITY_FEE
        balance = int(chain.retry(lambda w, c: w.eth.get_balance(account.address)))
        required = fee + gas * max_fee
        if balance < required:
            raise RuntimeError(
                f"insufficient funds: balanceWei={balance} requiredWei≈{required} "
                f"(mintFeeWei={fee}, gas={gas})"
            )
        transaction = txfn.build_transaction({
            "from": account.address, "value": fee,
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
