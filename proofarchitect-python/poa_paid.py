#!/usr/bin/env python3
"""Proof of Architect paid PoW miner for Arc mainnet.

The proof input is the same packed byte layout used by the official miner:

    uint256(chainId) || address(contract) || address(miner) || uint256(nonce)

The OpenCL worker keeps the upper 192 nonce bits fixed for a round and scans
the lower 64 bits.  This preserves the full uint256 nonce space while keeping
the kernel fast and allowing each GPU to use the same work engine as the
other miners in this repository.
"""

import math
import os
import queue
import secrets
import threading
import time

import numpy as np
import pyopencl as cl
from Crypto.Hash import keccak
from web3 import Web3


CHAIN_ID = 5042
CONTRACT = Web3.to_checksum_address("0x3E20bb7be2C46f94Cab78d340D3F79Afc2a9Fed4")
RPC_URLS = [
    value.strip()
    for value in os.getenv(
        "POA_RPCS",
        "https://rpc.mainnet.arc.io,https://rpc.blockdaemon.mainnet.arc.io,https://rpc.drpc.mainnet.arc.io",
    ).split(",")
    if value.strip()
]

ITERS = int(os.getenv("POA_ITERS", "16"))
GLOBAL_SIZE = int(os.getenv("POA_GLOBAL", str(8 * 1024 * 1024)))
STALE_SECONDS = float(os.getenv("POA_STALE_CHECK_SECONDS", "3"))
MAX_MINTS = int(os.getenv("POA_MAX_MINTS", "1"))
FEE_FLOOR_WEI = int(os.getenv("POA_FEE_FLOOR_WEI", "50000000000"))

# Arc exposes USDC as its native gas currency.  The contract values use the
# EVM 18-decimal denomination (for example 1 USDC is 1e18 here).
NATIVE_DECIMALS = 18


ABI = [
    {
        "inputs": [],
        "name": "currentMintDue",
        "outputs": [
            {"name": "due", "type": "uint256"},
            {"name": "fee", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "currentWave",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalMinted",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "maxSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "mintPaused",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "miner", "type": "address"}],
        "name": "requiredBits",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "miner", "type": "address"}],
        "name": "requiredMilli",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "miner", "type": "address"}],
        "name": "targetFor",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "miner", "type": "address"},
            {"name": "nonce", "type": "uint256"},
        ],
        "name": "nonceUsed",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "nonce", "type": "uint256"}],
        "name": "mint",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function",
    },
]


# One Keccak-256 block.  The message is 104 bytes, so it fits in the 136-byte
# Keccak rate and can be hashed without a second permutation.
KERNEL = r"""
__constant ulong RC[24]={1UL,0x8082UL,0x800000000000808aUL,0x8000000080008000UL,0x808bUL,0x80000001UL,0x8000000080008081UL,0x8000000000008009UL,0x8aUL,0x88UL,0x80008009UL,0x8000000aUL,0x8000808bUL,0x800000000000008bUL,0x8000000000008089UL,0x8000000000008003UL,0x8000000000008002UL,0x8000000000000080UL,0x800aUL,0x800000008000000aUL,0x8000000080008081UL,0x8000000000008080UL,0x80000001UL,0x8000000080008008UL};
__constant int PI[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
__constant int RH[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
ulong rol(ulong x,int n){return n?((x<<n)|(x>>(64-n))):x;}
void perm(ulong a[25]){for(int r=0;r<24;r++){ulong c[5],d[5];for(int x=0;x<5;x++)c[x]=a[x]^a[x+5]^a[x+10]^a[x+15]^a[x+20];for(int x=0;x<5;x++)d[x]=c[(x+4)%5]^rol(c[(x+1)%5],1);for(int i=0;i<25;i++)a[i]^=d[i%5];ulong t=a[1],u;for(int i=0;i<24;i++){u=a[PI[i]];a[PI[i]]=rol(t,RH[i]);t=u;}for(int y=0;y<5;y++){ulong b0=a[5*y],b1=a[1+5*y],b2=a[2+5*y],b3=a[3+5*y],b4=a[4+5*y];a[5*y]=b0^((~b1)&b2);a[1+5*y]=b1^((~b2)&b3);a[2+5*y]=b2^((~b3)&b4);a[3+5*y]=b3^((~b4)&b0);a[4+5*y]=b4^((~b0)&b1);}a[0]^=RC[r];}}
__kernel void mine(__global const uchar* prefix,__global const uchar* target,ulong base,volatile __global uint* found,__global ulong* nonce,__global uint* bestbits,uint iters){
 ulong n=base+(ulong)get_global_id(0); uchar m[136]; for(int i=0;i<136;i++)m[i]=0; for(int i=0;i<96;i++)m[i]=prefix[i];
 for(int i=0;i<8;i++)m[103-i]=(uchar)(n>>(8*i)); m[104]=1; m[135]=0x80; ulong a[25]; for(int i=0;i<25;i++)a[i]=0;
 for(uint it=0;it<iters;it++){
  for(int i=0;i<25;i++)a[i]=0;
  for(int i=0;i<17;i++){ulong v=0;for(int k=0;k<8;k++)v|=((ulong)m[i*8+k])<<(8*k);a[i]^=v;} perm(a); uchar h[32];
  for(int i=0;i<4;i++)for(int k=0;k<8;k++)h[i*8+k]=(uchar)(a[i]>>(8*k)); uint bits=0;for(int i=0;i<32;i++){if(h[i]==0)bits+=8;else{bits+=clz((uint)h[i])-24;break;}} atomic_max(bestbits,bits);
  int less=0;for(int i=0;i<32;i++){if(h[i]<target[i]){less=1;break;}if(h[i]>target[i])break;} if(less&&atomic_cmpxchg(found,0,1)==0)nonce[0]=n;
  if(found[0])return;
  n+=get_global_size(0);
  for(int i=0;i<8;i++)m[103-i]=(uchar)(n>>(8*i));
 }
}
"""


def keccak256(data):
    return keccak.new(data=bytes(data), digest_bits=256).digest()


def proof_bytes(address, nonce):
    """Return abi.encodePacked(uint256(chainId), address, address, uint256)."""
    return (
        CHAIN_ID.to_bytes(32, "big")
        + bytes.fromhex(CONTRACT[2:])
        + bytes.fromhex(address[2:])
        + int(nonce).to_bytes(32, "big")
    )


def work_hash(address, nonce):
    return keccak256(proof_bytes(address, nonce))


def leading_zero_bits(digest):
    value = int.from_bytes(digest, "big")
    return 256 if value == 0 else 256 - value.bit_length()


def fmt_rate(rate):
    units = ["H/s", "KH/s", "MH/s", "GH/s", "TH/s"]
    index = 0
    while rate >= 1000 and index < len(units) - 1:
        rate /= 1000
        index += 1
    return f"{rate:.2f} {units[index]}"


def fmt_time(seconds):
    if not math.isfinite(seconds):
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def fmt_native(value):
    return f"{int(value) / (10 ** NATIVE_DECIMALS):.6f} USDC"


def work_bits(target):
    if target <= 0:
        return float("inf")
    return max(0.0, 256.0 - math.log2(target))


def gpu_devices():
    try:
        platforms = cl.get_platforms()
    except cl.LogicError as exc:
        if "PLATFORM_NOT_FOUND_KHR" in str(exc):
            raise RuntimeError(
                "OpenCL has no platform. Install the vendor OpenCL ICD "
                "(for NVIDIA, nvidia-opencl-icd), verify `nvidia-smi` and "
                "`clinfo`, and expose the GPU with the NVIDIA container runtime."
            ) from exc
        raise

    devices = []
    for platform in platforms:
        devices.extend(platform.get_devices(device_type=cl.device_type.GPU))
    if not devices:
        raise RuntimeError("No OpenCL GPU devices found")
    return devices


class Chain:
    def __init__(self):
        if not RPC_URLS:
            raise RuntimeError("POA_RPCS is empty")
        self.index = -1
        self.w3 = None
        self.contract = None
        self.rotate()

    def rotate(self):
        last = None
        for _ in RPC_URLS:
            self.index = (self.index + 1) % len(RPC_URLS)
            try:
                provider = Web3.HTTPProvider(
                    RPC_URLS[self.index],
                    request_kwargs={
                        "timeout": 20,
                        "headers": {"User-Agent": "proofarchitect-python-miner/1.0"},
                    },
                )
                w3 = Web3(provider)
                if int(w3.eth.chain_id) != CHAIN_ID:
                    raise RuntimeError(f"wrong chain id from {RPC_URLS[self.index]}")
                self.w3 = w3
                self.contract = w3.eth.contract(address=CONTRACT, abi=ABI)
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
                if attempt + 1 >= attempts:
                    break
                print(f"  RPC error: {exc}; rotating endpoint", flush=True)
                time.sleep(min(2**attempt, 8))
                self.rotate()
        raise last


def mine_round(devices, prefix, target, is_stale):
    stop = threading.Event()
    found = queue.Queue()
    hashes = [0] * len(devices)
    best = [0] * len(devices)
    started = time.time()
    last_time = started
    last_hashes = 0

    def worker(index, device):
        try:
            context = cl.Context([device])
            command_queue = cl.CommandQueue(context)
            program = cl.Program(context, KERNEL).build()
            kernel = cl.Kernel(program, "mine")
            flags = cl.mem_flags
            prefix_buf = cl.Buffer(
                context,
                flags.READ_ONLY | flags.COPY_HOST_PTR,
                hostbuf=np.frombuffer(prefix, dtype=np.uint8),
            )
            target_buf = cl.Buffer(
                context,
                flags.READ_ONLY | flags.COPY_HOST_PTR,
                hostbuf=np.frombuffer(int(target).to_bytes(32, "big"), dtype=np.uint8),
            )
            found_host = np.zeros(1, np.uint32)
            nonce_host = np.zeros(1, np.uint64)
            best_host = np.zeros(1, np.uint32)
            found_buf = cl.Buffer(context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=found_host)
            nonce_buf = cl.Buffer(context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=nonce_host)
            best_buf = cl.Buffer(context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=best_host)

            base = (secrets.randbits(64) + index * GLOBAL_SIZE) & ((1 << 64) - 1)
            iters_arg = np.uint32(ITERS)
            while not stop.is_set():
                found_host[0] = 0
                cl.enqueue_copy(command_queue, found_buf, found_host)
                kernel(
                    command_queue,
                    (GLOBAL_SIZE,),
                    None,
                    prefix_buf,
                    target_buf,
                    np.uint64(base),
                    found_buf,
                    nonce_buf,
                    best_buf,
                    iters_arg,
                )
                command_queue.finish()
                cl.enqueue_copy(command_queue, found_host, found_buf)
                cl.enqueue_copy(command_queue, nonce_host, nonce_buf)
                cl.enqueue_copy(command_queue, best_host, best_buf)
                command_queue.finish()
                batch = GLOBAL_SIZE * ITERS
                hashes[index] += batch
                best[index] = max(best[index], int(best_host[0]))
                if found_host[0]:
                    found.put(int(nonce_host[0]))
                    stop.set()
                    return
                base = (base + batch * len(devices)) & ((1 << 64) - 1)
        except Exception as exc:
            found.put(RuntimeError(f"GPU {index} ({device.name.strip()}) failed: {exc}"))
            stop.set()

    workers = [threading.Thread(target=worker, args=(i, device), daemon=True) for i, device in enumerate(devices)]
    for worker_thread in workers:
        worker_thread.start()

    try:
        while not stop.is_set():
            time.sleep(1)
            now = time.time()
            total = sum(hashes)
            rate = (total - last_hashes) / max(now - last_time, 0.001)
            last_hashes = total
            last_time = now
            probability = int(target) / float(1 << 256)
            expected = 1 / (rate * probability) if rate and probability else float("inf")
            chance = (1 - math.exp(-rate * 60 * probability)) * 100 if rate else 0.0
            print(
                f"  GPUs {len(devices)}  {fmt_rate(rate)}  hashes {total}  "
                f"best {max(best)}/256 bits  elapsed {fmt_time(now - started)}  "
                f"expected {fmt_time(expected)}  chance/min {chance:.2f}%",
                flush=True,
            )
            try:
                if is_stale():
                    stop.set()
                    return None
            except Exception as exc:
                # A transient RPC failure must not throw away a healthy GPU
                # round. The next interval will retry the live-target check.
                print(f"  stale check failed: {exc}; continuing", flush=True)

        item = found.get_nowait() if not found.empty() else None
        if isinstance(item, Exception):
            raise item
        return item
    finally:
        stop.set()
        for worker_thread in workers:
            worker_thread.join(timeout=5)


def snapshot(chain, address):
    def read(w3, contract):
        due, fee = contract.functions.currentMintDue().call()
        return {
            "due": int(due),
            "fee": int(fee),
            "wave": int(contract.functions.currentWave().call()),
            "minted": int(contract.functions.totalMinted().call()),
            "supply": int(contract.functions.maxSupply().call()),
            "paused": bool(contract.functions.mintPaused().call()),
            "bits": int(contract.functions.requiredBits(address).call()),
            "milli": int(contract.functions.requiredMilli(address).call()),
            "target": int(contract.functions.targetFor(address).call()),
        }

    return chain.retry(read)


def live_target(chain, address):
    def read(w3, contract):
        return bool(contract.functions.mintPaused().call()), int(
            contract.functions.targetFor(address).call()
        )

    return chain.retry(read)


def build_mint(chain, account, nonce, due):
    def read(w3, contract):
        txfn = contract.functions.mint(int(nonce))
        call = {"from": account.address, "value": int(due)}
        txfn.call(call)
        gas = int(
            w3.eth.estimate_gas(
                {
                    "from": account.address,
                    "to": CONTRACT,
                    "value": int(due),
                    "data": txfn._encode_transaction_data(),
                }
            )
        )
        block = w3.eth.get_block("latest")
        base_fee = int(block.get("baseFeePerGas") or 0)
        max_fee = max(FEE_FLOOR_WEI, base_fee * 2)
        priority = max_fee // 2
        balance = int(w3.eth.get_balance(account.address))
        pending_nonce = int(w3.eth.get_transaction_count(account.address, "pending"))
        required = int(due) + gas * max_fee
        if balance < required:
            raise RuntimeError(
                f"insufficient native USDC: balance={fmt_native(balance)} "
                f"required≈{fmt_native(required)} (mint={fmt_native(due)}, gas={gas}, maxFee={max_fee})"
            )
        tx = txfn.build_transaction(
            {
                "from": account.address,
                "value": int(due),
                "nonce": pending_nonce,
                "chainId": CHAIN_ID,
                "gas": gas,
                "type": 2,
                "maxPriorityFeePerGas": priority,
                "maxFeePerGas": max_fee,
            }
        )
        return tx, gas, max_fee, priority

    return chain.retry(read)


def main():
    private_key = os.getenv("POA_PRIVATE_KEY")
    if not private_key:
        raise SystemExit("ERROR: POA_PRIVATE_KEY is not set")
    if MAX_MINTS < 0:
        raise SystemExit("ERROR: POA_MAX_MINTS must be 0 or greater")
    if GLOBAL_SIZE < 1 or ITERS < 1:
        raise SystemExit("ERROR: POA_GLOBAL and POA_ITERS must both be positive")

    chain = Chain()
    account = chain.w3.eth.account.from_key(private_key)
    devices = gpu_devices()
    minted = 0
    print(f"Proof of Architect Python OpenCL miner | address={account.address} GPUs={len(devices)}")
    print(
        "GPUs: " + ", ".join(f"{i}:{device.name.strip()}" for i, device in enumerate(devices)),
        flush=True,
    )
    print(
        f"Arc chain={CHAIN_ID} contract={CONTRACT} feeFloor={FEE_FLOOR_WEI / 1e9:.0f} gwei "
        f"GLOBAL={GLOBAL_SIZE} ITERS={ITERS} "
        f"maxMints={'∞' if MAX_MINTS == 0 else MAX_MINTS}",
        flush=True,
    )

    while MAX_MINTS == 0 or minted < MAX_MINTS:
        state = snapshot(chain, account.address)
        if state["paused"]:
            raise RuntimeError("PoA minting is paused")
        if state["target"] <= 0:
            raise RuntimeError("contract returned an invalid zero target")

        high = secrets.token_bytes(24)
        prefix = (
            CHAIN_ID.to_bytes(32, "big")
            + bytes.fromhex(CONTRACT[2:])
            + bytes.fromhex(account.address[2:])
            + high
        )
        if len(prefix) != 96:
            raise AssertionError(f"unexpected static prefix length: {len(prefix)}")

        target = state["target"]
        print(
            f"[{time.strftime('%H:%M:%S')}] wave={state['wave']} minted={state['minted']}/{state['supply']} "
            f"required={state['bits']} bits ({state['milli']}/1000) work≈2^{work_bits(target):.2f} "
            f"due={fmt_native(state['due'])} fee={fmt_native(state['fee'])} "
            f"target=0x{target:064x}",
            flush=True,
        )

        def stale():
            paused, fresh_target = live_target(chain, account.address)
            return paused or fresh_target != target

        low_nonce = mine_round(devices, prefix, target, stale)
        if low_nonce is None:
            print("  live target changed -> remine", flush=True)
            continue

        nonce = (int.from_bytes(high, "big") << 64) | int(low_nonce)
        digest = work_hash(account.address, nonce)
        if int.from_bytes(digest, "big") >= target:
            print(
                f"  invalid GPU result discarded nonce={nonce} hash=0x{digest.hex()} target=0x{target:064x}",
                flush=True,
            )
            continue
        print(
            f"  FOUND nonce={nonce} hash=0x{digest.hex()} bits={leading_zero_bits(digest)}; verifying",
            flush=True,
        )

        fresh = snapshot(chain, account.address)
        if fresh["paused"] or fresh["target"] != target:
            print("  proof became stale before submit -> remine", flush=True)
            continue
        used = chain.retry(
            lambda w3, contract: contract.functions.nonceUsed(account.address, nonce).call()
        )
        if used:
            print("  nonce is already used -> remine", flush=True)
            continue

        try:
            due_now = int(
                chain.retry(lambda w3, contract: contract.functions.currentMintDue().call()[0])
            )
            tx, gas, max_fee, priority = build_mint(chain, account, nonce, due_now)
        except Exception as exc:
            if "insufficient native USDC" in str(exc):
                raise
            print(f"  mint preflight failed: {exc}; remine", flush=True)
            continue
        signed = account.sign_transaction(tx)
        tx_hash = chain.retry(lambda w3, contract: w3.eth.send_raw_transaction(signed.raw_transaction))
        print(
            f"  signed and submitted tx={tx_hash.hex()} value={fmt_native(due_now)} "
            f"gas={gas} maxFee={max_fee / 1e9:.2f}gwei priority={priority / 1e9:.2f}gwei",
            flush=True,
        )
        receipt = chain.retry(
            lambda w3, contract: w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        )
        print(
            f"  confirmed block={receipt.blockNumber} status={receipt.status} "
            f"gasUsed={receipt.gasUsed}",
            flush=True,
        )
        if int(receipt.status) != 1:
            raise RuntimeError("mint transaction reverted")
        minted += 1

    print(f"completed {minted} mint(s)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped by user")
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise
