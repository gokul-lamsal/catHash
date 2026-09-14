#!/usr/bin/env python3
"""MinerPotatos multi-GPU OpenCL miner for Robinhood Chain (4663)."""

import importlib.util
import math
import os
from pathlib import Path
import queue
import random
import threading
import time

import numpy as np
import pyopencl as cl
from Crypto.Hash import keccak
from web3 import Web3


# Reuse the tested OpenCL device discovery and display helpers.
shared_path = Path(__file__).resolve().parents[1] / "prspct-python" / "prspct_paid.py"
spec = importlib.util.spec_from_file_location("cathash_opencl", shared_path)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)
GLOBAL_SIZE = int(os.getenv("MINERPOTATOS_GLOBAL", str(1 << 22)))
LOCAL_SIZE = int(os.getenv("MINERPOTATOS_LOCAL", "256"))
NONCES_PER_ITEM = int(os.getenv("MINERPOTATOS_NONCES_PER_ITEM", "64"))
STALE_SECONDS = float(os.getenv("MINERPOTATOS_STALE_CHECK_SECONDS", "3"))

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

# This proof occupies exactly one Keccak-256 rate block. Lanes 0..12 are
# invariant; lanes 13 and 14 contain the low 64-bit nonce. Constructing those
# lanes directly avoids rebuilding and copying 136 private bytes for each hash.
FAST_KERNEL = r"""
__constant ulong RC[24]={1UL,0x8082UL,0x800000000000808aUL,0x8000000080008000UL,0x808bUL,0x80000001UL,0x8000000080008081UL,0x8000000000008009UL,0x8aUL,0x88UL,0x80008009UL,0x8000000aUL,0x8000808bUL,0x800000000000008bUL,0x8000000000008089UL,0x8000000000008003UL,0x8000000000008002UL,0x8000000000000080UL,0x800aUL,0x800000008000000aUL,0x8000000080008081UL,0x8000000000008080UL,0x80000001UL,0x8000000080008008UL};
__constant int PI[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
__constant int RH[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
ulong rol(ulong x,int n){return (x<<n)|(x>>(64-n));}
uint sw32(uint x){return ((x&0xffU)<<24)|((x&0xff00U)<<8)|((x>>8)&0xff00U)|(x>>24);}
ulong sw64(ulong x){return ((ulong)sw32((uint)x)<<32)|(ulong)sw32((uint)(x>>32));}
void perm(ulong a[25]){for(int r=0;r<24;r++){ulong c[5],d[5];for(int x=0;x<5;x++)c[x]=a[x]^a[x+5]^a[x+10]^a[x+15]^a[x+20];for(int x=0;x<5;x++)d[x]=c[(x+4)%5]^rol(c[(x+1)%5],1);for(int i=0;i<25;i++)a[i]^=d[i%5];ulong t=a[1],u;for(int i=0;i<24;i++){u=a[PI[i]];a[PI[i]]=rol(t,RH[i]);t=u;}for(int y=0;y<5;y++){ulong b0=a[5*y],b1=a[1+5*y],b2=a[2+5*y],b3=a[3+5*y],b4=a[4+5*y];a[5*y]=b0^((~b1)&b2);a[1+5*y]=b1^((~b2)&b3);a[2+5*y]=b2^((~b3)&b4);a[3+5*y]=b3^((~b4)&b0);a[4+5*y]=b4^((~b0)&b1);}a[0]^=RC[r];}}
__kernel void mine(__global const ulong* base_state,__global const ulong* target,ulong base,uint iterations,volatile __global uint* found,__global ulong* nonce,__global uint* bestbits){
 ulong first=base+(ulong)get_global_id(0)*(ulong)iterations;
 for(uint iteration=0;iteration<iterations;iteration++){
 ulong n=first+(ulong)iteration; ulong a[25];
 for(int i=0;i<13;i++)a[i]=base_state[i];
 a[13]=((ulong)sw32((uint)(n>>32)))<<32;
 a[14]=(ulong)sw32((uint)n)|0x0000000100000000UL;
 a[15]=0UL;a[16]=0x8000000000000000UL;for(int i=17;i<25;i++)a[i]=0UL;
 perm(a);
 ulong h0=sw64(a[0]); uint bits=clz(h0);
 if(bits>=24)atomic_max(bestbits,bits);
 int less=0;
 if(h0<target[0])less=1;
 else if(h0==target[0]){ulong h1=sw64(a[1]);if(h1<target[1])less=1;else if(h1==target[1]){ulong h2=sw64(a[2]);if(h2<target[2])less=1;else if(h2==target[2]){ulong h3=sw64(a[3]);if(h3<target[3])less=1;}}}
 if(less){if(atomic_cmpxchg(found,0,1)==0)nonce[0]=n;return;}
 }
}
"""

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


def mine_round(devices, prefix, target, is_stale):
    """Mine one job on all GPUs, with optimized one-block Keccak input."""
    if LOCAL_SIZE <= 0 or GLOBAL_SIZE < LOCAL_SIZE or GLOBAL_SIZE % LOCAL_SIZE:
        raise RuntimeError("MINERPOTATOS_GLOBAL must be a multiple of MINERPOTATOS_LOCAL")
    if NONCES_PER_ITEM <= 0:
        raise RuntimeError("MINERPOTATOS_NONCES_PER_ITEM must be positive")
    if len(prefix) != 116:
        raise RuntimeError(f"internal error: expected 116 prefix bytes, got {len(prefix)}")

    base_state = np.frombuffer(prefix[:104], dtype="<u8").copy()
    target_bytes = int(target).to_bytes(32, "big")
    target_words = np.array(
        [int.from_bytes(target_bytes[i:i + 8], "big") for i in range(0, 32, 8)],
        dtype=np.uint64,
    )
    stop = threading.Event()
    found = queue.Queue()
    stats = [0] * len(devices)
    best = [0] * len(devices)
    previous = [0] * len(devices)
    started = last_report = time.time()

    def worker(index, device):
        try:
            context = cl.Context([device])
            command_queue = cl.CommandQueue(context)
            program = cl.Program(context, FAST_KERNEL).build()
            kernel = cl.Kernel(program, "mine")
            flags = cl.mem_flags
            state_buffer = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=base_state)
            target_buffer = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=target_words)
            hit = np.zeros(1, np.uint32)
            nonce = np.zeros(1, np.uint64)
            bits = np.zeros(1, np.uint32)
            hit_buffer = cl.Buffer(context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=hit)
            nonce_buffer = cl.Buffer(context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=nonce)
            bits_buffer = cl.Buffer(context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=bits)
            base = (random.getrandbits(64) + index * GLOBAL_SIZE) & ((1 << 64) - 1)
            while not stop.is_set():
                hit[0] = 0
                cl.enqueue_copy(command_queue, hit_buffer, hit)
                kernel(
                    command_queue, (GLOBAL_SIZE,), (LOCAL_SIZE,), state_buffer, target_buffer,
                    np.uint64(base), np.uint32(NONCES_PER_ITEM), hit_buffer, nonce_buffer, bits_buffer,
                )
                command_queue.finish()
                cl.enqueue_copy(command_queue, hit, hit_buffer)
                cl.enqueue_copy(command_queue, nonce, nonce_buffer)
                cl.enqueue_copy(command_queue, bits, bits_buffer)
                command_queue.finish()
                hashes_per_launch = GLOBAL_SIZE * NONCES_PER_ITEM
                stats[index] += hashes_per_launch
                best[index] = max(best[index], int(bits[0]))
                if hit[0]:
                    found.put(int(nonce[0]))
                    stop.set()
                    return
                base = (base + hashes_per_launch * len(devices)) & ((1 << 64) - 1)
        except Exception as exc:
            found.put(RuntimeError(f"GPU {index} ({device.name.strip()}): {exc}"))
            stop.set()

    workers = [threading.Thread(target=worker, args=(i, device), daemon=True)
               for i, device in enumerate(devices)]
    for worker_thread in workers:
        worker_thread.start()
    try:
        next_stale = time.time() + STALE_SECONDS
        while not stop.is_set():
            time.sleep(1)
            now = time.time()
            interval = max(now - last_report, 0.001)
            rates = [(stats[i] - previous[i]) / interval for i in range(len(devices))]
            previous[:] = stats
            last_report = now
            speed = sum(rates)
            probability = int(target) / 2**256
            expected = 1 / (speed * probability) if speed and probability else float("inf")
            chance = (1 - math.exp(-speed * 60 * probability)) * 100 if speed else 0
            per_gpu = " ".join(f"{i}:{gpu.fmt_rate(rate)}" for i, rate in enumerate(rates))
            print(
                f"  GPUs {len(devices)}  total {gpu.fmt_rate(speed)}  [{per_gpu}]  "
                f"hashes {sum(stats)}  best {max(best)}/256 bits  "
                f"elapsed {gpu.fmt_time(now-started)}  expected {gpu.fmt_time(expected)}  "
                f"chance/min {chance:.2f}%",
                flush=True,
            )
            if now >= next_stale:
                next_stale = now + STALE_SECONDS
                try:
                    if is_stale():
                        stop.set()
                        return None
                except Exception as exc:
                    print(f"  stale check failed: {exc}; continuing", flush=True)
        item = found.get_nowait() if not found.empty() else None
        if isinstance(item, Exception):
            raise item
        return item
    finally:
        stop.set()
        for worker_thread in workers:
            worker_thread.join(timeout=3)


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

        nonce = mine_round(devices, prefix, target, stale)
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
