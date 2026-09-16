#!/usr/bin/env python3
"""Tower of Babel multi-GPU OpenCL miner for Arc mainnet."""

import importlib.util
import math
import os
import queue
import random
import threading
from pathlib import Path
import time

import numpy as np
import pyopencl as cl
from Crypto.Hash import keccak
from web3 import Web3

ITERS = int(os.getenv("BABEL_ITERS", "16"))


# ── GPU helpers (re-use prspct for gpu_devices + leading_zero_bits) ──────────
shared_path = Path(__file__).resolve().parents[1] / "prspct-python" / "prspct_paid.py"
spec = importlib.util.spec_from_file_location("cathash_opencl", shared_path)
_gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_gpu)
gpu_devices = _gpu.gpu_devices
leading_zero_bits = _gpu.leading_zero_bits
STALE_SECONDS = float(os.getenv("BABEL_STALE_CHECK_SECONDS", "3"))
GLOBAL_SIZE = int(os.getenv("BABEL_GLOBAL", str(8 * 1024 * 1024)))


# ── Optimised kernel: each work-item loops ITERS times ───────────────────────
KERNEL = r"""
__constant uint RCL[24]={0x00000001u,0x00008082u,0x0000808au,0x80008000u,0x0000808bu,0x80000001u,0x80008081u,0x00008009u,0x0000008au,0x00000088u,0x80008009u,0x8000000au,0x8000808bu,0x0000008bu,0x00008089u,0x00008003u,0x00008002u,0x00000080u,0x0000800au,0x8000000au,0x80008081u,0x00008080u,0x80000001u,0x80008008u};
__constant uint RCH[24]={0x00000000u,0x00000000u,0x80000000u,0x80000000u,0x00000000u,0x00000000u,0x80000000u,0x80000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x80000000u,0x80000000u,0x80000000u,0x80000000u,0x80000000u,0x00000000u,0x80000000u,0x80000000u,0x80000000u,0x00000000u,0x80000000u};
__constant int PI[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
__constant int RH[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
void perm(uint s[50]){
 for(int r=0;r<24;r++){
  uint c0=s[0]^s[10]^s[20]^s[30]^s[40];uint c1=s[2]^s[12]^s[22]^s[32]^s[42];uint c2=s[4]^s[14]^s[24]^s[34]^s[44];uint c3=s[6]^s[16]^s[26]^s[36]^s[46];uint c4=s[8]^s[18]^s[28]^s[38]^s[48];
  uint g0=s[1]^s[11]^s[21]^s[31]^s[41];uint g1=s[3]^s[13]^s[23]^s[33]^s[43];uint g2=s[5]^s[15]^s[25]^s[35]^s[45];uint g3=s[7]^s[17]^s[27]^s[37]^s[47];uint g4=s[9]^s[19]^s[29]^s[39]^s[49];
  uint d0=c4^((c1<<1)|(g1>>31));uint e0=g4^((g1<<1)|(c1>>31));
  uint d1=c0^((c2<<1)|(g2>>31));uint e1=g0^((g2<<1)|(c2>>31));
  uint d2=c1^((c3<<1)|(g3>>31));uint e2=g1^((g3<<1)|(c3>>31));
  uint d3=c2^((c4<<1)|(g4>>31));uint e3=g2^((g4<<1)|(c4>>31));
  uint d4=c3^((c0<<1)|(g0>>31));uint e4=g3^((g0<<1)|(c0>>31));
  s[0]^=d0;s[1]^=e0;s[2]^=d1;s[3]^=e1;s[4]^=d2;s[5]^=e2;s[6]^=d3;s[7]^=e3;s[8]^=d4;s[9]^=e4;
  s[10]^=d0;s[11]^=e0;s[12]^=d1;s[13]^=e1;s[14]^=d2;s[15]^=e2;s[16]^=d3;s[17]^=e3;s[18]^=d4;s[19]^=e4;
  s[20]^=d0;s[21]^=e0;s[22]^=d1;s[23]^=e1;s[24]^=d2;s[25]^=e2;s[26]^=d3;s[27]^=e3;s[28]^=d4;s[29]^=e4;
  s[30]^=d0;s[31]^=e0;s[32]^=d1;s[33]^=e1;s[34]^=d2;s[35]^=e2;s[36]^=d3;s[37]^=e3;s[38]^=d4;s[39]^=e4;
  s[40]^=d0;s[41]^=e0;s[42]^=d1;s[43]^=e1;s[44]^=d2;s[45]^=e2;s[46]^=d3;s[47]^=e3;s[48]^=d4;s[49]^=e4;
  uint t0=s[2],t1=s[3];
  for(int i=0;i<24;i++){int q=PI[i];int rt=RH[i];uint u0=s[2*q],u1=s[2*q+1],x0=t0,x1=t1;if(rt>=32){rt-=32;x0=t1;x1=t0;}if(rt==0){s[2*q]=x0;s[2*q+1]=x1;}else{s[2*q]=(x0<<rt)|(x1>>(32-rt));s[2*q+1]=(x1<<rt)|(x0>>(32-rt));}t0=u0;t1=u1;}
  for(int y=0;y<5;y++){
   int b=y*10;
   uint x0=s[b],x1=s[b+2],x2=s[b+4],x3=s[b+6],x4=s[b+8];
   uint y0=s[b+1],y1=s[b+3],y2=s[b+5],y3=s[b+7],y4=s[b+9];
   s[b]=x0^((~x1)&x2);s[b+1]=y0^((~y1)&y2);
   s[b+2]=x1^((~x2)&x3);s[b+3]=y1^((~y2)&y3);
   s[b+4]=x2^((~x3)&x4);s[b+5]=y2^((~y3)&y4);
   s[b+6]=x3^((~x4)&x0);s[b+7]=y3^((~y4)&y0);
   s[b+8]=x4^((~x0)&x1);s[b+9]=y4^((~y0)&y1);
  }
  s[0]^=RCL[r];s[1]^=RCH[r];
 }
}
__kernel void mine(__global const uchar* prefix,__global const uchar* target,ulong base,volatile __global uint* found,__global ulong* nonce,__global uint* bestbits,uint iters){
 ulong n=base+(ulong)get_global_id(0);
 uchar m[136]; for(int i=0;i<136;i++)m[i]=0; for(int i=0;i<76;i++)m[i]=prefix[i];
 for(int i=0;i<8;i++)m[83-i]=(uchar)(n>>(8*i)); m[84]=1; m[135]=0x80;
 for(uint it=0;it<iters;it++){
  uint a[50]; for(int i=0;i<50;i++)a[i]=0;
  for(int i=0;i<17;i++){
   uint v0=((uint)m[8*i])|((uint)m[8*i+1]<<8)|((uint)m[8*i+2]<<16)|((uint)m[8*i+3]<<24);
   uint v1=((uint)m[8*i+4])|((uint)m[8*i+5]<<8)|((uint)m[8*i+6]<<16)|((uint)m[8*i+7]<<24);
   a[2*i]^=v0;a[2*i+1]^=v1;
  }
  perm(a);
  uchar h[32];
  for(int i=0;i<4;i++){
   uint v0=a[2*i],v1=a[2*i+1];
   h[8*i]=(uchar)v0;h[8*i+1]=(uchar)(v0>>8);h[8*i+2]=(uchar)(v0>>16);h[8*i+3]=(uchar)(v0>>24);
   h[8*i+4]=(uchar)v1;h[8*i+5]=(uchar)(v1>>8);h[8*i+6]=(uchar)(v1>>16);h[8*i+7]=(uchar)(v1>>24);
  }
  uint bits=0;for(int i=0;i<32;i++){if(h[i]==0)bits+=8;else{bits+=clz((uint)h[i])-24;break;}} atomic_max(bestbits,bits);
  int less=0;for(int i=0;i<32;i++){if(h[i]<target[i]){less=1;break;}if(h[i]>target[i])break;} if(less&&atomic_cmpxchg(found,0,1)==0)nonce[0]=n;
  if(found[0])return;
  n+=get_global_size(0);
  for(int i=0;i<8;i++)m[83-i]=(uchar)(n>>(8*i));
 }
}
"""


def k256(data):
    return keccak.new(data=data, digest_bits=256).digest()


def proof_hash(seed, address, nonce):
    packed = bytes(seed) + bytes.fromhex(address[2:]) + int(nonce).to_bytes(32, "big")
    return k256(packed)


def fmt_rate(rate):
    units = ["H/s", "KH/s", "MH/s", "GH/s", "TH/s"]
    i = 0
    while rate >= 1000 and i < len(units) - 1:
        rate /= 1000; i += 1
    return f"{rate:.2f} {units[i]}"


def fmt_time(seconds):
    if not math.isfinite(seconds): return "---"
    if seconds < 60: return f"{seconds:.1f}s"
    if seconds < 3600: return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def mine_round(devs, prefix, target, is_stale):
    stop = threading.Event(); found = queue.Queue()
    stats = [0] * len(devs); best = [0] * len(devs)
    started = last = time.time(); last_total = 0

    def worker(index, dev):
        try:
            ctx = cl.Context([dev]); cq = cl.CommandQueue(ctx)
            program = cl.Program(ctx, KERNEL).build(); kernel = cl.Kernel(program, "mine")
            mf = cl.mem_flags
            prefix_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                   hostbuf=np.frombuffer(prefix, dtype=np.uint8))
            target_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                   hostbuf=np.frombuffer(int(target).to_bytes(32, "big"), dtype=np.uint8))
            flag = np.zeros(1, np.uint32); nonce = np.zeros(1, np.uint64); bits = np.zeros(1, np.uint32)
            flag_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=flag)
            nonce_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nonce)
            bits_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bits)
            base = (random.getrandbits(64) + index * GLOBAL_SIZE) & ((1 << 64) - 1)
            iters_arg = np.uint32(ITERS)
            while not stop.is_set():
                flag[0] = 0; cl.enqueue_copy(cq, flag_buf, flag)
                kernel(cq, (GLOBAL_SIZE,), None,
                       prefix_buf, target_buf, np.uint64(base),
                       flag_buf, nonce_buf, bits_buf, iters_arg)
                cq.finish()
                cl.enqueue_copy(cq, flag, flag_buf)
                cl.enqueue_copy(cq, nonce, nonce_buf)
                cl.enqueue_copy(cq, bits, bits_buf)
                cq.finish()
                batch = GLOBAL_SIZE * ITERS
                stats[index] += batch
                best[index] = max(best[index], int(bits[0]))
                if flag[0]:
                    found.put(int(nonce[0])); stop.set(); return
                base = (base + batch * len(devs)) & ((1 << 64) - 1)
        except Exception as exc:
            found.put(exc); stop.set()

    workers = [threading.Thread(target=worker, args=(i, d), daemon=True)
               for i, d in enumerate(devs)]
    for t in workers: t.start()
    try:
        next_stale = time.time() + STALE_SECONDS
        while not stop.is_set():
            time.sleep(1); now = time.time()
            total = sum(stats); speed = (total - last_total) / max(now - last, 0.001)
            last_total = total; last = now
            probability = int(target) / (2 ** 256)
            expected = 1 / (speed * probability) if speed and probability else float("inf")
            chance = (1 - math.exp(-speed * 60 * probability)) * 100 if speed else 0
            print(f"  GPUs {len(devs)}  {fmt_rate(speed)}  hashes {total}  "
                  f"best {max(best)}/256 bits  elapsed {fmt_time(now - started)}  "
                  f"expected {fmt_time(expected)}  chance/min {chance:.2f}%", flush=True)
            if now >= next_stale:
                next_stale = now + STALE_SECONDS
                try:
                    if is_stale(): stop.set(); return None
                except Exception as exc:
                    print(f"  stale check failed: {exc}; continuing", flush=True)
        item = found.get_nowait() if not found.empty() else None
        if isinstance(item, Exception): raise item
        return item
    finally:
        stop.set()
        for t in workers: t.join(timeout=3)


# ── Chain + contract ─────────────────────────────────────────────────────────
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
        self.index = -1; self.w3 = None; self.contract = None; self.rotate()

    def rotate(self):
        last = None
        for _ in RPC_URLS:
            self.index = (self.index + 1) % len(RPC_URLS)
            try:
                candidate = Web3(Web3.HTTPProvider(
                    RPC_URLS[self.index], request_kwargs={"timeout": 20}))
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
                time.sleep(min(2 ** attempt, 8)); self.rotate()
        raise last


def state_dict(raw):
    return {name: raw[index] for index, name in enumerate(STATE_FIELDS)}


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
    devices = gpu_devices()
    completed = 0; coin_bps = COIN_PERCENT * 100
    print(
        f"Tower of Babel Python OpenCL miner | address={account.address} "
        f"GPUs={len(devices)} pay={COIN_PERCENT}% mine={100-COIN_PERCENT}% "
        f"GLOBAL={GLOBAL_SIZE} ITERS={ITERS}",
        flush=True,
    )
    print("GPUs: " + ", ".join(
        f"{index}:{device.name.strip()}" for index, device in enumerate(devices)
    ), flush=True)

    while MAX_MINTS == 0 or completed < MAX_MINTS:
        state = state_dict(chain.retry(lambda w3, contract: contract.functions.state().call()))
        laid = int(state["laid"]); seed = bytes(state["seed"])
        if laid >= MAX_SUPPLY:
            print("the Tower of Babel is complete", flush=True); return
        if seed == bytes(32):
            raise RuntimeError("mining is not open: contract returned a zero seed")
        price = int(state["price"]); payment = price * coin_bps // 10000
        target = int(chain.retry(
            lambda w3, contract: contract.functions.targetAt(laid, coin_bps).call()))
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
                lambda w3, contract: contract.functions.state().call(), 2))
            return int(fresh["laid"]) != laid or bytes(fresh["seed"]) != seed

        nonce_low = mine_round(devices, prefix, target, stale)
        if nonce_low is None:
            print("  another brick was laid -> remine", flush=True); continue
        nonce = (int.from_bytes(nonce_high, "big") << 64) | nonce_low

        digest = proof_hash(seed, account.address, nonce)
        if int.from_bytes(digest, "big") >= target:
            print(f"  invalid GPU result discarded nonce={nonce} hash=0x{digest.hex()}", flush=True)
            continue
        print(
            f"  FOUND nonce={nonce} hash=0x{digest.hex()} "
            f"bits={leading_zero_bits(digest)}; verifying",
            flush=True,
        )

        fresh = state_dict(chain.retry(lambda w3, contract: contract.functions.state().call()))
        if int(fresh["laid"]) != laid or bytes(fresh["seed"]) != seed:
            print("  proof became stale before submit -> remine", flush=True); continue
        fresh_price = int(fresh["price"]); value = fresh_price * coin_bps // 10000
        txfn = chain.contract.functions.lay(SPONSOR, nonce)
        call = {"from": account.address, "value": value}
        chain.retry(lambda w3, contract: txfn.call(call))
        gas = int(chain.retry(lambda w3, contract: w3.eth.estimate_gas({
            "from": account.address, "to": CONTRACT, "value": value,
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
            "from": account.address, "value": value,
            "nonce": chain.retry(
                lambda w3, contract: w3.eth.get_transaction_count(account.address, "pending")),
            "chainId": CHAIN_ID, "gas": gas, "type": 2,
            "maxPriorityFeePerGas": PRIORITY_FEE, "maxFeePerGas": max_fee,
        })
        signed = account.sign_transaction(transaction)
        txhash = chain.retry(
            lambda w3, contract: w3.eth.send_raw_transaction(signed.raw_transaction))
        print(f"  signed and submitted tx={txhash.hex()}", flush=True)
        receipt = chain.retry(
            lambda w3, contract: w3.eth.wait_for_transaction_receipt(txhash, timeout=180))
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
