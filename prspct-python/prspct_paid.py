#!/usr/bin/env python3
"""PRSPCT multi-GPU OpenCL miner for Robinhood Chain (4663)."""

import math
import os
import queue
import random
import threading
import time

import numpy as np
import pyopencl as cl
from Crypto.Hash import keccak
from web3 import Web3

CHAIN_ID = 4663
CONTRACT = Web3.to_checksum_address("0xd078008c3D887A52CE722A3cA0539cA1F4971dD1")
RPC_URLS = [x.strip() for x in os.getenv("PRSPCT_RPCS", "https://rpc.mainnet.chain.robinhood.com,https://robinhood-rpc.publicnode.com,https://rpc.ordofi.network").split(",") if x.strip()]
COIN_PERCENT = float(os.getenv("PRSPCT_COIN_PERCENT", "0"))
GLOBAL_SIZE = int(os.getenv("PRSPCT_GLOBAL", str(1 << 20)))
STALE_SECONDS = float(os.getenv("PRSPCT_STALE_CHECK_SECONDS", "4"))
MAX_CLAIMS = int(os.getenv("PRSPCT_MAX_CLAIMS", "0"))
PRIORITY_FEE = int(os.getenv("PRSPCT_PRIORITY_FEE_WEI", "2000000"))

ABI = [
    {"inputs": [], "name": "state", "outputs": [{"components": [
        {"name":"depth","type":"uint256"},{"name":"seed","type":"bytes32"},{"name":"openAt","type":"uint256"},
        {"name":"price","type":"uint256"},{"name":"target","type":"uint256"},{"name":"coinWeight","type":"uint256"},
        {"name":"sweatWeight","type":"uint256"},{"name":"coinIn","type":"uint256"},{"name":"tillPaid","type":"uint256"},
        {"name":"day","type":"uint256"},{"name":"pot","type":"uint256"},{"name":"best","type":"bytes32"},
        {"name":"who","type":"address"},{"name":"companyOwed","type":"uint256"}], "name":"", "type":"tuple"}], "stateMutability":"view", "type":"function"},
    {"inputs":[{"name":"n","type":"uint256"}],"name":"priceOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"n","type":"uint256"},{"name":"valueWei","type":"uint256"}],"name":"targetOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"nonce","type":"uint256"}],"name":"claim","outputs":[{"type":"uint256"}],"stateMutability":"payable","type":"function"},
]

KERNEL = r"""
__constant ulong RC[24]={1UL,0x8082UL,0x800000000000808aUL,0x8000000080008000UL,0x808bUL,0x80000001UL,0x8000000080008081UL,0x8000000000008009UL,0x8aUL,0x88UL,0x80008009UL,0x8000000aUL,0x8000808bUL,0x800000000000008bUL,0x8000000000008089UL,0x8000000000008003UL,0x8000000000008002UL,0x8000000000000080UL,0x800aUL,0x800000008000000aUL,0x8000000080008081UL,0x8000000000008080UL,0x80000001UL,0x8000000080008008UL};
__constant int PI[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
__constant int RH[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
ulong rol(ulong x,int n){return n?((x<<n)|(x>>(64-n))):x;}
void perm(ulong a[25]){for(int r=0;r<24;r++){ulong c[5],d[5];for(int x=0;x<5;x++)c[x]=a[x]^a[x+5]^a[x+10]^a[x+15]^a[x+20];for(int x=0;x<5;x++)d[x]=c[(x+4)%5]^rol(c[(x+1)%5],1);for(int i=0;i<25;i++)a[i]^=d[i%5];ulong t=a[1],u;for(int i=0;i<24;i++){u=a[PI[i]];a[PI[i]]=rol(t,RH[i]);t=u;}for(int y=0;y<5;y++){ulong b0=a[5*y],b1=a[1+5*y],b2=a[2+5*y],b3=a[3+5*y],b4=a[4+5*y];a[5*y]=b0^((~b1)&b2);a[1+5*y]=b1^((~b2)&b3);a[2+5*y]=b2^((~b3)&b4);a[3+5*y]=b3^((~b4)&b0);a[4+5*y]=b4^((~b0)&b1);}a[0]^=RC[r];}}
__kernel void mine(__global const uchar* prefix,__global const uchar* target,ulong base,volatile __global uint* found,__global ulong* nonce,__global uint* bestbits){
 ulong n=base+(ulong)get_global_id(0); uchar m[136]; for(int i=0;i<136;i++)m[i]=0; for(int i=0;i<52;i++)m[i]=prefix[i];
 for(int i=0;i<8;i++)m[83-i]=(uchar)(n>>(8*i)); m[84]=1; m[135]=0x80; ulong a[25]; for(int i=0;i<25;i++)a[i]=0;
 for(int i=0;i<17;i++){ulong v=0;for(int k=0;k<8;k++)v|=((ulong)m[i*8+k])<<(8*k);a[i]^=v;} perm(a); uchar h[32];
 for(int i=0;i<4;i++)for(int k=0;k<8;k++)h[i*8+k]=(uchar)(a[i]>>(8*k)); uint bits=0;for(int i=0;i<32;i++){if(h[i]==0)bits+=8;else{bits+=clz((uint)h[i])-24;break;}} atomic_max(bestbits,bits);
 int less=0;for(int i=0;i<32;i++){if(h[i]<target[i]){less=1;break;}if(h[i]>target[i])break;} if(less&&atomic_cmpxchg(found,0,1)==0)nonce[0]=n;
}
"""


def k256(data):
    return keccak.new(data=data, digest_bits=256).digest()


def work_hash(seed, address, nonce):
    return k256(bytes(seed) + bytes.fromhex(address[2:]) + int(nonce).to_bytes(32, "big"))


def leading_zero_bits(digest):
    value = int.from_bytes(digest, "big")
    return 256 if value == 0 else 256 - value.bit_length()


def fmt_rate(rate):
    units = ["H/s", "KH/s", "MH/s", "GH/s", "TH/s"]
    i = 0
    while rate >= 1000 and i < len(units) - 1:
        rate /= 1000
        i += 1
    return f"{rate:.2f} {units[i]}"


def fmt_time(seconds):
    if not math.isfinite(seconds): return "—"
    if seconds < 60: return f"{seconds:.1f}s"
    if seconds < 3600: return f"{seconds/60:.1f}m"
    if seconds < 86400: return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"


def gpu_devices():
    result = []
    for platform in cl.get_platforms():
        result.extend(platform.get_devices(device_type=cl.device_type.GPU))
    if not result: raise RuntimeError("No OpenCL GPU devices found")
    return result


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
                w3 = Web3(Web3.HTTPProvider(RPC_URLS[self.index], request_kwargs={"timeout": 20}))
                w3.eth.block_number
                self.w3 = w3
                self.contract = w3.eth.contract(CONTRACT, abi=ABI)
                return
            except Exception as exc: last = exc
        raise RuntimeError(f"all RPC endpoints failed: {last}")

    def retry(self, fn, attempts=4):
        last = None
        for n in range(attempts):
            try: return fn(self.w3, self.contract)
            except Exception as exc:
                last = exc
                print(f"  RPC error: {exc}; retrying", flush=True)
                time.sleep(min(2 ** n, 8))
                self.rotate()
        raise last


def mine_round(devs, prefix, target, is_stale):
    stop = threading.Event(); found = queue.Queue(); stats = [0] * len(devs); best = [0] * len(devs)
    started = last = time.time(); last_total = 0

    def worker(index, dev):
        try:
            ctx=cl.Context([dev]); cq=cl.CommandQueue(ctx); program=cl.Program(ctx,KERNEL).build(); kernel=cl.Kernel(program,"mine"); mf=cl.mem_flags
            prefix_buf=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.frombuffer(prefix,dtype=np.uint8)); target_buf=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.frombuffer(int(target).to_bytes(32,"big"),dtype=np.uint8))
            flag=np.zeros(1,np.uint32); nonce=np.zeros(1,np.uint64); bits=np.zeros(1,np.uint32)
            flag_buf=cl.Buffer(ctx,mf.READ_WRITE|mf.COPY_HOST_PTR,hostbuf=flag); nonce_buf=cl.Buffer(ctx,mf.READ_WRITE|mf.COPY_HOST_PTR,hostbuf=nonce); bits_buf=cl.Buffer(ctx,mf.READ_WRITE|mf.COPY_HOST_PTR,hostbuf=bits)
            base=(random.getrandbits(64)+index*GLOBAL_SIZE)&((1<<64)-1)
            while not stop.is_set():
                flag[0]=0; cl.enqueue_copy(cq,flag_buf,flag)
                kernel(cq,(GLOBAL_SIZE,),None,prefix_buf,target_buf,np.uint64(base),flag_buf,nonce_buf,bits_buf); cq.finish()
                cl.enqueue_copy(cq,flag,flag_buf); cl.enqueue_copy(cq,nonce,nonce_buf); cl.enqueue_copy(cq,bits,bits_buf); cq.finish()
                stats[index]+=GLOBAL_SIZE; best[index]=max(best[index],int(bits[0]))
                if flag[0]: found.put(int(nonce[0])); stop.set(); return
                base=(base+GLOBAL_SIZE*len(devs))&((1<<64)-1)
        except Exception as exc:
            found.put(exc); stop.set()

    workers=[threading.Thread(target=worker,args=(i,d),daemon=True) for i,d in enumerate(devs)]
    for worker_thread in workers: worker_thread.start()
    try:
        next_stale=time.time()+STALE_SECONDS
        while not stop.is_set():
            time.sleep(1); now=time.time(); total=sum(stats); speed=(total-last_total)/max(now-last,0.001); last_total,total_before=total,total; last=now
            probability=int(target)/(2**256); expected=1/(speed*probability) if speed and probability else float("inf"); chance=(1-math.exp(-speed*60*probability))*100 if speed else 0
            print(f"  GPUs {len(devs)}  {fmt_rate(speed)}  hashes {total}  best {max(best)}/256 bits  elapsed {fmt_time(now-started)}  expected {fmt_time(expected)}  chance/min {chance:.2f}%",flush=True)
            if now>=next_stale:
                next_stale=now+STALE_SECONDS
                try:
                    if is_stale(): stop.set(); return None
                except Exception as exc: print(f"  stale check failed: {exc}; continuing",flush=True)
        item=found.get_nowait() if not found.empty() else None
        if isinstance(item,Exception): raise item
        return item
    finally:
        stop.set()
        for worker_thread in workers: worker_thread.join(timeout=3)


def main():
    private_key=os.getenv("PRSPCT_PRIVATE_KEY")
    if not private_key: raise SystemExit("ERROR: PRSPCT_PRIVATE_KEY is not set")
    if not 0 <= COIN_PERCENT < 100: raise SystemExit("ERROR: PRSPCT_COIN_PERCENT must be from 0 up to (but not including) 100")
    chain=Chain(); account=chain.w3.eth.account.from_key(private_key); devs=gpu_devices(); claims=0
    print(f"PRSPCT Python OpenCL miner | address={account.address} GPUs={len(devs)} coin={COIN_PERCENT:g}%")
    print("GPUs: " + ", ".join(f"{i}:{d.name.strip()}" for i,d in enumerate(devs)),flush=True)
    while MAX_CLAIMS == 0 or claims < MAX_CLAIMS:
        state=chain.retry(lambda w,c:c.functions.state().call()); depth=int(state[0]); seed=bytes(state[1]); price=int(state[3]); pay=int(price*COIN_PERCENT/100)
        target=int(chain.retry(lambda w,c:c.functions.targetOf(depth,pay).call()))
        if seed == bytes(32): raise RuntimeError("the PRSPCT mine is not open yet")
        print(f"[{time.strftime('%H:%M:%S')}] foot={depth+1} seed=0x{seed.hex()[:16]}… priceWei={price} payWei={pay} target=0x{target:064x}",flush=True)
        prefix=seed+bytes.fromhex(account.address[2:])
        def stale():
            fresh=chain.retry(lambda w,c:c.functions.state().call(),2)
            return int(fresh[0]) != depth or bytes(fresh[1]) != seed
        nonce=mine_round(devs,prefix,target,stale)
        if nonce is None: print("  somebody claimed this foot -> remine",flush=True); continue
        digest=work_hash(seed,account.address,nonce)
        if int.from_bytes(digest,"big") >= target:
            print(f"  invalid GPU result discarded nonce={nonce} hash=0x{digest.hex()}",flush=True); continue
        print(f"  FOUND nonce={nonce} hash=0x{digest.hex()} bits={leading_zero_bits(digest)}; verifying",flush=True)
        fresh=chain.retry(lambda w,c:c.functions.state().call())
        if int(fresh[0]) != depth or bytes(fresh[1]) != seed:
            print("  proof became stale before submit -> remine",flush=True); continue
        balance=chain.retry(lambda w,c:w.eth.get_balance(account.address)); latest=chain.retry(lambda w,c:w.eth.get_block("latest")); base_fee=int(latest.get("baseFeePerGas",0)); max_fee=base_fee*2+PRIORITY_FEE
        txfn=chain.contract.functions.claim(nonce)
        call={"from":account.address,"value":pay}; chain.retry(lambda w,c:txfn.call(call))
        gas=chain.retry(lambda w,c:w.eth.estimate_gas({"from":account.address,"to":CONTRACT,"value":pay,"data":txfn._encode_transaction_data()})); required=pay+gas*max_fee
        if balance < required: raise RuntimeError(f"insufficient funds: balanceWei={balance} requiredWei≈{required} (payWei={pay}, gas={gas})")
        tx=txfn.build_transaction({"from":account.address,"value":pay,"nonce":chain.retry(lambda w,c:w.eth.get_transaction_count(account.address,"pending")),"chainId":CHAIN_ID,"gas":gas,"type":2,"maxPriorityFeePerGas":PRIORITY_FEE,"maxFeePerGas":max_fee})
        signed=account.sign_transaction(tx); txhash=chain.retry(lambda w,c:w.eth.send_raw_transaction(signed.raw_transaction)); print(f"  signed and submitted tx={txhash.hex()}",flush=True)
        receipt=chain.retry(lambda w,c:w.eth.wait_for_transaction_receipt(txhash,timeout=180)); print(f"  confirmed block={receipt.blockNumber} status={receipt.status}",flush=True)
        if receipt.status != 1: raise RuntimeError("claim transaction reverted")
        claims+=1
    print(f"completed {claims} claim(s)")


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nstopped by user")
    except Exception as exc:
        print(f"ERROR: {exc}",flush=True)
        raise
