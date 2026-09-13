#!/usr/bin/env python3
"""ShibaHash multi-GPU OpenCL miner for Robinhood Chain (4663).

The worker hashes the official packed preimage:
  address || uint256(nonce) || prevWork || anchor

It verifies candidates locally and with the contract's pure workHash() before
simulating and signing mine(uint256,uint256). It does not trust a local price
formula or an ABI heuristic.
"""

import os, sys, time, threading, queue, socket
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyopencl as cl
from Crypto.Hash import keccak
from web3 import Web3
from web3.exceptions import ContractLogicError

RPC = os.getenv("SHIBAHASH_RPC", "https://rpc.mainnet.chain.robinhood.com")
CONTRACT = Web3.to_checksum_address("0xF46A1d2eDDD1004B1345F3D0B5A2Db8e28939c67")
CHAIN_ID = 4663
ABI = [
 {"inputs":[],"name":"currentAnchor","outputs":[{"type":"uint256"},{"type":"bytes32"}],"stateMutability":"view","type":"function"},
 {"inputs":[],"name":"prevWork","outputs":[{"type":"bytes32"}],"stateMutability":"view","type":"function"},
 {"inputs":[{"name":"miner","type":"address"}],"name":"targetFor","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
 {"inputs":[],"name":"mintPrice","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
 {"inputs":[{"name":"miner","type":"address"},{"name":"nonce","type":"uint256"},{"name":"prev","type":"bytes32"},{"name":"anchor","type":"bytes32"}],"name":"workHash","outputs":[{"type":"bytes32"}],"stateMutability":"pure","type":"function"},
 {"inputs":[{"name":"nonce","type":"uint256"},{"name":"anchorBlock","type":"uint256"}],"name":"mine","outputs":[{"type":"uint256"}],"stateMutability":"payable","type":"function"},
]

KERNEL = r"""
__constant ulong RC[24]={1UL,0x8082UL,0x800000000000808aUL,0x8000000080008000UL,0x808bUL,0x80000001UL,0x8000000080008081UL,0x8000000000008009UL,0x8aUL,0x88UL,0x80008009UL,0x8000000aUL,0x8000808bUL,0x800000000000008bUL,0x8000000000008089UL,0x8000000000008003UL,0x8000000000008002UL,0x8000000000000080UL,0x800aUL,0x800000008000000aUL,0x8000000080008081UL,0x8000000000008080UL,0x80000001UL,0x8000000080008008UL};
__constant int PI[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
__constant int RH[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
ulong rol(ulong x,int n){return n?((x<<n)|(x>>(64-n))):x;}
void perm(ulong a[25]){for(int r=0;r<24;r++){ulong c[5],d[5];for(int x=0;x<5;x++)c[x]=a[x]^a[x+5]^a[x+10]^a[x+15]^a[x+20];for(int x=0;x<5;x++)d[x]=c[(x+4)%5]^rol(c[(x+1)%5],1);for(int i=0;i<25;i++)a[i]^=d[i%5];ulong t=a[1],u;for(int i=0;i<24;i++){u=a[PI[i]];a[PI[i]]=rol(t,RH[i]);t=u;}for(int y=0;y<5;y++){ulong b0=a[5*y],b1=a[1+5*y],b2=a[2+5*y],b3=a[3+5*y],b4=a[4+5*y];a[5*y]=b0^((~b1)&b2);a[1+5*y]=b1^((~b2)&b3);a[2+5*y]=b2^((~b3)&b4);a[3+5*y]=b3^((~b4)&b0);a[4+5*y]=b4^((~b0)&b1);}a[0]^=RC[r];}}
__kernel void mine(__global const uchar* prefix, __global const uchar* target, ulong base, volatile __global uint* found, __global ulong* nonce, __global ulong* best, __global uint* bestbits){ulong n=base+(ulong)get_global_id(0);uchar m[136];for(int i=0;i<136;i++)m[i]=0;for(int i=0;i<116;i++)m[i]=prefix[i];for(int i=0;i<8;i++)m[51-i]=(uchar)(n>>(8*i));m[116]=1;m[135]=0x80;ulong a[25];for(int i=0;i<25;i++)a[i]=0;for(int i=0;i<17;i++){ulong v=0;for(int k=0;k<8;k++)v|=((ulong)m[i*8+k])<<(8*k);a[i]^=v;}perm(a);uchar h[32];for(int i=0;i<4;i++)for(int k=0;k<8;k++)h[i*8+k]=(uchar)(a[i]>>(8*k));uint bits=0;for(int i=0;i<32;i++){if(h[i]==0)bits+=8;else{bits+=clz((uint)h[i])-24;break;}}atomic_max(bestbits,bits);int less=0;for(int i=0;i<32;i++){if(h[i]<target[i]){less=1;break;}if(h[i]>target[i])break;}if(less&&atomic_cmpxchg(found,0,1)==0){nonce[0]=n;for(int i=0;i<4;i++)best[i]=a[i];}}
"""

def keccak256(data):
    return keccak.new(data=data, digest_bits=256).digest()

def fmt_time(seconds):
    if not np.isfinite(seconds): return "—"
    if seconds < 60: return f"{seconds:.1f}s"
    if seconds < 3600: return f"{seconds/60:.1f}m"
    if seconds < 86400: return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"

def leading_bits(h):
    n=0
    for b in h:
        if b == 0: n += 8
        else: return n + (8-len(bin(b)[2:]))
    return 256

def devices():
    out=[]
    for platform in cl.get_platforms():
        for device in platform.get_devices(device_type=cl.device_type.GPU): out.append(device)
    if not out: raise RuntimeError("No OpenCL GPU devices found")
    return out

def make_prefix(address, prev, anchor):
    p=bytearray(116); p[:20]=bytes.fromhex(address[2:]); p[52:84]=bytes(prev); p[84:116]=bytes(anchor); return p

def cpu_verify(address, nonce, prev, anchor, target):
    data=bytes.fromhex(address[2:])+int(nonce).to_bytes(32,"big")+bytes(prev)+bytes(anchor)
    h=keccak256(data); return int.from_bytes(h,"big") < int(target), h

def mine_round(devs, prefix, target, stale):
    stop=threading.Event(); result=queue.Queue(); stats=[0]*len(devs); best=[0]*len(devs); start=time.time(); last=start; last_total=0
    batch=int(os.getenv("SHIBAHASH_BATCH", str(1<<20))); global_size=int(os.getenv("SHIBAHASH_GLOBAL", str(1<<20)))
    def worker(i, dev):
        ctx=cl.Context([dev]); q=cl.CommandQueue(ctx); prg=cl.Program(ctx,KERNEL).build(); k=cl.Kernel(prg,"mine"); mf=cl.mem_flags
        pb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.frombuffer(prefix,dtype=np.uint8)); tb=cl.Buffer(ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.frombuffer(int(target).to_bytes(32,"big"),dtype=np.uint8))
        fg=np.zeros(1,np.uint32); nn=np.zeros(1,np.uint64); bb=np.zeros(4,np.uint64); bs=np.zeros(1,np.uint32); fgb=cl.Buffer(ctx,mf.READ_WRITE|mf.COPY_HOST_PTR,hostbuf=fg); nb=cl.Buffer(ctx,mf.READ_WRITE|mf.COPY_HOST_PTR,hostbuf=nn); bsb=cl.Buffer(ctx,mf.READ_WRITE|mf.COPY_HOST_PTR,hostbuf=bs); bbx=cl.Buffer(ctx,mf.READ_WRITE|mf.COPY_HOST_PTR,hostbuf=bb)
        base=(int.from_bytes(os.urandom(8),"big")+i)&((1<<64)-1)
        while not stop.is_set():
            fg[0]=0; cl.enqueue_copy(q,fg,fgb); k(q,(global_size,),None,pb,tb,np.uint64(base),fgb,nb,bbx,bsb); q.finish(); cl.enqueue_copy(q,fg,fgb); cl.enqueue_copy(q,nn,nb); cl.enqueue_copy(q,bs,bsb); q.finish(); stats[i]+=global_size; best[i]=max(best[i],int(bs[0]))
            if fg[0]: result.put(int(nn[0])); stop.set(); return
            base=(base+global_size*len(devs))&((1<<64)-1)
    threads=[threading.Thread(target=worker,args=(i,d),daemon=True) for i,d in enumerate(devs)]
    for t in threads:t.start()
    try:
        while not stop.is_set():
            time.sleep(2); total=sum(stats); now=time.time(); speed=(total-last_total)/max(now-last,.1); last_total,last=total,now; p=float(target)/(2**256); chance=(1-(1-p)**(speed*60))*100; expected=1/(speed*p) if speed and p else float("inf"); print(f"  GPUs {len(devs)}  {speed/1e9:.2f} GH/s  hashes {total}  best {max(best)}/256 bits  expected {fmt_time(expected)}  chance/min {chance:.2f}%",flush=True)
            try:
                if stale(): stop.set(); return None
            except Exception as e: print(f"  RPC check failed: {e}; continuing",flush=True)
        return result.get_nowait() if not result.empty() else None
    finally:
        stop.set()
        for t in threads:t.join(timeout=3)

def main():
    pk=os.getenv("SHIBAHASH_PRIVATE_KEY")
    if not pk: raise SystemExit("Set SHIBAHASH_PRIVATE_KEY")
    w3=Web3(Web3.HTTPProvider(RPC,request_kwargs={"timeout":20})); acct=w3.eth.account.from_key(pk); c=w3.eth.contract(CONTRACT,abi=ABI); devs=devices(); print(f"ShibaHash Python OpenCL miner | address={acct.address} GPUs={len(devs)}")
    while True:
        block,anchor=c.functions.currentAnchor().call(); prev=c.functions.prevWork().call(); target=c.functions.targetFor(acct.address).call(); price=c.functions.mintPrice().call(); print(f"[{time.strftime('%H:%M:%S')}] challenge anchor={block} target=0x{target:064x} priceWei={price}")
        prefix=make_prefix(acct.address,prev,anchor); stale=lambda: c.functions.prevWork().call()!=prev or c.functions.currentAnchor().call()[0]-block>=180
        nonce=mine_round(devs,prefix,target,stale)
        if nonce is None: print("  challenge changed -> remine"); continue
        ok,h=cpu_verify(acct.address,nonce,prev,anchor,target)
        onchain=c.functions.workHash(acct.address,nonce,prev,anchor).call()
        if not ok or bytes(onchain)!=h: print(f"  invalid GPU result discarded nonce={nonce} hash=0x{h.hex()}"); continue
        price=c.functions.mintPrice().call(); txfn=c.functions.mine(nonce,block); txfn.call({"from":acct.address,"value":price}); tx=txfn.build_transaction({"from":acct.address,"value":price,"nonce":w3.eth.get_transaction_count(acct.address,"pending"),"chainId":CHAIN_ID,"gas":w3.eth.estimate_gas({"from":acct.address,"to":CONTRACT,"value":price,"data":txfn._encode_transaction_data()})}); latest=w3.eth.get_block("latest"); tx.update({"type":2,"maxPriorityFeePerGas":int(os.getenv("SHIBAHASH_PRIORITY_FEE_WEI","2000000")),"maxFeePerGas":int(latest.get("baseFeePerGas",0))*2+int(os.getenv("SHIBAHASH_PRIORITY_FEE_WEI","2000000"))}); signed=acct.sign_transaction(tx); txh=w3.eth.send_raw_transaction(signed.raw_transaction); print(f"  signed and submitted tx={txh.hex()}"); receipt=w3.eth.wait_for_transaction_receipt(txh); print(f"  confirmed block={receipt.blockNumber} status={receipt.status}")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\nstopped by user")
    except Exception as e: print(f"ERROR: {e}"); raise
