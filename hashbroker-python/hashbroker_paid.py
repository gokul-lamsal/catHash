#!/usr/bin/env python3
# ============================================================================
#  HashBroker 本地 GPU 挖矿器 (单文件, NVIDIA GPU)
#  链上 SHA-256 工作量证明 -> 支持付费 mint。挖到就自动用你的钱包提交。
#
#  依赖:  pip install pyopencl eth-account
#  用法:  设置环境变量 PK=你的钱包私钥(0x...), 然后 python hashbroker.py
#         可选 COUNT=挖几个(默认一直挖到开始收费)。停: Ctrl+C。
#
#  原理: preimage = 地址(20) || 零(24) || nonce(8) || challenge(32)  -> SHA-256
#        需前导零 bit >= 链上 currentDifficulty(); 提交 mine(nonce, challenge)。
#        challenge 每当有人 mint 就变 -> 本器盯着, 变了自动换新 challenge 重挖,
#        挖到后先验 challenge 未过期再提交(否则会被抢先/revert)。
#  ⚠️ 竞争: 难度随全局供应上涨, 算力要压过 challenge 刷新率才稳中。单卡靠运气窗口。
# ============================================================================
import os, sys, time, json, struct, urllib.request, urllib.error, threading, queue, socket
import numpy as np
import pyopencl as cl
from eth_account import Account

# ---- 配置 (换项目改这里) ----
RPC      = "https://rpc.mainnet.chain.robinhood.com"
CONTRACT = "0x4272D6f51771839F596082eF48fa84D35239Bab3"
CHAIN_ID = 4663
SEL_CHALLENGE  = "0xd2ef7398"   # challenge() -> bytes32
SEL_DIFFICULTY = "0x5c062d6c"   # currentDifficulty() -> uint256
SEL_PRICE      = "0x6817c76c"   # mintPrice() -> uint256
SEL_MINE       = "0xe43e322c"   # mine(uint256 nonce, bytes32 challenge)

PK = os.environ.get("HASHBROKER_PRIVATE_KEY", os.environ.get("PK", "")).strip()
if not PK:
    sys.exit("请设置环境变量 PK=你的钱包私钥 (0x...)")
acct = Account.from_key(PK)
ADDR = acct.address
COUNT = int(os.environ.get("COUNT", "999"))
MAX_PRICE_WEI = int(os.environ.get("HASHBROKER_MAX_PRICE_WEI", "0"))

# Some VPS networks advertise IPv6 while not routing it. Prefer IPv4 for RPC.
_getaddrinfo = socket.getaddrinfo
def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return [item for item in _getaddrinfo(host, port, family, type, proto, flags) if item[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_only

def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    # 注意: 很多 RPC 不带 User-Agent 会 403, 必须带上
    req = urllib.request.Request(RPC, data=body, headers={"content-type": "application/json", "user-agent": "Mozilla/5.0"})
    last_error = None
    for attempt in range(4):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=20).read())
            if "error" in r: raise RuntimeError(r["error"])
            return r["result"]
        except (OSError, urllib.error.URLError, RuntimeError) as error:
            last_error = error
            if attempt < 3: time.sleep(1 + attempt)
    raise last_error

def call(data):
    return rpc("eth_call", [{"to": CONTRACT, "data": data}, "latest"])

def challenge():   return call(SEL_CHALLENGE)            # 0x + 64 hex
def difficulty():  return int(call(SEL_DIFFICULTY), 16)
def mint_price():  return int(call(SEL_PRICE), 16)

# ---- OpenCL 挖矿 (NVIDIA) ----
devs = [d for p in cl.get_platforms() if "NVIDIA" in p.name for d in p.get_devices()]
if not devs: devs = [d for p in cl.get_platforms() for d in p.get_devices()]
if not devs: sys.exit("No OpenCL GPU devices found")
print(f"GPUs: {len(devs)}  wallet: {ADDR}", flush=True)

KERNEL = r"""
__constant uint K[64]={0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,
0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,
0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,0x748f82eeu,0x78a5636fu,
0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
#define ROR(x,n) rotate((uint)(x),(uint)(32-(n)))
#define S0(x) (ROR(x,2)^ROR(x,13)^ROR(x,22))
#define S1(x) (ROR(x,6)^ROR(x,11)^ROR(x,25))
#define s0(x) (ROR(x,7)^ROR(x,18)^((x)>>3))
#define s1(x) (ROR(x,17)^ROR(x,19)^((x)>>10))
void blk(uint* h,const uint* in){uint w[64];for(int i=0;i<16;i++)w[i]=in[i];
 for(int i=16;i<64;i++)w[i]=s1(w[i-2])+w[i-7]+s0(w[i-15])+w[i-16];
 uint a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
 for(int i=0;i<64;i++){uint t1=hh+S1(e)+((e&f)^(~e&g))+K[i]+w[i];uint t2=S0(a)+((a&b)^(a&c)^(b&c));hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
 h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;}
__kernel void mine(ulong base,__global volatile int* found,__global ulong* out,__global volatile uint* best){
 ulong nb=base+(ulong)get_global_id(0)*ITERS;
 for(uint it=0;it<ITERS;it++){ if(*found)return; ulong nonce=nb+it;
  uint b1[16]={A0,A1,A2,A3,A4,0,0,0,0,0,0,(uint)(nonce>>32),(uint)(nonce&0xffffffffu),C0,C1,C2};
  uint b2[16]={C3,C4,C5,C6,C7,0x80000000u,0,0,0,0,0,0,0,0,0,672u};
  uint h[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
  blk(h,b1);blk(h,b2);
  int lz=0;for(int i=0;i<8;i++){if(h[i]==0u)lz+=32;else{lz+=clz(h[i]);break;}}
  atomic_max(best,(uint)lz);
  if(lz>=DIFF){if(atomic_cmpxchg(found,0,1)==0)*out=nonce;return;}}}
"""

def mine(ch_hex, diff, check_stale):
    """挖当前 challenge。返回 nonce(int) 或 None(challenge 变了/需重挖)。"""
    addr_w = struct.unpack(">5I", bytes.fromhex(ADDR[2:]))
    ch_w   = struct.unpack(">8I", bytes.fromhex(ch_hex[2:]))
    ITERS = 4096
    defs = dict(ITERS=ITERS, DIFF=diff, A0=addr_w[0],A1=addr_w[1],A2=addr_w[2],A3=addr_w[3],A4=addr_w[4],
                C0=ch_w[0],C1=ch_w[1],C2=ch_w[2],C3=ch_w[3],C4=ch_w[4],C5=ch_w[5],C6=ch_w[6],C7=ch_w[7])
    src = KERNEL
    for k,v in defs.items(): src = src.replace(k, f"{v}u" if k in ("A0","A1","A2","A3","A4","C0","C1","C2","C3","C4","C5","C6","C7") else str(v))
    def worker(dev, index, stop, result, stats):
        ctx = cl.Context([dev]); q = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, src).build(); kernel = cl.Kernel(prg, "mine")
        mf = cl.mem_flags; found = np.zeros(1, np.int32); out = np.zeros(1, np.uint64); best = np.zeros(1, np.uint32)
        fg = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=found); og = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=out); bg = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=best)
        GLOBAL = 1<<20; per = GLOBAL*ITERS; base = (int.from_bytes(os.urandom(6), "big") + index) & ((1<<64)-1)
        while not stop.is_set():
            kernel(q, (GLOBAL,), None, np.uint64(base), fg, og, bg); q.finish()
            cl.enqueue_copy(q, found, fg); q.finish(); stats[index] = stats.get(index, 0) + per
            cl.enqueue_copy(q, best, bg); q.finish(); stats[(index, "best")] = max(stats.get((index, "best"), 0), int(best[0]))
            if found[0]:
                cl.enqueue_copy(q, out, og); q.finish(); result.put(int(out[0])); stop.set(); return
            base = (base + per * len(devs)) & ((1<<64)-1)

    stop = threading.Event(); result = queue.Queue(); stats = {}
    threads = [threading.Thread(target=worker, args=(dev, i, stop, result, stats), daemon=True) for i, dev in enumerate(devs)]
    for thread in threads: thread.start()
    started = last_time = time.time(); last_total = 0
    try:
        while any(thread.is_alive() for thread in threads):
            time.sleep(5); total = sum(stats.values()); now = time.time()
            speed = (total-last_total) / max(now-last_time, 0.1); last_total, last_time = total, now
            expected = float("inf") if speed <= 0 else (2**diff) / speed
            chance = (1 - (1 - 2**-diff) ** (max(speed, 0)*60)) * 100
            if expected < 60: wait = f"{expected:.1f}s"
            elif expected < 3600: wait = f"{expected/60:.1f}m"
            elif expected < 86400: wait = f"{expected/3600:.1f}h"
            else: wait = f"{expected/86400:.1f}d"
            best_bits = max((value for key, value in stats.items() if isinstance(key, tuple) and key[1] == "best"), default=0)
            print(f"  GPUs {len(devs)}  {speed/1e9:.2f} GH/s  hashes {total}  best {best_bits}/{diff} bits  expected {wait}  chance/min {chance:.2f}%", flush=True)
            try:
                if check_stale(): stop.set(); break
            except Exception as error:
                print(f"  RPC check failed ({error}); continuing current challenge", flush=True)
        try: return result.get_nowait()
        except queue.Empty: return None
    finally:
        stop.set()
        for thread in threads: thread.join(timeout=2)

def submit(nonce, ch_hex, price_wei):
    data = SEL_MINE + f"{nonce:064x}" + ch_hex[2:]
    latest = rpc("eth_getBlockByNumber", ["latest", False])
    base_fee = int(latest.get("baseFeePerGas", "0x0"), 16)
    priority_fee = int(os.environ.get("HASHBROKER_PRIORITY_FEE_WEI", "2000000"))
    max_fee = base_fee * 2 + priority_fee
    tx = {"nonce": int(rpc("eth_getTransactionCount", [ADDR, "pending"]), 16),
          "to": CONTRACT, "value": price_wei, "gas": 250000,
          "maxFeePerGas": max_fee, "maxPriorityFeePerGas": priority_fee,
          "data": data, "chainId": CHAIN_ID, "type": 2}
    print(f"  signing paid transaction value={price_wei} maxFeePerGas={max_fee} priorityFee={priority_fee}", flush=True)
    signed = acct.sign_transaction(tx)
    h = rpc("eth_sendRawTransaction", ["0x"+signed.raw_transaction.hex()])
    for _ in range(40):
        r = rpc("eth_getTransactionReceipt", [h])
        if r: return r.get("status") == "0x1", h
        time.sleep(0.4)
    return False, h

got = 0
try:
    while got < COUNT:
        price = mint_price()
        if MAX_PRICE_WEI and price > MAX_PRICE_WEI:
            print(f"⛔ mintPrice {price} wei exceeds HASHBROKER_MAX_PRICE_WEI {MAX_PRICE_WEI}; stopping."); break
        ch = challenge(); diff = difficulty()
        print(f"[{time.strftime('%H:%M:%S')}] mining challenge {ch[:12]}... difficulty {diff} priceWei {price}", flush=True)
        stale = lambda: challenge().lower() != ch.lower()
        nonce = mine(ch, diff, stale)
        if nonce is None:
            print("  challenge 变了 → 重挖"); continue
        if challenge().lower() != ch.lower():
            print("  解出但 challenge 已过期 → 重挖"); continue
        if challenge().lower() != ch.lower():
            print("  proof challenge expired -> remine"); continue
        ok, txh = submit(nonce, ch, price)
        if ok: got += 1; print(f"  ✓ 挖到并 mint 成功! nonce {nonce}  tx {txh}")
        else:  print(f"  ✗ 提交 revert(被抢先/过期)→ 重挖  {txh[:12]}")
    print(f"=== finished · minted {got} ===")
except KeyboardInterrupt:
    print(f"\n已停止 · 共免费挖到 {got} 个")
