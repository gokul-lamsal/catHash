import { createHash } from "node:crypto";
import { cpus } from "node:os";
import { execFileSync, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { ethers } from "ethers";

const RPC = process.env.HASHBROKER_RPC ?? "https://rpc.mainnet.chain.robinhood.com";
const CONTRACT = "0x4272D6f51771839F596082eF48fa84D35239Bab3";
const ABI = [
  "function totalSupply() view returns (uint256)",
  "function mintPrice() view returns (uint256)",
  "function currentDifficulty() view returns (uint256)",
  "function challenge() view returns (bytes32)",
  "function mine(uint256 nonce, bytes32 proofChallenge) payable returns (uint256)"
];

function stamp() { return new Date().toISOString().replace("T", " ").slice(0, 19); }
function log(level, message, fields = {}) {
  const suffix = Object.entries(fields).map(([k, v]) => `${k}=${v}`).join(" ");
  console.log(`[${stamp()}] ${level.padEnd(5)} ${message}${suffix ? ` | ${suffix}` : ""}`);
}
function rate(n) {
  const units = [[1e9, "GH/s"], [1e6, "MH/s"], [1e3, "kH/s"], [1, "H/s"]];
  const [factor, unit] = units.find(([f]) => n >= f) ?? units.at(-1);
  return `${(n / factor).toFixed(2)} ${unit}`;
}
function duration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86400).toFixed(1)}d`;
}
function gpuInfo() {
  try {
    const text = execFileSync("nvidia-smi", ["--query-gpu=index,name", "--format=csv,noheader,nounits"], { encoding: "utf8", timeout: 5000 });
    return text.trim().split(/\r?\n/).filter(Boolean).map((line) => {
      const [index, ...name] = line.split(",");
      return { index: Number(index.trim()), name: name.join(",").trim() };
    }).filter((gpu) => Number.isInteger(gpu.index));
  } catch { return []; }
}

const cudaBinary = process.env.HASHBROKER_CUDA_BIN ?? fileURLToPath(new URL("./cuda/hashbroker_cuda", import.meta.url));
function cudaWords(address, challenge) {
  return [...address.slice(2).matchAll(/.{8}/g), ...challenge.slice(2).matchAll(/.{8}/g)]
    .map((match) => match[0]).join(" ") + "\n";
}

function makeMessage(address, nonce, challenge) {
  // Matches Hashbroker's miner.js: address(20) + 24 zero bytes + nonce(uint64 BE) + challenge(32).
  const message = Buffer.alloc(84);
  Buffer.from(address.slice(2), "hex").copy(message, 0);
  message.writeBigUInt64BE(BigInt(nonce), 44);
  Buffer.from(challenge.slice(2), "hex").copy(message, 52);
  return message;
}
function hashProof(address, nonce, challenge) {
  return createHash("sha256").update(makeMessage(address, nonce, challenge)).digest();
}
function leadingBits(hash) {
  let bits = 0;
  for (const byte of hash) {
    if (byte === 0) { bits += 8; continue; }
    bits += Math.clz32(byte) - 24;
    break;
  }
  return bits;
}

if (!isMainThread) {
  const { address, challenge, difficulty, start, stride } = workerData;
  let nonce = BigInt(start);
  let hashes = 0;
  let bestBits = 0;
  let bestHash = null;
  const target = BigInt(1) << BigInt(256 - difficulty);
  while (true) {
    const hash = hashProof(address, nonce, challenge);
    const bits = leadingBits(hash);
    hashes++;
    if (bits > bestBits) { bestBits = bits; bestHash = hash.toString("hex"); }
    if (BigInt(`0x${hash.toString("hex")}`) < target) {
      parentPort.postMessage({ type: "found", nonce: nonce.toString(), hash: hash.toString("hex"), hashes });
      break;
    }
    nonce += BigInt(stride);
    if (hashes % 100000 === 0) {
      parentPort.postMessage({ type: "progress", hashes, bestBits, bestHash });
      hashes = 0;
    }
  }
  process.exit(0);
}

const args = process.argv.slice(2);
const argValue = (name) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : undefined; };
const workersRequested = Number(argValue("--workers") ?? cpus().length);
const key = process.env.HASHBROKER_PRIVATE_KEY;
if (!key) { log("ERROR", "HASHBROKER_PRIVATE_KEY is not set"); process.exit(2); }

const wallet = new ethers.Wallet(key);
const provider = new ethers.JsonRpcProvider(RPC, { chainId: 4663, name: "Robinhood Chain" });
const signer = wallet.connect(provider);
const contract = new ethers.Contract(CONTRACT, ABI, signer);
const gpus = gpuInfo();
log("INFO", "starting Hashbroker CLI", { address: wallet.address, workers: Math.max(1, workersRequested), rpc: RPC });
if (gpus.length && existsSync(cudaBinary)) log("INFO", "CUDA backend ready", { gpus: gpus.map((gpu) => `${gpu.index}:${gpu.name}`).join(", ") });
else if (gpus.length) log("WARN", "NVIDIA GPU detected but CUDA binary is missing; using CPU", { binary: cudaBinary });
else log("INFO", "no NVIDIA GPU detected; using CPU workers");

let stopped = false;
let activeWorkers = [];
function stopWorkers() { for (const worker of activeWorkers) worker.terminate(); activeWorkers = []; }
let activeCuda = [];
function stopCuda() { for (const child of activeCuda) child.kill("SIGTERM"); activeCuda = []; }
function stopMining() { stopWorkers(); stopCuda(); }
process.on("SIGINT", () => { stopped = true; stopMining(); log("INFO", "stopped by user"); process.exit(0); });

async function mineRound() {
  const [supply, price, difficulty, challenge] = await Promise.all([
    contract.totalSupply(), contract.mintPrice(), contract.currentDifficulty(), contract.challenge()
  ]);
  const diff = Number(difficulty);
  const challengeHex = String(challenge);
  const workerCount = Math.max(1, Math.min(workersRequested, 256));
  log("INFO", "new challenge", { supply: supply.toString(), difficulty: `${diff} bits`, priceWei: price.toString(), challenge: challengeHex });
  const useCuda = gpus.length > 0 && existsSync(cudaBinary);
  log("INFO", "mining", { mode: useCuda ? "CUDA" : "CPU", devices: useCuda ? gpus.length : Math.max(1, Math.min(workersRequested, 256)), expected: "calculating", chancePerMinute: "calculating" });

  if (useCuda) {
    try { return await mineCudaRound({ diff, challengeHex, price }); }
    catch (error) { log("WARN", "CUDA mining failed; falling back to CPU", { error: error.message }); }
  }

  return await new Promise((resolve, reject) => {
    let total = 0; let best = 0; let bestHash = null; let last = Date.now();
    let resolved = false;
    const finish = (value, error) => { if (resolved) return; resolved = true; stopWorkers(); error ? reject(error) : resolve(value); };
    for (let i = 0; i < workerCount; i++) {
      const worker = new Worker(new URL(import.meta.url), { workerData: { address: wallet.address, challenge: challengeHex, difficulty: diff, start: i, stride: workerCount } });
      activeWorkers.push(worker);
      worker.on("message", (msg) => {
        if (msg.type === "progress") {
          total += msg.hashes; best = Math.max(best, msg.bestBits); bestHash = msg.bestHash ?? bestHash;
          const now = Date.now(); const seconds = Math.max((now - last) / 1000, 0.1);
          const speed = msg.hashes / seconds; const probability = 2 ** -diff;
          const chance = (1 - Math.exp(-speed * 60 * probability)) * 100;
          log("INFO", "mining", { speed: rate(speed), hashes: total, best: `${best}/${diff} bits`, hash: bestHash ? `0x${bestHash.slice(0, 16)}...` : "-", expected: duration(1 / (speed * probability)), chancePerMinute: chance < 0.01 ? "<0.01%" : `${chance.toFixed(2)}%` });
          last = now;
        } else if (msg.type === "found") finish({ ...msg, challenge: challengeHex, price });
      });
      worker.on("error", (error) => finish(null, error));
    }
  });
}

function mineCudaRound({ diff, challengeHex, price }) {
  return new Promise((resolve, reject) => {
    const children = []; let total = 0n; let lastTotal = 0n; let best = 0; let last = Date.now(); let settled = false;
    const finish = (value, error) => { if (settled) return; settled = true; stopCuda(); error ? reject(error) : resolve(value); };
    for (let i = 0; i < gpus.length; i++) {
      const gpu = gpus[i]; const child = spawn(cudaBinary, [String(gpu.index), String(diff), String(i), String(gpus.length)], { stdio: ["pipe", "pipe", "pipe"] });
      children.push(child); activeCuda.push(child); let output = "";
      child.stdin.end(cudaWords(wallet.address, challengeHex));
      child.stdout.setEncoding("utf8"); child.stdout.on("data", (chunk) => {
        output += chunk; const lines = output.split(/\r?\n/); output = lines.pop() ?? "";
        for (const line of lines) {
          const [kind, value, bestValue] = line.trim().split(/\s+/);
          if (kind === "PROGRESS") {
            let count; try { count = BigInt(value); } catch { continue; }
            total += count;
            const reportedBest = Number(bestValue); if (Number.isInteger(reportedBest)) best = Math.max(best, reportedBest);
            const now = Date.now(); const seconds = Math.max((now - last) / 1000, 0.1); const delta = total - lastTotal; const speed = Number(delta) / seconds; lastTotal = total; last = now;
            const probability = 2 ** -diff; const chance = (1 - Math.exp(-Math.max(speed, 1) * 60 * probability)) * 100;
            log("INFO", "mining", { mode: "CUDA", gpu: `${gpu.index}:${gpu.name}`, speed: rate(Math.max(speed, 0)), hashes: total.toString(), best: `${best}/${diff} bits`, expected: duration(1 / (Math.max(speed, 1) * probability)), chancePerMinute: chance < 0.01 ? "<0.01%" : `${chance.toFixed(2)}%` });
          } else if (kind === "FOUND") {
            const nonce = value; const hash = hashProof(wallet.address, nonce, challengeHex).toString("hex"); finish({ nonce, hash, challenge: challengeHex, price });
          }
        }
      });
      child.stderr.setEncoding("utf8"); child.stderr.on("data", (data) => { if (data.trim()) log("ERROR", "CUDA device error", { gpu: gpu.index, error: data.trim() }); });
      child.on("error", (error) => finish(null, error));
      child.on("close", (code, signal) => { if (!settled && code !== 0) finish(null, new Error(`CUDA worker ${gpu.index} exited with code ${code}${signal ? ` signal=${signal}` : ""}`)); });
    }
  });
}

async function run() {
  while (!stopped) {
    try {
      const proof = await mineRound();
      log("INFO", "valid proof found", { nonce: proof.nonce, hash: `0x${proof.hash}`, challenge: proof.challenge });
      const fresh = await contract.challenge();
      if (String(fresh).toLowerCase() !== proof.challenge.toLowerCase()) { log("WARN", "discarding stale proof"); continue; }
      log("INFO", "signing transaction", { to: CONTRACT, valueWei: proof.price.toString(), nonce: proof.nonce });
      const tx = await contract.mine(proof.nonce, proof.challenge, { value: proof.price, gasLimit: 400000 });
      log("INFO", "transaction broadcast", { hash: tx.hash });
      const receipt = await tx.wait();
      if (receipt.status !== 1) throw new Error("transaction reverted");
      log("INFO", "transaction confirmed", { hash: tx.hash, block: receipt.blockNumber });
    } catch (error) {
      if (stopped) break;
      log("ERROR", "round failed; refreshing", { error: error.shortMessage ?? error.message });
      await new Promise((r) => setTimeout(r, 3000));
    }
  }
}
run().catch((error) => { log("ERROR", "fatal", { error: error.message }); process.exit(1); });
