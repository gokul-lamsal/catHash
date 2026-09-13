use anyhow::{Context, Result};
use ethers::{
    contract::Contract,
    middleware::SignerMiddleware,
    providers::{Http, Provider},
    signers::{LocalWallet, Signer},
    types::{Address, U256},
};
use std::{env, process::Stdio, sync::Arc};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
    sync::mpsc,
    task::JoinSet,
};

const RPC: &str = "https://rpc.mainnet.chain.robinhood.com";
const CONTRACT: &str = "0xF46A1d2eDDD1004B1345F3D0B5A2Db8e28939c67";
const ABI: &str = r#"[
 {"inputs":[],"name":"currentAnchor","outputs":[{"name":"anchorBlock","type":"uint256"},{"name":"anchor","type":"bytes32"}],"stateMutability":"view","type":"function"},
 {"inputs":[],"name":"prevWork","outputs":[{"name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},
 {"inputs":[{"name":"miner","type":"address"}],"name":"targetFor","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
 {"inputs":[{"name":"miner","type":"address"},{"name":"nonce","type":"uint256"},{"name":"prev","type":"bytes32"},{"name":"anchor","type":"bytes32"}],"name":"workHash","outputs":[{"name":"","type":"bytes32"}],"stateMutability":"pure","type":"function"},
 {"inputs":[],"name":"mintPrice","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
 {"inputs":[{"name":"nonce","type":"uint256"},{"name":"anchorBlock","type":"uint256"}],"name":"mine","outputs":[{"name":"tokenId","type":"uint256"}],"stateMutability":"payable","type":"function"}
]"#;

fn gpu_count() -> usize {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().count())
        .filter(|n| *n > 0)
        .unwrap_or(1)
}

fn u256_hex(value: U256) -> String {
    format!("{value:064x}")
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "—".into();
    }
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else if seconds < 3600.0 {
        format!("{:.1}m", seconds / 60.0)
    } else if seconds < 86_400.0 {
        format!("{:.1}h", seconds / 3600.0)
    } else {
        format!("{:.1}d", seconds / 86_400.0)
    }
}

enum WorkerEvent {
    Progress { gpu: usize, hashes: u64 },
    Found { gpu: usize, nonce: U256 },
    Error { gpu: usize, message: String },
}

async fn run_worker(
    gpu: usize,
    devices: usize,
    bin: String,
    encoded: String,
    target_hex: String,
    tx: mpsc::UnboundedSender<WorkerEvent>,
) {
    let result: Result<()> = async {
        let mut child = Command::new(bin)
            .args([
                gpu.to_string(),
                gpu.to_string(),
                devices.to_string(),
                target_hex,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("start CUDA worker {gpu}"))?;
        child
            .stdin
            .take()
            .context("CUDA worker stdin unavailable")?
            .write_all(encoded.as_bytes())
            .await?;
        let stdout = child
            .stdout
            .take()
            .context("CUDA worker stdout unavailable")?;
        let mut lines = BufReader::new(stdout).lines();
        while let Some(line) = lines.next_line().await? {
            if let Some(value) = line.strip_prefix("PROGRESS ") {
                if let Ok(hashes) = value.trim().parse::<u64>() {
                    let _ = tx.send(WorkerEvent::Progress { gpu, hashes });
                }
            } else if let Some(value) = line.strip_prefix("FOUND ") {
                let nonce = value.trim().parse::<U256>()?;
                let _ = tx.send(WorkerEvent::Found { gpu, nonce });
                return Ok(());
            }
        }
        let status = child.wait().await?;
        if !status.success() {
            let _ = tx.send(WorkerEvent::Error {
                gpu,
                message: format!("worker exited with {status}"),
            });
        }
        Ok(())
    }
    .await;
    if let Err(error) = result {
        let _ = tx.send(WorkerEvent::Error {
            gpu,
            message: error.to_string(),
        });
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let key = env::var("SHIBAHASH_PRIVATE_KEY").context("SHIBAHASH_PRIVATE_KEY is not set")?;
    let wallet: LocalWallet = key.parse::<LocalWallet>()?.with_chain_id(4663u64);
    let provider =
        Provider::<Http>::try_from(env::var("SHIBAHASH_RPC").unwrap_or_else(|_| RPC.into()))?;
    let client = Arc::new(SignerMiddleware::new(provider, wallet));
    let address: Address = client.address();
    let contract = Contract::new(
        CONTRACT.parse::<Address>()?,
        serde_json::from_str::<ethers::abi::Abi>(ABI)?,
        client.clone(),
    );
    let bin = env::var("SHIBAHASH_CUDA_BIN").unwrap_or_else(|_| "./cuda/shibahash_cuda".into());
    let devices = gpu_count();

    println!("ShibaHash CUDA miner | address={address:?} GPUs={devices}");

    loop {
        let (anchor_block, anchor): (U256, [u8; 32]) =
            contract.method("currentAnchor", ())?.call().await?;
        let prev: [u8; 32] = contract.method("prevWork", ())?.call().await?;
        let target: U256 = contract.method("targetFor", address)?.call().await?;
        let price: U256 = contract.method("mintPrice", ())?.call().await?;
        println!(
            "challenge anchor={anchor_block} target=0x{} priceWei={price}",
            u256_hex(target)
        );

        let mut prefix = [0u8; 84];
        prefix[..20].copy_from_slice(address.as_bytes());
        prefix[20..52].copy_from_slice(&prev);
        prefix[52..84].copy_from_slice(&anchor);
        let encoded = hex::encode(prefix);
        let target_hex = u256_hex(target);

        let started = std::time::Instant::now();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut workers = JoinSet::new();
        for gpu in 0..devices {
            let worker_tx = tx.clone();
            workers.spawn(run_worker(
                gpu,
                devices,
                bin.clone(),
                encoded.clone(),
                target_hex.clone(),
                worker_tx,
            ));
        }

        let mut found = None;
        let mut per_gpu = vec![0u64; devices];
        let mut last_report = std::time::Instant::now();
        while let Some(event) = rx.recv().await {
            match event {
                WorkerEvent::Progress { gpu, hashes } => {
                    per_gpu[gpu] = per_gpu[gpu].saturating_add(hashes);
                    let total: u64 = per_gpu.iter().sum();
                    let elapsed = started.elapsed().as_secs_f64().max(0.001);
                    if last_report.elapsed().as_secs_f64() < 1.0 {
                        continue;
                    }
                    last_report = std::time::Instant::now();
                    let speed = total as f64 / elapsed;
                    let target_ratio =
                        target.to_string().parse::<f64>().unwrap_or(0.0) / 2f64.powi(256);
                    let expected = if speed > 0.0 && target_ratio > 0.0 {
                        1.0 / (speed * target_ratio)
                    } else {
                        f64::INFINITY
                    };
                    let chance = 1.0 - (-speed * 60.0 * target_ratio).exp();
                    println!(
                        "mining GPUs={} speed={:.2} MH/s hashes={} expected={} chance/min={:.2}%",
                        devices,
                        speed / 1_000_000.0,
                        total,
                        format_duration(expected),
                        chance * 100.0
                    );
                }
                WorkerEvent::Found { gpu, nonce } => {
                    println!("CUDA GPU {gpu} found candidate nonce={nonce}");
                    found = Some(nonce);
                    workers.abort_all();
                    break;
                }
                WorkerEvent::Error { gpu, message } => {
                    eprintln!("CUDA worker {gpu} error: {message}");
                }
            }
        }
        while workers.join_next().await.is_some() {}

        let Some(nonce) = found else {
            println!("all CUDA workers stopped; refreshing challenge");
            continue;
        };
        let work: ethers::types::H256 = contract
            .method("workHash", (address, nonce, prev, anchor))?
            .call()
            .await?;
        let work_value = U256::from_big_endian(work.as_bytes());
        if work_value >= target {
            eprintln!(
                "discarding invalid CUDA result nonce={nonce} hash={work:?} target=0x{}",
                u256_hex(target)
            );
            continue;
        }
        let (fresh_anchor_block, fresh_anchor): (U256, [u8; 32]) =
            contract.method("currentAnchor", ())?.call().await?;
        let fresh_prev: [u8; 32] = contract.method("prevWork", ())?.call().await?;
        let fresh_target: U256 = contract.method("targetFor", address)?.call().await?;
        if fresh_anchor_block != anchor_block || fresh_anchor != anchor || fresh_prev != prev {
            println!("challenge changed before submit; discarding nonce and restarting");
            continue;
        }
        if work_value >= fresh_target {
            println!("target changed before submit; discarding nonce and restarting");
            continue;
        }
        println!("FOUND nonce={nonce}; submitting paid mint...");
        let call = contract
            .method::<_, U256>("mine", (nonce, anchor_block))?
            .value(price);
        let pending = call.send().await?;
        let tx_hash = pending.tx_hash();
        println!("submitted tx={tx_hash:?}");
        match pending.await? {
            Some(receipt) => println!(
                "confirmed block={} status={:?}",
                receipt.block_number.unwrap_or_default(),
                receipt.status
            ),
            None => println!("transaction dropped or not yet mined: {tx_hash:?}"),
        }
    }
}
