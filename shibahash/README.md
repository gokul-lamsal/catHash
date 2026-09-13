# ShibaHash native miner

Rust CLI plus CUDA Keccak-256 worker. ShibaHash is not SHA-256: it hashes
`address || uint256(nonce) || prevWork || anchor` and submits
`mine(nonce, anchorBlock)` with the live `mintPrice()`.

Build the CUDA worker with `make -C cuda`, then build the Rust CLI with
`cargo build --release`. The Makefile emits `sm_89` and `sm_120` by default;
override with `NVCC_ARCH=89`, for example, when using a specific toolkit.

Set `SHIBAHASH_PRIVATE_KEY`; the CLI detects all GPUs reported by `nvidia-smi`,
partitions the nonce space across them, reads the live target and paid mint
price, signs the transaction locally, and submits it. Optional overrides are
`SHIBAHASH_RPC` and `SHIBAHASH_CUDA_BIN`.

On Linux:

```bash
export SHIBAHASH_PRIVATE_KEY=0x...
make -C cuda
cargo build --release
./target/release/shibahash-miner
```

Never reuse a private key that has been pasted into chat, logs, shell history,
or a public repository.
