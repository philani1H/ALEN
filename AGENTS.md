# AGENTS.md

## Cursor Cloud specific instructions

ALEN is a single self-contained Rust binary (`alen`) that runs an Axum HTTP API on
`0.0.0.0:3000` **and** serves its web UI from `web/` at `/`. There is no separate
frontend/backend and no external services: SQLite is compiled in via `rusqlite`'s
`bundled` feature, and data is auto-created under `~/.local/share/alen` (override with
`ALEN_DATA_DIR`). See `README.md` for the API reference and config env vars.

### Toolchain (important, non-obvious)
- The base image's default rustup toolchain is pinned to an old version (1.83.0) that
  **cannot build this project** — a transitive dependency requires the `edition2024`
  Cargo feature (needs Rust/Cargo 1.85+). The update script runs `rustup default stable`
  to fix this. If you ever see `feature \`edition2024\` is required`, run
  `rustup default stable`.

### Build / run
- Build (dev): `cargo build`. Run the server: `cargo run` (listens on port 3000). Both
  are standard; the build emits many warnings (unused imports/vars) that are expected.
- The web UI is served at `http://localhost:3000/`; `/health`, `/train`, `/infer`,
  `/stats`, etc. are documented in `README.md`.

### Testing / lint caveats
- `cargo test` currently **fails to compile** due to pre-existing errors in the repo's
  test/example code (e.g. `NeuralChainOfThoughtReasoner::new` called with the wrong
  argument count, stale imports/struct fields in `tests/` and `examples/`). This is a
  source-code issue in the repo, not an environment problem — do not "fix" it as part of
  environment setup.
- `cargo fmt --check` reports pre-existing formatting diffs in the committed code.

### API gotcha
- `POST /train` expects `context` as an **array** of strings
  (e.g. `{"input":"...","expected_answer":"...","context":["a","b"]}`). The README
  example that passes `context` as a plain string is outdated and returns a
  deserialization error. `context` may also be omitted.
- The model is untrained by default, so `/infer` and the chat UI return low-quality /
  nonsensical text until trained — this is expected behavior, not a bug.
