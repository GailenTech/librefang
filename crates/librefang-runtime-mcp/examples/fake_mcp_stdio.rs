//! The fake MCP server of `tests/support/fake_mcp.rs` as a stdio child process, for the concurrency tests that need a real subprocess (a server that dies with calls in flight, out-of-order answers on one pipe).
//! Test-only: it lives under `examples/` so `cargo test` builds it next to the test binaries.

#[path = "../tests/support/fake_mcp.rs"]
mod fake_mcp;

use rmcp::ServiceExt;

#[tokio::main]
async fn main() {
    let ctl = fake_mcp::Control::new();
    ctl.set_allow_exit();
    let running = fake_mcp::FakeServer::new(ctl)
        .serve(rmcp::transport::stdio())
        .await
        .expect("serve on stdio");
    let _ = running.waiting().await;
}
