//! Concurrency tests for the kernel's MCP connection list: reconnect, disconnect and calls racing each other, against a real `rmcp` server.
//!
//! The entry-level behaviour is covered in `librefang-runtime-mcp/tests/mcp_concurrency.rs` and the dispatcher in `librefang-runtime/tests/mcp_dispatch_concurrency.rs`; what only the kernel can show is the list, the tool cache, the health record and the connection-ops lock working together.
//! Waiting is done on the fake server's counters and on gates the test opens; the wall clock appears only as an upper bound.

#[path = "../../../librefang-runtime-mcp/tests/support/fake_mcp.rs"]
mod fake_mcp;

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fake_mcp::{serve_http, serve_http_at, Control};
use librefang_runtime::mcp::McpConnectionEntry;
use librefang_runtime::tool_runner::{execute_tool_raw, ToolExecContext};
use librefang_types::config::{KernelConfig, McpServerConfigEntry, McpTransportEntry};
use librefang_types::tool::ToolResult;
use serde_json::json;

use super::LibreFangKernel;

const BOUND: Duration = Duration::from_secs(30);

async fn within<T>(what: &str, fut: impl Future<Output = T>) -> T {
    match tokio::time::timeout(BOUND, fut).await {
        Ok(v) => v,
        Err(_) => panic!("timed out after {BOUND:?} (deadlock or lost wake-up?): {what}"),
    }
}

fn http_server(name: &str, url: &str, max_concurrent_calls: u32) -> McpServerConfigEntry {
    McpServerConfigEntry {
        name: name.to_string(),
        template_id: None,
        transport: Some(McpTransportEntry::Http {
            url: url.to_string(),
        }),
        timeout_secs: 20,
        env: Vec::new(),
        headers: Vec::new(),
        oauth: None,
        taint_scanning: true,
        taint_policy: None,
        max_concurrent_calls,
    }
}

async fn boot(servers: Vec<McpServerConfigEntry>) -> (Arc<LibreFangKernel>, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = KernelConfig {
        home_dir: tmp.path().to_path_buf(),
        data_dir: tmp.path().join("data"),
        ..KernelConfig::default()
    };
    config.mcp_servers = servers;
    let kernel = Arc::new(LibreFangKernel::boot_with_config(config).expect("boot"));
    kernel.connect_mcp_servers().await;
    (kernel, tmp)
}

async fn live(kernel: &LibreFangKernel, name: &str) -> Vec<Arc<McpConnectionEntry>> {
    kernel
        .mcp
        .mcp_connections
        .lock()
        .await
        .iter()
        .filter(|e| e.name() == name)
        .cloned()
        .collect()
}

fn cached_tools(kernel: &LibreFangKernel) -> usize {
    kernel.mcp.mcp_tools.lock().unwrap().len()
}

fn ctx<'a>(list: &'a librefang_runtime::mcp::McpConnectionList) -> ToolExecContext<'a> {
    ToolExecContext {
        kernel: None,
        allowed_tools: None,
        available_tools: None,
        caller_agent_id: Some("test-agent"),
        skill_registry: None,
        allowed_skills: None,
        mcp_connections: Some(list),
        web_ctx: None,
        browser_ctx: None,
        allowed_env_vars: None,
        workspace_root: None,
        media_engine: None,
        media_drivers: None,
        exec_policy: None,
        tts_engine: None,
        tts_config: None,
        docker_config: None,
        process_manager: None,
        process_registry: None,
        sender_id: None,
        channel: None,
        chat_id: None,
        sender_account_id: None,
        session_id: None,
        spill_threshold_bytes: 0,
        max_artifact_bytes: 0,
        checkpoint_manager: None,
        interrupt: None,
        dangerous_command_checker: None,
        acting_principal: None,
    }
}

/// A call through the same dispatcher an agent uses, against the kernel's own list.
async fn dispatch(kernel: &LibreFangKernel, tool: &str, id: &str) -> ToolResult {
    execute_tool_raw(
        id,
        tool,
        &json!({ "id": id }),
        &ctx(&kernel.mcp.mcp_connections),
    )
    .await
}

// ------------------------------------------------------------------------------------------- config

/// Risk: `max_concurrent_calls` from the config is lost on one of the paths that build an entry (boot, reconnect), or 0 reaches the semaphore.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_configured_limit_reaches_the_entry_at_boot_and_after_reconnect() {
    let fake = serve_http(Control::new(), false).await;
    let (kernel, _tmp) = boot(vec![
        http_server("three", &fake.url, 3),
        http_server("zero", &fake.url, 0),
    ])
    .await;
    assert_eq!(live(&kernel, "three").await[0].max_concurrent_calls(), 3);
    assert_eq!(live(&kernel, "zero").await[0].max_concurrent_calls(), 1);

    for name in ["three", "zero"] {
        within("reconnect", kernel.reconnect_mcp_server(name))
            .await
            .expect("reconnect");
    }
    assert_eq!(live(&kernel, "three").await[0].max_concurrent_calls(), 3);
    assert_eq!(live(&kernel, "zero").await[0].max_concurrent_calls(), 1);
    kernel.shutdown();
    fake.stop().await;
}

// ------------------------------------------------------------------------------------------- reconnect

/// Risk: during a reconnect of a dial-first transport, a reader or a caller finds the server missing, the tool cache emptied or doubled, or a call fails because it grabbed the entry that was just retired.
/// Readers sample the list and the cache under the list lock; four callers go through the real dispatcher against a server that allows one call at a time, so there is always a queue when a reconnect swaps the entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnecting_under_load_never_exposes_a_gap_or_fails_a_call() {
    let fake = serve_http(Control::new(), false).await;
    let (kernel, _tmp) = boot(vec![http_server("srv", &fake.url, 1)]).await;
    let tools_before = cached_tools(&kernel);
    assert!(tools_before > 0, "the fake server's tools were not cached");

    let stop = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut watchers = tokio::task::JoinSet::new();

    for _ in 0..2 {
        let (kernel, stop) = (Arc::clone(&kernel), Arc::clone(&stop));
        watchers.spawn(async move {
            let mut samples = 0u64;
            while !stop.load(Ordering::Relaxed) {
                {
                    let list = kernel.mcp.mcp_connections.lock().await;
                    let n = list.iter().filter(|e| e.name() == "srv").count();
                    assert_eq!(n, 1, "{n} entries named srv in the list");
                    assert_eq!(
                        cached_tools(&kernel),
                        tools_before,
                        "tool cache changed under the list lock"
                    );
                }
                samples += 1;
                tokio::task::yield_now().await;
            }
            samples
        });
    }
    for c in 0..4 {
        let (kernel, stop, completed) = (
            Arc::clone(&kernel),
            Arc::clone(&stop),
            Arc::clone(&completed),
        );
        watchers.spawn(async move {
            let mut calls = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let id = format!("{c}-{calls}");
                let r = dispatch(&kernel, "mcp_srv_echo", &id).await;
                assert!(
                    !r.is_error,
                    "call {id} failed during a reconnect: {}",
                    r.content
                );
                assert_eq!(r.content, format!("echo:{id}"));
                calls += 1;
                completed.fetch_add(1, Ordering::Relaxed);
            }
            calls
        });
    }

    for i in 0..12 {
        // Pace the reconnects by the calls themselves: with one slot and four callers there is always a queue, and eight completions mean every caller that was queued at the previous swap has been served.
        let target = completed.load(Ordering::Relaxed) + 8;
        within("the callers keep making progress", async {
            while completed.load(Ordering::Relaxed) < target {
                tokio::task::yield_now().await;
            }
        })
        .await;
        within(
            &format!("reconnect {i}"),
            kernel.reconnect_mcp_server("srv"),
        )
        .await
        .unwrap_or_else(|e| panic!("reconnect {i}: {e:?}"));
    }
    stop.store(true, Ordering::Relaxed);
    let mut total = 0;
    while let Some(done) = within("a watcher", watchers.join_next()).await {
        total += done.expect("a watcher panicked");
    }
    assert!(
        total > 12,
        "the watchers barely ran ({total}); the test proved little"
    );
    assert_eq!(live(&kernel, "srv").await.len(), 1);
    kernel.shutdown();
    fake.stop().await;
}

/// Risk: reconnect tears the old transport down under a call that is still running, or loses that call's answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_call_in_flight_finishes_on_the_old_connection_while_the_new_one_serves() {
    let ctl = Control::new();
    let fake = serve_http(Arc::clone(&ctl), false).await;
    let (kernel, _tmp) = boot(vec![http_server("srv", &fake.url, 2)]).await;
    let old = live(&kernel, "srv").await.remove(0);

    let in_flight = {
        let kernel = Arc::clone(&kernel);
        tokio::spawn(async move { dispatch(&kernel, "mcp_srv_gate", "g").await })
    };
    within("the call reaches the server", ctl.wait_arrived(1)).await;

    within(
        "reconnect while a call runs",
        kernel.reconnect_mcp_server("srv"),
    )
    .await
    .expect("reconnect");
    let now = live(&kernel, "srv").await;
    assert_eq!(now.len(), 1);
    assert!(!Arc::ptr_eq(&now[0], &old));
    assert!(old.is_retired());

    // The new connection serves while the old call is still running.
    let r = within(
        "call on the new connection",
        dispatch(&kernel, "mcp_srv_echo", "n"),
    )
    .await;
    assert_eq!(r.content, "echo:n");

    ctl.open("g");
    let r = within("the call that was in flight", in_flight)
        .await
        .unwrap();
    assert!(
        !r.is_error,
        "the call in flight was cut by the reconnect: {}",
        r.content
    );
    assert_eq!(r.content, "gate:g");
    kernel.shutdown();
    fake.stop().await;
}

/// Risk: stale reads. The entry copies `name` and `tools` at construction; after a reconnect to a server whose tool set changed, the cache and the generation counter must show the new set, and a tool that disappeared must stop being callable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reconnect_picks_up_a_changed_tool_set_and_bumps_the_generation() {
    let ctl = Control::new();
    let fake = serve_http(Arc::clone(&ctl), false).await;
    let (kernel, _tmp) = boot(vec![http_server("srv", &fake.url, 2)]).await;
    let before = cached_tools(&kernel);
    let generation = || kernel.mcp.mcp_generation.load(Ordering::Relaxed);
    let gen_before = generation();

    ctl.set_extra_tool(true);
    within(
        "reconnect with a new tool",
        kernel.reconnect_mcp_server("srv"),
    )
    .await
    .expect("reconnect");
    assert_eq!(cached_tools(&kernel), before + 1);
    assert!(
        generation() > gen_before,
        "the generation counter did not move"
    );
    let r = dispatch(&kernel, "mcp_srv_extra", "e").await;
    assert_eq!(r.content, "echo:e");

    let gen_mid = generation();
    ctl.set_extra_tool(false);
    within("reconnect without it", kernel.reconnect_mcp_server("srv"))
        .await
        .expect("reconnect");
    assert_eq!(cached_tools(&kernel), before);
    assert!(generation() > gen_mid);
    let r = dispatch(&kernel, "mcp_srv_extra", "e").await;
    assert!(
        r.is_error,
        "a tool the server no longer offers was still callable: {}",
        r.content
    );
    kernel.shutdown();
    fake.stop().await;
}

/// Risk: several reconnects of the same server at once (health loop plus an operator click) leave two live entries, a doubled tool cache, or deadlock on the connection-ops lock.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_reconnects_leave_exactly_one_live_entry() {
    let fake = serve_http(Control::new(), false).await;
    let (kernel, _tmp) = boot(vec![http_server("srv", &fake.url, 2)]).await;
    let tools_before = cached_tools(&kernel);
    let first = live(&kernel, "srv").await.remove(0);

    let reconnects: Vec<_> = (0..6)
        .map(|_| {
            let kernel = Arc::clone(&kernel);
            tokio::spawn(async move { kernel.reconnect_mcp_server("srv").await })
        })
        .collect();
    for r in reconnects {
        within("a reconnect", r).await.unwrap().expect("reconnect");
    }
    let now = live(&kernel, "srv").await;
    assert_eq!(now.len(), 1, "two live entries for one server");
    assert!(!now[0].is_retired());
    assert!(first.is_retired());
    assert_eq!(cached_tools(&kernel), tools_before);
    let r = dispatch(&kernel, "mcp_srv_echo", "x").await;
    assert_eq!(r.content, "echo:x");
    kernel.shutdown();
    fake.stop().await;
}

/// Risk: a failed dial. The old connection is taken out and closed (existing behaviour); the list and the tool cache must agree, nothing may hang, and once the server is back a reconnect must restore exactly one entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_reconnect_empties_the_list_consistently_and_a_later_one_recovers() {
    let fake = serve_http(Control::new(), false).await;
    let addr = fake.addr.to_string();
    let (kernel, _tmp) = boot(vec![http_server("srv", &fake.url, 2)]).await;
    assert_eq!(live(&kernel, "srv").await.len(), 1);

    fake.stop().await;
    let err = within(
        "reconnect to a dead server",
        kernel.reconnect_mcp_server("srv"),
    )
    .await
    .expect_err("nothing is listening");
    eprintln!("expected failure: {err:?}");
    assert!(
        live(&kernel, "srv").await.is_empty(),
        "a failed dial left the old entry in the list"
    );
    assert_eq!(
        cached_tools(&kernel),
        0,
        "tools of a removed server stayed in the cache"
    );
    let r = within(
        "call with no server",
        dispatch(&kernel, "mcp_srv_echo", "x"),
    )
    .await;
    assert!(r.is_error);

    let back = serve_http_at(Control::new(), false, &addr).await;
    within(
        "reconnect to the revived server",
        kernel.reconnect_mcp_server("srv"),
    )
    .await
    .expect("reconnect");
    assert_eq!(live(&kernel, "srv").await.len(), 1);
    assert!(cached_tools(&kernel) > 0);
    let r = dispatch(&kernel, "mcp_srv_echo", "y").await;
    assert_eq!(r.content, "echo:y");
    kernel.shutdown();
    back.stop().await;
}

/// Characterisation of the close-first path (stdio, or no transport configured): the server leaves the list for as long as reconnect waits for the calls in flight.
/// What must hold: the wait ends when the call ends (no hang), the call in flight is not cut, and callers that arrive during the gap fail at once instead of queueing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn close_first_reconnect_waits_for_the_call_in_flight_and_callers_in_the_gap_fail_fast() {
    let ctl = Control::new();
    let fake = serve_http(Arc::clone(&ctl), false).await;
    let (kernel, _tmp) = boot(vec![http_server("srv", &fake.url, 2)]).await;
    // No transport: reconnect takes the close-first path and then has nothing to dial.
    kernel
        .mcp
        .effective_mcp_servers
        .write()
        .unwrap()
        .iter_mut()
        .for_each(|s| s.transport = None);

    let in_flight = {
        let kernel = Arc::clone(&kernel);
        tokio::spawn(async move { dispatch(&kernel, "mcp_srv_gate", "g").await })
    };
    within("the call reaches the server", ctl.wait_arrived(1)).await;

    let reconnect = {
        let kernel = Arc::clone(&kernel);
        tokio::spawn(async move { kernel.reconnect_mcp_server("srv").await })
    };
    // The entry leaves the list as soon as reconnect takes it, before it starts waiting for the call.
    within("the server leaves the list", async {
        while !live(&kernel, "srv").await.is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(
        !reconnect.is_finished(),
        "reconnect did not wait for the call in flight"
    );

    let r = within(
        "a call in the gap",
        dispatch(&kernel, "mcp_srv_echo", "gap"),
    )
    .await;
    assert!(r.is_error, "a call in the gap was served: {}", r.content);

    ctl.open("g");
    let r = within("the call in flight", in_flight).await.unwrap();
    assert!(!r.is_error, "the call in flight was cut: {}", r.content);
    let outcome = within("reconnect finishes", reconnect).await.unwrap();
    assert!(
        outcome.is_err(),
        "reconnect without a transport cannot succeed"
    );
    kernel.shutdown();
    fake.stop().await;
}

// ------------------------------------------------------------------------------------------- disconnect

/// Risk: reconnects, disconnects and calls interleaved (health loop, operator, hot reload) deadlock on the connection-ops lock and the list lock, or leave the tool cache out of step with the list.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnects_disconnects_and_calls_interleaved_neither_deadlock_nor_desync_the_cache() {
    let fake = serve_http(Control::new(), false).await;
    let (kernel, _tmp) = boot(vec![http_server("srv", &fake.url, 2)]).await;
    let per_server = cached_tools(&kernel);

    let mut tasks = tokio::task::JoinSet::new();
    for n in 0..6u32 {
        let kernel = Arc::clone(&kernel);
        tasks.spawn(async move {
            for round in 0..4u32 {
                match (n + round) % 3 {
                    0 => {
                        let _ = kernel.reconnect_mcp_server("srv").await;
                    }
                    1 => {
                        let _ = kernel.disconnect_mcp_server("srv").await;
                    }
                    _ => {
                        let id = format!("{n}-{round}");
                        let r = dispatch(&kernel, "mcp_srv_echo", &id).await;
                        // Served, or refused because the server is momentarily gone: never anything else.
                        assert!(
                            r.content == format!("echo:{id}") || r.is_error,
                            "unexpected result {:?}",
                            r.content
                        );
                    }
                }
            }
        });
    }
    while let Some(done) = within("an interleaved task", tasks.join_next()).await {
        done.expect("an interleaved task panicked");
    }

    // Whatever order they ran in, the cache describes exactly the list.
    let entries = live(&kernel, "srv").await.len();
    assert!(entries <= 1, "{entries} entries for one server");
    assert_eq!(cached_tools(&kernel), entries * per_server);
    // And the server can be brought back.
    within("final reconnect", kernel.reconnect_mcp_server("srv"))
        .await
        .expect("reconnect");
    assert_eq!(live(&kernel, "srv").await.len(), 1);
    assert_eq!(cached_tools(&kernel), per_server);
    kernel.shutdown();
    fake.stop().await;
}

/// Risk: removing a server with calls queued behind a slow one. The queued calls must fail instead of waiting their turn, the call in flight must finish, disconnect must wait for it and then return, and the list and the cache must end empty.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disconnect_with_queued_calls_fails_them_and_waits_for_the_running_one() {
    let ctl = Control::new();
    let fake = serve_http(Arc::clone(&ctl), false).await;
    let (kernel, _tmp) = boot(vec![http_server("srv", &fake.url, 1)]).await;

    let running = {
        let kernel = Arc::clone(&kernel);
        tokio::spawn(async move { dispatch(&kernel, "mcp_srv_gate", "run").await })
    };
    within("the running call reaches the server", ctl.wait_arrived(1)).await;
    let queued: Vec<_> = (0..4)
        .map(|i| {
            let kernel = Arc::clone(&kernel);
            tokio::spawn(async move { dispatch(&kernel, "mcp_srv_echo", &format!("q{i}")).await })
        })
        .collect();
    let disconnect = {
        let kernel = Arc::clone(&kernel);
        tokio::spawn(async move { kernel.disconnect_mcp_server("srv").await })
    };

    for q in queued {
        let r = within("a queued call must fail promptly", q).await.unwrap();
        assert!(
            r.is_error,
            "a queued call was served by a server being removed: {}",
            r.content
        );
    }
    assert!(
        !disconnect.is_finished(),
        "disconnect returned while a call was still running"
    );

    ctl.open("run");
    let r = within("the running call", running).await.unwrap();
    assert!(!r.is_error, "the running call was cut: {}", r.content);
    assert!(within("disconnect", disconnect).await.unwrap());
    assert!(live(&kernel, "srv").await.is_empty());
    assert_eq!(cached_tools(&kernel), 0);
    kernel.shutdown();
    fake.stop().await;
}
