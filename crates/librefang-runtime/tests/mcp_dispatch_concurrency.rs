//! Adversarial tests for the MCP dispatch path (`execute_tool_raw`) and for `tool_wait`, against a real `rmcp` server.
//!
//! The entry-level behaviour (permits, close, correlation) is covered in `librefang-runtime-mcp/tests/mcp_concurrency.rs`; here the subject is what the dispatcher and the timeout helper do with it.

#[path = "../../librefang-runtime-mcp/tests/support/fake_mcp.rs"]
mod fake_mcp;

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use fake_mcp::{serve_http, Control};
use librefang_runtime::mcp::{
    empty_taint_rule_sets_handle, McpConnection, McpConnectionEntry, McpConnectionList,
    McpServerConfig, McpTransport,
};
use librefang_runtime::tool_runner::{execute_tool_raw, ToolExecContext};
use librefang_runtime::tool_wait::{begin_wait, timeout_excluding_waits};
use librefang_types::tool::ToolResult;
use serde_json::json;

const BOUND: Duration = Duration::from_secs(30);

async fn within<T>(what: &str, fut: impl Future<Output = T>) -> T {
    match tokio::time::timeout(BOUND, fut).await {
        Ok(v) => v,
        Err(_) => panic!("timed out after {BOUND:?} (deadlock or lost wake-up?): {what}"),
    }
}

async fn entry_for(
    name: &str,
    url: &str,
    limit: usize,
    timeout_secs: u64,
) -> Arc<McpConnectionEntry> {
    let conn = within(
        "connect",
        McpConnection::connect(McpServerConfig {
            name: name.to_string(),
            transport: McpTransport::Http {
                url: url.to_string(),
            },
            timeout_secs,
            env: vec![],
            headers: vec![],
            oauth_provider: None,
            oauth_config: None,
            taint_scanning: true,
            taint_policy: None,
            taint_rule_sets: empty_taint_rule_sets_handle(),
            roots: vec![],
        }),
    )
    .await
    .expect("connect to the fake server");
    McpConnectionEntry::new(conn, limit)
}

fn ctx<'a>(mcp_connections: &'a McpConnectionList) -> ToolExecContext<'a> {
    ToolExecContext {
        kernel: None,
        allowed_tools: None,
        available_tools: None,
        caller_agent_id: Some("test-agent"),
        skill_registry: None,
        allowed_skills: None,
        mcp_connections: Some(mcp_connections),
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

async fn dispatch(list: &McpConnectionList, tool: &str, id: &str) -> ToolResult {
    execute_tool_raw(id, tool, &json!({ "id": id }), &ctx(list)).await
}

// ------------------------------------------------------------------------------------------- dispatcher

/// Risk: a call queued behind a busy slot waits forever, or fails with a message that says the tool failed when it was never sent.
/// The slot is held directly by the test (not by a running call, which would end on its own `timeout_secs` at about the same moment the queued call gives up): with the server's `timeout_secs` at 1 the queued call must give up after about a second, say it was not sent, and never reach the server. Once the slot is free the same call works.
#[tokio::test]
async fn a_call_that_cannot_get_a_slot_gives_up_and_was_never_sent() {
    let ctl = Control::new();
    let fake = serve_http(Arc::clone(&ctl), false).await;
    let entry = entry_for("srv", &fake.url, 1, 1).await;
    let list: Arc<McpConnectionList> = Arc::new(tokio::sync::Mutex::new(vec![Arc::clone(&entry)]));

    let held = entry.acquire().await.expect("the only slot");
    let started = std::time::Instant::now();
    let queued = within("queued call", dispatch(&list, "mcp_srv_echo", "b")).await;
    let waited = started.elapsed();

    assert!(
        queued.is_error,
        "queued call should have failed: {}",
        queued.content
    );
    assert!(
        queued.content.contains("not sent"),
        "the error must say the call was never sent: {}",
        queued.content
    );
    assert!(
        waited >= Duration::from_millis(900),
        "gave up too early: {waited:?}"
    );
    assert!(
        waited < Duration::from_secs(10),
        "gave up too late: {waited:?}"
    );
    assert_eq!(ctl.arrived(), 0, "the abandoned call reached the server");

    drop(held);
    let ok = within(
        "call once the slot is free",
        dispatch(&list, "mcp_srv_echo", "c"),
    )
    .await;
    assert!(!ok.is_error, "{}", ok.content);
    assert_eq!(ok.content, "echo:c");
    fake.stop().await;
}

/// Risk: reconnect (dial-first transports) swaps a new entry in while a call is queued on the old one. The old entry is then closed, which fails every caller still queued on it, although the server is back and the replacement is in the list.
/// A call queued across a reconnect should be served by the replacement.
#[tokio::test]
async fn a_call_queued_on_an_entry_that_gets_replaced_is_served_by_the_replacement() {
    let fake = serve_http(Control::new(), false).await;
    let old = entry_for("srv", &fake.url, 1, 20).await;
    let new = entry_for("srv", &fake.url, 1, 20).await;
    let list: Arc<McpConnectionList> = Arc::new(tokio::sync::Mutex::new(vec![Arc::clone(&old)]));

    let held = old.acquire().await.expect("the only slot of the old entry");
    let queued = {
        let list = Arc::clone(&list);
        tokio::spawn(async move { dispatch(&list, "mcp_srv_echo", "q").await })
    };
    // Current-thread runtime: yielding lets the spawned call run up to its first suspension point, which is the wait for the busy slot.
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert!(
        !queued.is_finished(),
        "the call did not queue behind the busy slot"
    );

    // What `replace_mcp_connection` plus the background close do.
    {
        let mut g = list.lock().await;
        g.clear();
        g.push(Arc::clone(&new));
        old.retire();
    }
    let closer = {
        let old = Arc::clone(&old);
        tokio::spawn(async move { old.close().await })
    };
    drop(held);

    let r = within("the queued call", queued).await.unwrap();
    assert!(
        !r.is_error,
        "a call queued across a reconnect failed although the replacement was ready: {}",
        r.content
    );
    assert_eq!(r.content, "echo:q");
    within("old entry closes", closer).await.unwrap();
    fake.stop().await;
}

/// Risk: a tool name with the MCP prefix but no connected server hangs or panics instead of failing cleanly.
#[tokio::test]
async fn a_tool_of_an_unknown_server_fails_without_hanging() {
    let fake = serve_http(Control::new(), false).await;
    let list: McpConnectionList =
        tokio::sync::Mutex::new(vec![entry_for("srv", &fake.url, 1, 5).await]);
    for tool in ["mcp_nosuch_echo", "mcp_", "mcp_srv"] {
        let r = within(tool, dispatch(&list, tool, "x")).await;
        assert!(r.is_error, "{tool} should have failed, got: {}", r.content);
    }
    fake.stop().await;
}

// ------------------------------------------------------------------------------------------- tool_wait

/// Risk: `tool_timeout_secs` set absurdly high ("effectively no timeout") makes `Instant + Duration` overflow and panic in every tool call, where `tokio::time::timeout` would simply wait.
#[tokio::test]
async fn an_enormous_tool_timeout_does_not_panic() {
    let r = timeout_excluding_waits(Duration::from_secs(u64::MAX), async { 7 }).await;
    assert_eq!(r.ok(), Some(7));
}

/// Risk: the helper mixes the wall clock with the tokio clock. Under `start_paused` the tokio timer fires at once while the wall clock has not moved, and the loop spins until real time catches up.
#[tokio::test(start_paused = true)]
async fn the_timeout_follows_the_tokio_clock() {
    let real = std::time::Instant::now();
    let r = timeout_excluding_waits(
        Duration::from_secs(5),
        tokio::time::sleep(Duration::from_secs(10)),
    )
    .await;
    assert!(r.is_err(), "10 s of work under a 5 s budget must time out");
    assert!(
        real.elapsed() < Duration::from_secs(2),
        "spent {:?} of real time on 5 s of virtual time",
        real.elapsed()
    );
}

/// Same, with a wait excluded from the budget: 8 s waiting plus 3 s working fits in a 5 s budget.
#[tokio::test(start_paused = true)]
async fn waits_are_excluded_under_the_tokio_clock() {
    let real = std::time::Instant::now();
    let r = timeout_excluding_waits(Duration::from_secs(5), async {
        let guard = begin_wait();
        tokio::time::sleep(Duration::from_secs(8)).await;
        drop(guard);
        tokio::time::sleep(Duration::from_secs(3)).await;
    })
    .await;
    assert!(
        r.is_ok(),
        "the 8 s spent waiting were charged to the budget"
    );
    assert!(
        real.elapsed() < Duration::from_secs(2),
        "spent {:?} of real time on virtual time",
        real.elapsed()
    );
}

/// Risk: a wait that is abandoned half way (the future is dropped while the guard is alive) leaves the accounting in a state that stretches later deadlines.
/// Each scope has its own accounting, so a second scope must not inherit anything from the abandoned one.
#[tokio::test(start_paused = true)]
async fn an_abandoned_wait_does_not_stretch_a_later_scope() {
    let first = timeout_excluding_waits(Duration::from_secs(1), async {
        let _guard = begin_wait();
        std::future::pending::<()>().await;
    });
    // The wait is open and never ends; the outer timeout drops the whole thing.
    assert!(tokio::time::timeout(Duration::from_secs(30), first)
        .await
        .is_err());

    let r = timeout_excluding_waits(
        Duration::from_secs(2),
        tokio::time::sleep(Duration::from_secs(10)),
    )
    .await;
    assert!(
        r.is_err(),
        "a later scope was not bounded by its own budget"
    );
}
