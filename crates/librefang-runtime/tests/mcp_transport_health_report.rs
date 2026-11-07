//! #7963 — the tool-call dispatch path reports MCP transport failures to the health monitor, and only transport failures.
//!
//! These tests exercise the real injection site: a live [`McpConnection`] against a stub MCP server, dispatched through `execute_tool_raw`, with a recording [`McpTransportHealthReporter`] standing in for the kernel's health monitor.
//!
//! Testing the wiring rather than only the implementation is deliberate.
//! `McpConnection::health_reporter` is an `Option` and the classification is a branch — both compile perfectly well while doing nothing, which is exactly how #7963 stayed invisible: every piece of the auto-reconnect machinery worked, and nothing connected them.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use librefang_runtime::mcp::{
    empty_taint_rule_sets_handle, McpConnection, McpConnectionEntry, McpConnectionList,
    McpServerConfig, McpTransport, McpTransportHealthReporter,
};
use librefang_runtime::tool_runner::{execute_tool_raw, ToolExecContext};
use serde_json::json;

// --------------------------------------------------------------------------- Recording reporter ---------------------------------------------------------------------------

#[derive(Default)]
struct RecordingReporter {
    transport_failures: Mutex<Vec<(String, String)>>,
    ok_calls: Mutex<Vec<(String, usize)>>,
}

impl RecordingReporter {
    fn transport_failures(&self) -> Vec<(String, String)> {
        self.transport_failures.lock().unwrap().clone()
    }

    fn ok_calls(&self) -> Vec<(String, usize)> {
        self.ok_calls.lock().unwrap().clone()
    }
}

impl McpTransportHealthReporter for RecordingReporter {
    fn report_transport_failure(&self, server: &str, error: &str) {
        self.transport_failures
            .lock()
            .unwrap()
            .push((server.to_string(), error.to_string()));
    }

    fn report_call_ok(&self, server: &str, tool_count: usize) {
        self.ok_calls
            .lock()
            .unwrap()
            .push((server.to_string(), tool_count));
    }
}

// --------------------------------------------------------------------------- Stub MCP server (SSE transport: one HTTP POST per JSON-RPC request) ---------------------------------------------------------------------------

/// What the stub does when it receives `tools/call`.
#[derive(Clone, Copy)]
enum CallBehaviour {
    /// Accept the request and never answer — the #7963 wedge: the server is alive, the transport carries nothing back, every call expires at `timeout_secs`.
    Wedge,
    /// Answer with a well-formed JSON-RPC error object, the way a server reports bad arguments or a missing file.
    JsonRpcError,
    /// Answer with a normal successful result.
    Success,
    /// Answer with a successful result after sleeping this many milliseconds. Each request is served on its own thread, so slow calls do not queue inside the stub.
    SlowSuccess(u64),
}

fn json_body(stream: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// Spawn a stub server that answers `initialize` / `tools/list` so the connection handshakes and registers one tool, then applies `behaviour` to `tools/call`.
///
/// The returned handle is *not* a shutdown handle — dropping a `JoinHandle` detaches the thread rather than stopping it, and the thread is parked in `listener.incoming()` (holding the wedged sockets open, which the wedge test needs). It is returned only so a caller can keep the stub alive for the duration of the test; the thread and its listener go away with the test binary.
fn spawn_stub(behaviour: CallBehaviour) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub server");
    let addr = listener.local_addr().expect("stub addr");
    let handle = std::thread::spawn(move || {
        // Hold every wedged connection open for the lifetime of the stub — dropping the socket would give reqwest an early EOF instead of the timeout we are testing.
        let mut wedged: Vec<TcpStream> = Vec::new();
        for incoming in listener.incoming() {
            let Ok(mut stream) = incoming else { break };
            let mut buf = [0u8; 8192];
            let Ok(read) = stream.read(&mut buf) else {
                continue;
            };
            if read == 0 {
                continue;
            }
            let request = String::from_utf8_lossy(&buf[..read]).to_string();
            let body = request.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
            let parsed: serde_json::Value =
                serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
            let method = parsed
                .get("method")
                .and_then(|m| m.as_str())
                .unwrap_or("")
                .to_string();
            let id = parsed.get("id").and_then(|i| i.as_u64()).unwrap_or(0);

            match method.as_str() {
                "initialize" => json_body(
                    &mut stream,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            // No `resources` capability — keeps the handshake to exactly initialize + tools/list.
                            "capabilities": {},
                            "serverInfo": { "name": "stub", "version": "0" }
                        }
                    })
                    .to_string(),
                ),
                "tools/list" => json_body(
                    &mut stream,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "tools": [{
                                "name": "slowop",
                                "description": "a tool",
                                "inputSchema": { "type": "object", "properties": {} }
                            }]
                        }
                    })
                    .to_string(),
                ),
                "tools/call" => match behaviour {
                    CallBehaviour::Wedge => wedged.push(stream),
                    CallBehaviour::JsonRpcError => json_body(
                        &mut stream,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": { "code": -32602, "message": "file not found" }
                        })
                        .to_string(),
                    ),
                    CallBehaviour::SlowSuccess(ms) => {
                        let mut stream = stream;
                        std::thread::spawn(move || {
                            std::thread::sleep(std::time::Duration::from_millis(ms));
                            json_body(
                                &mut stream,
                                &json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "result": { "content": [{ "type": "text", "text": "done" }] }
                                })
                                .to_string(),
                            );
                        });
                    }
                    CallBehaviour::Success => json_body(
                        &mut stream,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [{ "type": "text", "text": "done" }]
                            }
                        })
                        .to_string(),
                    ),
                },
                // `notifications/initialized` and anything else: accept and close without a body.
                _ => {
                    let _ = stream.write_all(
                        b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                }
            }
        }
    });
    (format!("http://{addr}/sse"), handle)
}

async fn connect_stub(url: String, reporter: Option<Arc<RecordingReporter>>) -> McpConnection {
    connect_stub_named("wedged", url, reporter).await
}

async fn connect_stub_named(
    name: &str,
    url: String,
    reporter: Option<Arc<RecordingReporter>>,
) -> McpConnection {
    // Short so the wedge test does not sit for a minute.
    connect_stub_timeout(name, url, reporter, 2).await
}

async fn connect_stub_timeout(
    name: &str,
    url: String,
    reporter: Option<Arc<RecordingReporter>>,
    timeout_secs: u64,
) -> McpConnection {
    let conn = McpConnection::connect(McpServerConfig {
        name: name.to_string(),
        transport: McpTransport::Sse { url },
        timeout_secs,
        env: vec![],
        headers: vec![],
        oauth_provider: None,
        oauth_config: None,
        taint_scanning: true,
        taint_policy: None,
        taint_rule_sets: empty_taint_rule_sets_handle(),
        roots: vec![],
    })
    .await
    .expect("stub MCP server should complete the handshake");
    assert_eq!(
        conn.tools().len(),
        1,
        "the stub advertises exactly one tool"
    );
    match reporter {
        Some(r) => conn.with_health_reporter(r),
        None => conn,
    }
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

// --------------------------------------------------------------------------- Tests ---------------------------------------------------------------------------

/// The bug: a server that handshook fine and then wedged.
/// The dispatch path must report the timeout so auto-reconnect can eventually engage.
#[tokio::test]
async fn transport_failure_is_reported_from_the_dispatch_path() {
    let (url, _stub) = spawn_stub(CallBehaviour::Wedge);
    let reporter = Arc::new(RecordingReporter::default());
    let conn = connect_stub(url, Some(Arc::clone(&reporter) as Arc<RecordingReporter>)).await;
    let conns = tokio::sync::Mutex::new(vec![McpConnectionEntry::new(conn, 1)]);

    let result = execute_tool_raw("t1", "mcp_wedged_slowop", &json!({}), &ctx(&conns)).await;

    assert!(
        result.is_error,
        "a wedged transport must fail the tool call"
    );
    let failures = reporter.transport_failures();
    assert_eq!(
        failures.len(),
        1,
        "exactly one transport failure must be reported, got {failures:?}"
    );
    assert_eq!(failures[0].0, "wedged", "reported under the server's name");
    assert!(
        reporter.ok_calls().is_empty(),
        "a failed call must not report success"
    );
}

/// The other half of the contract, and the reason the classification exists: a server that answers with an error is healthy.
/// Reporting this would tear down a working server every time a model passed bad arguments.
#[tokio::test]
async fn application_error_is_not_reported_from_the_dispatch_path() {
    let (url, _stub) = spawn_stub(CallBehaviour::JsonRpcError);
    let reporter = Arc::new(RecordingReporter::default());
    let conn = connect_stub(url, Some(Arc::clone(&reporter) as Arc<RecordingReporter>)).await;
    let conns = tokio::sync::Mutex::new(vec![McpConnectionEntry::new(conn, 1)]);

    let result = execute_tool_raw("t1", "mcp_wedged_slowop", &json!({}), &ctx(&conns)).await;

    assert!(
        result.is_error,
        "the application error is still surfaced to the model"
    );
    assert!(
        result.content.contains("file not found"),
        "the server's message must reach the model: {}",
        result.content
    );
    assert!(
        reporter.transport_failures().is_empty(),
        "an application error must NOT be reported as a transport failure: {:?}",
        reporter.transport_failures()
    );
}

/// A successful call resets the consecutive-failure run, so the threshold only ever counts an unbroken sequence of failures.
#[tokio::test]
async fn successful_call_is_reported_as_ok_from_the_dispatch_path() {
    let (url, _stub) = spawn_stub(CallBehaviour::Success);
    let reporter = Arc::new(RecordingReporter::default());
    let conn = connect_stub(url, Some(Arc::clone(&reporter) as Arc<RecordingReporter>)).await;
    let conns = tokio::sync::Mutex::new(vec![McpConnectionEntry::new(conn, 1)]);

    let result = execute_tool_raw("t1", "mcp_wedged_slowop", &json!({}), &ctx(&conns)).await;

    assert!(!result.is_error, "call should succeed: {}", result.content);
    assert_eq!(reporter.ok_calls(), vec![("wedged".to_string(), 1)]);
    assert!(reporter.transport_failures().is_empty());
}

/// A connection with no reporter attached — stand-alone callers, tests — must dispatch exactly as before rather than panicking on the `None`.
#[tokio::test]
async fn a_connection_without_a_reporter_still_dispatches() {
    let (url, _stub) = spawn_stub(CallBehaviour::Success);
    let conn = connect_stub(url, None).await;
    assert!(
        conn.health_reporter().is_none(),
        "connect must not invent a reporter"
    );
    let conns = tokio::sync::Mutex::new(vec![McpConnectionEntry::new(conn, 1)]);

    let result = execute_tool_raw("t1", "mcp_wedged_slowop", &json!({}), &ctx(&conns)).await;

    assert!(!result.is_error, "call should succeed: {}", result.content);
}

// --------------------------------------------------------------------------- Per-connection locking ---------------------------------------------------------------------------

/// `execute_tool_raw` with an empty argument object, as an owned future (no temporaries borrowed across `join!`).
async fn call(
    conns: &McpConnectionList,
    id: &str,
    tool: &str,
) -> librefang_types::tool::ToolResult {
    execute_tool_raw(id, tool, &json!({}), &ctx(conns)).await
}

/// The bug this guards: one process-wide mutex held for the whole `tools/call`, so a slow server delayed calls to every other server and every reader of the connection list.
/// A slow call on server `slow` must not delay a call on server `quick`, nor `lock()` on the list.
#[tokio::test]
async fn a_slow_call_does_not_block_other_servers_or_the_list() {
    let (slow_url, _s1) = spawn_stub(CallBehaviour::SlowSuccess(1500));
    let (quick_url, _s2) = spawn_stub(CallBehaviour::Success);
    let slow = connect_stub_named("slow", slow_url, None).await;
    let quick = connect_stub_named("quick", quick_url, None).await;
    let conns: McpConnectionList = tokio::sync::Mutex::new(vec![
        McpConnectionEntry::new(slow, 1),
        McpConnectionEntry::new(quick, 1),
    ]);

    let slow_call = async {
        let t = std::time::Instant::now();
        let r = call(&conns, "t1", "mcp_slow_slowop").await;
        (r, t.elapsed())
    };
    let others = async {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let t = std::time::Instant::now();
        let r = call(&conns, "t2", "mcp_quick_slowop").await;
        let quick_elapsed = t.elapsed();
        let t = std::time::Instant::now();
        let names: Vec<String> = conns
            .lock()
            .await
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        (r, quick_elapsed, t.elapsed(), names)
    };
    let ((slow_result, slow_elapsed), (quick_result, quick_elapsed, list_elapsed, names)) =
        tokio::join!(slow_call, others);

    assert!(!slow_result.is_error, "{}", slow_result.content);
    assert!(!quick_result.is_error, "{}", quick_result.content);
    assert!(
        slow_elapsed >= std::time::Duration::from_millis(1400),
        "the slow call really was slow: {slow_elapsed:?}"
    );
    assert!(
        quick_elapsed < std::time::Duration::from_millis(700),
        "a call on another server waited for the slow one: {quick_elapsed:?}"
    );
    assert!(
        list_elapsed < std::time::Duration::from_millis(200),
        "reading the connection list waited for the slow call: {list_elapsed:?}"
    );
    assert_eq!(names, vec!["slow", "quick"]);
}

/// Calls to the SAME server still go one at a time after this change (that is step 2's business, and opt-in): the second call starts when the first has finished.
#[tokio::test]
async fn calls_to_the_same_server_are_still_serialized() {
    let (url, _s) = spawn_stub(CallBehaviour::SlowSuccess(600));
    let conn = connect_stub_named("one", url, None).await;
    let conns: McpConnectionList = tokio::sync::Mutex::new(vec![McpConnectionEntry::new(conn, 1)]);

    let t = std::time::Instant::now();
    let (a, b) = tokio::join!(
        call(&conns, "t1", "mcp_one_slowop"),
        call(&conns, "t2", "mcp_one_slowop"),
    );
    assert!(!a.is_error && !b.is_error);
    assert!(
        t.elapsed() >= std::time::Duration::from_millis(1100),
        "two 600 ms calls on one server overlapped: {:?}",
        t.elapsed()
    );
}

/// With `max_concurrent_calls = 2`, two calls to the same server overlap, and a third waits for a free slot.
/// (The hand-rolled SSE transport the stub speaks sends one independent POST per call, so overlap here is real overlap on the wire.)
#[tokio::test]
async fn max_concurrent_calls_lets_calls_to_one_server_overlap_up_to_the_limit() {
    let (url, _s) = spawn_stub(CallBehaviour::SlowSuccess(600));
    let conn = connect_stub_named("one", url, None).await;
    let entry = McpConnectionEntry::new(conn, 2);
    assert_eq!(entry.max_concurrent_calls(), 2);
    let conns: McpConnectionList = tokio::sync::Mutex::new(vec![entry]);

    let t = std::time::Instant::now();
    let (a, b) = tokio::join!(
        call(&conns, "t1", "mcp_one_slowop"),
        call(&conns, "t2", "mcp_one_slowop"),
    );
    let two = t.elapsed();
    assert!(!a.is_error && !b.is_error, "{} / {}", a.content, b.content);
    assert!(
        two < std::time::Duration::from_millis(1000),
        "two calls under a limit of 2 did not overlap: {two:?}"
    );

    let t = std::time::Instant::now();
    let (a, b, c) = tokio::join!(
        call(&conns, "t1", "mcp_one_slowop"),
        call(&conns, "t2", "mcp_one_slowop"),
        call(&conns, "t3", "mcp_one_slowop"),
    );
    let three = t.elapsed();
    assert!(!a.is_error && !b.is_error && !c.is_error);
    assert!(
        three >= std::time::Duration::from_millis(1100),
        "the third call did not wait for a free slot: {three:?}"
    );
}

/// A limit of 0 is a configuration slip, not a way to disable the server: it behaves as 1.
#[tokio::test]
async fn a_limit_of_zero_behaves_as_one() {
    let (url, _s) = spawn_stub(CallBehaviour::Success);
    let conn = connect_stub_named("one", url, None).await;
    assert_eq!(McpConnectionEntry::new(conn, 0).max_concurrent_calls(), 1);
}

// --------------------------------------------------------------------------- Queueing vs tool timeout ---------------------------------------------------------------------------

/// The second call on a server queues behind the first. Its tool timeout (here 900 ms) is smaller than queue + run (600 + 600 ms) but larger than the run alone, so it only succeeds if queueing is left out of the clock.
/// The control, without the exclusion, shows the same calls DO fail under a plain timeout (the behaviour before this change).
#[tokio::test]
async fn time_queued_for_a_slot_does_not_count_against_the_tool_timeout() {
    use librefang_runtime::tool_wait::timeout_excluding_waits;
    use std::time::Duration;

    let (url, _s) = spawn_stub(CallBehaviour::SlowSuccess(600));
    let conn = connect_stub_named("one", url, None).await;
    let conns: McpConnectionList = tokio::sync::Mutex::new(vec![McpConnectionEntry::new(conn, 1)]);
    let budget = Duration::from_millis(900);

    let (a, b) = tokio::join!(
        timeout_excluding_waits(budget, call(&conns, "t1", "mcp_one_slowop")),
        async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            timeout_excluding_waits(budget, call(&conns, "t2", "mcp_one_slowop")).await
        },
    );
    let (a, b) = (
        a.expect("first call within budget"),
        b.expect("queued call must not be cut off while it waits"),
    );
    assert!(!a.is_error && !b.is_error, "{} / {}", a.content, b.content);

    // Control: same shape under a plain timeout, the queued call is cut off.
    let (_, b) = tokio::join!(
        tokio::time::timeout(budget, call(&conns, "t3", "mcp_one_slowop")),
        async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            tokio::time::timeout(budget, call(&conns, "t4", "mcp_one_slowop")).await
        },
    );
    assert!(
        b.is_err(),
        "control: a plain timeout does include the queueing"
    );
}

// --------------------------------------------------------------------------- Reconnect while a call is in flight ---------------------------------------------------------------------------

/// Reconnect closes the old entry while a call may be running on it.
/// `close` must wait for that call (so the transport is not torn down under it), and a call that finishes against a retired entry must not mark the replacement connection of the same name as failing.
#[tokio::test]
async fn closing_waits_for_the_call_in_flight_and_its_outcome_is_not_reported() {
    let (url, _stub) = spawn_stub(CallBehaviour::Wedge);
    let reporter = Arc::new(RecordingReporter::default());
    let conn = connect_stub(url, Some(Arc::clone(&reporter) as Arc<RecordingReporter>)).await;
    let conns: McpConnectionList = tokio::sync::Mutex::new(vec![McpConnectionEntry::new(conn, 1)]);
    let entry = Arc::clone(&conns.lock().await[0]);

    let call = async {
        let r = call(&conns, "t1", "mcp_wedged_slowop").await;
        (r, std::time::Instant::now())
    };
    let closer = async {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        // What reconnect does: take the entry off the list, then close it.
        conns.lock().await.clear();
        entry.close().await;
        std::time::Instant::now()
    };
    let ((result, call_done), close_done) = tokio::join!(call, closer);

    assert!(
        result.is_error,
        "the wedged call still fails on its own timeout"
    );
    assert!(
        close_done >= call_done,
        "close() returned before the call in flight finished"
    );
    assert!(entry.is_retired());
    assert!(
        reporter.transport_failures().is_empty(),
        "the outcome of a call on a retired connection reached the health monitor: {:?}",
        reporter.transport_failures()
    );
    assert!(
        entry.acquire().await.is_err(),
        "close() leaves the entry empty"
    );
}

/// A call that cloned the entry before it was closed and locks it afterwards gets a clear error, not a closed transport.
#[tokio::test]
async fn a_call_queued_behind_close_fails_cleanly() {
    let (url, _stub) = spawn_stub(CallBehaviour::SlowSuccess(800));
    let conn = connect_stub_named("one", url, None).await;
    let conns: McpConnectionList = tokio::sync::Mutex::new(vec![McpConnectionEntry::new(conn, 1)]);
    let entry = Arc::clone(&conns.lock().await[0]);

    let first = call(&conns, "t1", "mcp_one_slowop");
    let closer = async {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        entry.close().await;
    };
    let queued = async {
        // The entry is still in the list (closed but not yet removed): the call finds it, queues behind close().
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        call(&conns, "t2", "mcp_one_slowop").await
    };
    let (first, _, queued) = tokio::join!(first, closer, queued);

    assert!(
        !first.is_error,
        "the call in flight completes: {}",
        first.content
    );
    assert!(queued.is_error);
    assert!(
        queued
            .content
            .contains("was closed while the call was waiting"),
        "{}",
        queued.content
    );
}

// --------------------------------------------------------------------------- Queue wait limit ---------------------------------------------------------------------------

/// F1b: four calls of about a second to one server whose `timeout_secs` is 2, one slot. The fourth waits about three seconds, longer than `timeout_secs`, but every call ahead of it is bounded and finishes: it must be served, not given up on. (A fixed limit of `timeout_secs` failed it with "not sent".)
#[tokio::test]
async fn a_queue_of_slow_calls_longer_than_the_timeout_loses_no_call() {
    let (url, _s) = spawn_stub(CallBehaviour::SlowSuccess(1000));
    let conn = connect_stub_timeout("one", url, None, 2).await;
    let conns: McpConnectionList = tokio::sync::Mutex::new(vec![McpConnectionEntry::new(conn, 1)]);

    let t = std::time::Instant::now();
    let (a, b, c, d) = tokio::join!(
        call(&conns, "t1", "mcp_one_slowop"),
        call(&conns, "t2", "mcp_one_slowop"),
        call(&conns, "t3", "mcp_one_slowop"),
        call(&conns, "t4", "mcp_one_slowop"),
    );
    for (n, r) in [("t1", &a), ("t2", &b), ("t3", &c), ("t4", &d)] {
        assert!(
            !r.is_error,
            "{n} failed after {:?}: {}",
            t.elapsed(),
            r.content
        );
    }
    assert!(
        t.elapsed() >= std::time::Duration::from_millis(3900),
        "the four calls overlapped: {:?}",
        t.elapsed()
    );
}

/// The limit is a net against a stuck server, not an unbounded wait. The only slot is held by the test (a server that never finishes the call in front), `timeout_secs` is 1 and one call is ahead: the queued call must give up with "not sent" after `1 * (ceil(2 / 1) + 1)` = 3 seconds, no sooner and not much later.
#[tokio::test]
async fn a_stuck_server_still_cuts_the_wait_with_not_sent() {
    let (url, _s) = spawn_stub(CallBehaviour::Success);
    let conn = connect_stub_timeout("one", url, None, 1).await;
    let entry = McpConnectionEntry::new(conn, 1);
    let conns: McpConnectionList = tokio::sync::Mutex::new(vec![Arc::clone(&entry)]);

    let held = entry.acquire().await.expect("the only slot");
    let t = std::time::Instant::now();
    let r = call(&conns, "t1", "mcp_one_slowop").await;
    let waited = t.elapsed();
    assert!(r.is_error, "should have given up: {}", r.content);
    assert!(r.content.contains("not sent"), "{}", r.content);
    assert!(
        waited >= std::time::Duration::from_millis(2900),
        "gave up before the limit for a queue of one: {waited:?}"
    );
    assert!(
        waited < std::time::Duration::from_secs(8),
        "gave up far too late: {waited:?}"
    );
    drop(held);
    assert!(!call(&conns, "t2", "mcp_one_slowop").await.is_error);
}

/// A wedged server (accepts calls, never answers) with a queue: each call ahead ends at `timeout_secs`, so the queue advances and every caller gets the timeout of its own call, never "not sent".
#[tokio::test]
async fn a_wedged_server_times_out_each_queued_call_on_its_own() {
    let (url, _s) = spawn_stub(CallBehaviour::Wedge);
    let conn = connect_stub_timeout("one", url, None, 1).await;
    let conns: McpConnectionList = tokio::sync::Mutex::new(vec![McpConnectionEntry::new(conn, 1)]);

    let t = std::time::Instant::now();
    let (a, b, c) = tokio::join!(
        call(&conns, "t1", "mcp_one_slowop"),
        call(&conns, "t2", "mcp_one_slowop"),
        call(&conns, "t3", "mcp_one_slowop"),
    );
    for (n, r) in [("t1", &a), ("t2", &b), ("t3", &c)] {
        assert!(r.is_error, "{n}");
        assert!(!r.content.contains("not sent"), "{n}: {}", r.content);
    }
    assert!(
        t.elapsed() < std::time::Duration::from_secs(10),
        "{:?}",
        t.elapsed()
    );
}
