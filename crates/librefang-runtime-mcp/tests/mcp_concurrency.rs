//! Adversarial tests for `McpConnectionEntry` (one gate per MCP connection) against a real `rmcp` server.
//!
//! Every test names the failure it is trying to provoke. None of them sleeps to "give something time to happen": they wait on the fake server's own counters (`Control::wait_arrived`), on gates the test opens, or on a bounded `within` that turns a hang into a failure naming the hang.
//! Wall-clock time appears only as an upper bound, never as the thing being measured.

#[path = "support/fake_mcp.rs"]
mod fake_mcp;

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use fake_mcp::{serve_http, Control, HttpFake};
use librefang_runtime_mcp::{
    empty_taint_rule_sets_handle, McpCallError, McpConnection, McpConnectionEntry,
    McpConnectionList, McpServerConfig, McpTransport,
};
use serde_json::json;

/// Upper bound for anything that must not hang. A deadlock shows up as this panic, not as a CI job that runs for six hours.
const BOUND: Duration = Duration::from_secs(30);

async fn within<T>(what: &str, fut: impl Future<Output = T>) -> T {
    match tokio::time::timeout(BOUND, fut).await {
        Ok(v) => v,
        Err(_) => panic!("timed out after {BOUND:?} (deadlock or lost wake-up?): {what}"),
    }
}

fn config(name: &str, transport: McpTransport, timeout_secs: u64) -> McpServerConfig {
    McpServerConfig {
        name: name.to_string(),
        transport,
        timeout_secs,
        env: vec![],
        headers: vec![],
        oauth_provider: None,
        oauth_config: None,
        taint_scanning: true,
        taint_policy: None,
        taint_rule_sets: empty_taint_rule_sets_handle(),
        roots: vec![],
    }
}

async fn connect_http(name: &str, url: &str, timeout_secs: u64) -> McpConnection {
    within(
        "connect to the fake HTTP server",
        McpConnection::connect(config(
            name,
            McpTransport::Http {
                url: url.to_string(),
            },
            timeout_secs,
        )),
    )
    .await
    .expect("connect to the fake HTTP server")
}

async fn entry_http(
    name: &str,
    fake: &HttpFake,
    limit: usize,
    timeout_secs: u64,
) -> Arc<McpConnectionEntry> {
    McpConnectionEntry::new(connect_http(name, &fake.url, timeout_secs).await, limit)
}

/// One call the way the dispatcher makes it: take a slot, call on the connection, give the slot back.
async fn call_raw(
    entry: &McpConnectionEntry,
    tool: &str,
    args: serde_json::Value,
) -> Result<String, McpCallError> {
    let slot = entry
        .acquire()
        .await
        .map_err(|_| McpCallError::application("closed"))?;
    let conn = slot
        .connection()
        .ok_or_else(|| McpCallError::application("closed"))?;
    conn.call_tool(&format!("mcp_{}_{tool}", entry.name()), &args)
        .await
}

async fn call(entry: &McpConnectionEntry, tool: &str, id: &str) -> Result<String, String> {
    call_raw(entry, tool, json!({ "id": id }))
        .await
        .map_err(|e| e.message().to_string())
}

// ------------------------------------------------------------------------------------------- permits

/// Risk: a caller queued for a slot is dropped (its agent turn was cancelled) and the permit it was about to receive is lost, so the server's limit shrinks by one for good.
#[tokio::test]
async fn a_queued_acquire_that_is_dropped_does_not_take_the_slot_with_it() {
    let fake = serve_http(Control::new(), false).await;
    let entry = entry_http("srv", &fake, 1, 10).await;

    let held = entry.acquire().await.expect("first slot");
    {
        let waiter = entry.acquire();
        tokio::pin!(waiter);
        // Poll it once so it is really queued, then drop it.
        tokio::select! {
            biased;
            _ = &mut waiter => panic!("a second slot was granted under a limit of 1"),
            _ = std::future::ready(()) => {}
        }
    }
    drop(held);

    let slot = within("acquire after a queued waiter was dropped", entry.acquire()).await;
    assert!(slot.is_ok(), "the slot was lost with the dropped waiter");
    drop(slot);
    fake.stop().await;
}

/// Risk: the future of a call is dropped while the request is on the wire (tool timeout in the agent loop). The slot must come back, and when the server finally answers that abandoned request the answer must not be handed to a different caller.
#[tokio::test]
async fn dropping_a_call_in_flight_frees_the_slot_and_its_late_answer_reaches_nobody_else() {
    let ctl = Control::new();
    let fake = serve_http(Arc::clone(&ctl), false).await;
    let entry = entry_http("srv", &fake, 1, 10).await;

    {
        let abandoned = call(&entry, "gate", "a");
        tokio::pin!(abandoned);
        tokio::select! {
            r = &mut abandoned => panic!("the gated call answered by itself: {r:?}"),
            _ = within("the abandoned call reaches the server", ctl.wait_arrived(1)) => {}
        }
    } // dropped with the request on the wire

    let b = within("call after the abandoned one", call(&entry, "echo", "b")).await;
    assert_eq!(b.as_deref(), Ok("echo:b"));

    // Now the server finishes the abandoned request. Its answer is late and nobody is waiting for it.
    ctl.open("a");
    for i in 0..5 {
        let id = format!("c{i}");
        let r = within("calls after the late answer", call(&entry, "echo", &id)).await;
        assert_eq!(r, Ok(format!("echo:{id}")), "a late answer crossed over");
    }
    fake.stop().await;
}

// ------------------------------------------------------------------------------------------- close

/// Risk: `close` with callers queued. They must fail at once (not wait their turn on a connection going away), and `close` itself must wait for the call that is running before it tears the transport down.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn close_fails_queued_callers_at_once_and_waits_for_the_running_call() {
    let ctl = Control::new();
    let fake = serve_http(Arc::clone(&ctl), false).await;
    let entry = entry_http("srv", &fake, 1, 20).await;

    let running = {
        let e = Arc::clone(&entry);
        tokio::spawn(async move { call(&e, "gate", "run").await })
    };
    within("running call reaches the server", ctl.wait_arrived(1)).await;

    let queued: Vec<_> = (0..4)
        .map(|i| {
            let e = Arc::clone(&entry);
            tokio::spawn(async move { call(&e, "echo", &format!("q{i}")).await })
        })
        .collect();
    let closer = {
        let e = Arc::clone(&entry);
        tokio::spawn(async move { e.close().await })
    };

    for q in queued {
        let r = within("a queued caller must fail, not wait", q)
            .await
            .unwrap();
        assert_eq!(r, Err("closed".to_string()));
    }
    assert!(
        !closer.is_finished(),
        "close() returned while a call was still running on the connection"
    );
    assert!(entry.is_retired());

    ctl.open("run");
    let r = within("the running call completes", running).await.unwrap();
    assert_eq!(
        r,
        Ok("gate:run".to_string()),
        "close() cut the running call short"
    );
    within("close finishes once the call is done", closer)
        .await
        .unwrap();
    assert!(entry.acquire().await.is_err());
    fake.stop().await;
}

/// Risk: two closers at once (a hot reload and an API disconnect) deadlock on the write lock or double-close the transport.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_concurrent_closes_both_return() {
    let ctl = Control::new();
    let fake = serve_http(Arc::clone(&ctl), false).await;
    let entry = entry_http("srv", &fake, 2, 20).await;

    let running = {
        let e = Arc::clone(&entry);
        tokio::spawn(async move { call(&e, "gate", "run").await })
    };
    within("running call reaches the server", ctl.wait_arrived(1)).await;
    let closers: Vec<_> = (0..2)
        .map(|_| {
            let e = Arc::clone(&entry);
            tokio::spawn(async move { e.close().await })
        })
        .collect();
    ctl.open("run");
    assert_eq!(
        within("running call", running).await.unwrap(),
        Ok("gate:run".to_string())
    );
    for c in closers {
        within("a closer", c).await.unwrap();
    }
    fake.stop().await;
}

/// Risk: a call is made after the entry was closed (a reference cloned out of the list before removal). It must get a clean error, immediately.
#[tokio::test]
async fn a_call_after_close_fails_cleanly() {
    let fake = serve_http(Control::new(), false).await;
    let entry = entry_http("srv", &fake, 2, 10).await;
    entry.close().await;
    for _ in 0..3 {
        let r = within("call after close", call(&entry, "echo", "x")).await;
        assert_eq!(r, Err("closed".to_string()));
    }
    fake.stop().await;
}

// ------------------------------------------------------------------------------------------- limits

/// Risk: odd values of `max_concurrent_calls`. 0 means 1; a very large value must work; a value beyond what a semaphore can hold must not panic the daemon.
#[tokio::test]
async fn limit_zero_is_one_and_a_huge_limit_works() {
    let fake = serve_http(Control::new(), false).await;

    let zero = entry_http("srv", &fake, 0, 10).await;
    assert_eq!(zero.max_concurrent_calls(), 1);

    // What `u32::MAX` from the config file becomes.
    let big = entry_http("srv", &fake, u32::MAX as usize, 10).await;
    let (a, b, c) = within("three calls under a huge limit", async {
        tokio::join!(
            call(&big, "echo", "1"),
            call(&big, "echo", "2"),
            call(&big, "echo", "3")
        )
    })
    .await;
    assert_eq!(
        (a.as_deref(), b.as_deref(), c.as_deref()),
        (Ok("echo:1"), Ok("echo:2"), Ok("echo:3"))
    );
    fake.stop().await;
}

/// Risk: `McpConnectionEntry::new` is public and takes a `usize`. `tokio::sync::Semaphore::new` panics above `Semaphore::MAX_PERMITS`, so an unclamped value turns a configuration slip into a panic at connect time.
#[tokio::test]
async fn an_absurd_limit_is_clamped_instead_of_panicking() {
    let fake = serve_http(Control::new(), false).await;
    let conn = connect_http("srv", &fake.url, 10).await;
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        McpConnectionEntry::new(conn, usize::MAX)
    }));
    assert!(
        built.is_ok(),
        "McpConnectionEntry::new(conn, usize::MAX) panicked"
    );
    assert!(built.unwrap().max_concurrent_calls() >= 1);
    fake.stop().await;
}

// ------------------------------------------------------------------------------------------- correlation

/// Risk: with several requests in flight on one rmcp session, an answer is delivered to the wrong caller.
/// Eight calls are held at the server (so they are really concurrent, which the test proves with the server's own high-water mark), then released in reverse order of arrival; each caller must get exactly its own answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn answers_released_in_reverse_order_reach_the_right_callers_over_sse() {
    let ctl = Control::new();
    let fake = serve_http(Arc::clone(&ctl), false).await;
    let entry = entry_http("srv", &fake, 8, 30).await;

    let handles: Vec<_> = (0..8)
        .map(|i| {
            let e = Arc::clone(&entry);
            tokio::spawn(async move { call(&e, "gate", &i.to_string()).await })
        })
        .collect();
    within(
        "all eight calls reach the server together",
        ctl.wait_arrived(8),
    )
    .await;
    assert_eq!(ctl.max_in_flight(), 8);

    for (i, h) in handles.into_iter().enumerate().rev() {
        ctl.open(&i.to_string());
        let r = within("a released call", h).await.unwrap();
        assert_eq!(
            r,
            Ok(format!("gate:{i}")),
            "caller {i} got somebody else's answer"
        );
    }
    fake.stop().await;
}

/// Same risk on the plain-JSON flavour of Streamable HTTP, where the client library may or may not overlap the requests.
/// The test does not claim overlap; it claims that whatever interleaving happens, answers stay with their callers and nothing hangs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn answers_reach_the_right_callers_over_plain_json() {
    let ctl = Control::new();
    let fake = serve_http(Arc::clone(&ctl), true).await;
    let entry = entry_http("srv", &fake, 8, 30).await;

    let handles: Vec<_> = (0..8u64)
        .map(|i| {
            let e = Arc::clone(&entry);
            tokio::spawn(async move {
                call_raw(
                    &e,
                    "sleep",
                    json!({ "id": i.to_string(), "ms": (8 - i) * 15 }),
                )
                .await
                .map_err(|e| e.message().to_string())
            })
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        let r = within("a call over plain JSON", h).await.unwrap();
        assert_eq!(r, Ok(format!("sleep:{i}")));
    }
    eprintln!(
        "plain JSON: server high-water mark of concurrent calls = {}",
        ctl.max_in_flight()
    );
    fake.stop().await;
}

/// Characterisation, not a verdict: what a call abandoned by its caller (tool timeout in the agent loop) does to the calls that come after it.
/// Over SSE the next call is answered while the abandoned request is still pending. Over plain JSON the client library handles one POST at a time per connection, so the next call waits behind the abandoned request until the server answers it; the semaphore slot is free but the connection is not.
/// A server that never answers an abandoned request therefore starves a plain-JSON connection even though `max_concurrent_calls` is not exhausted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_abandoned_call_blocks_a_plain_json_connection_but_not_an_sse_one() {
    for json_response in [false, true] {
        let ctl = Control::new();
        let fake = serve_http(Arc::clone(&ctl), json_response).await;
        let entry = entry_http("srv", &fake, 2, 30).await;
        {
            let abandoned = call(&entry, "gate", "a");
            tokio::pin!(abandoned);
            tokio::select! {
                r = &mut abandoned => panic!("the gated call answered by itself: {r:?}"),
                _ = within("the abandoned call reaches the server", ctl.wait_arrived(1)) => {}
            }
        }
        let next =
            tokio::time::timeout(Duration::from_millis(700), call(&entry, "echo", "b")).await;
        if json_response {
            assert!(
                next.is_err(),
                "plain JSON: the next call was answered while an abandoned request was pending: {next:?}"
            );
        } else {
            assert_eq!(
                next.expect("SSE: the next call waited behind the abandoned one"),
                Ok("echo:b".to_string())
            );
        }
        // Whatever the transport, once the server answers the abandoned request everything flows again.
        ctl.open("a");
        assert_eq!(
            within(
                "call after releasing the abandoned request",
                call(&entry, "echo", "c")
            )
            .await,
            Ok("echo:c".to_string())
        );
        fake.stop().await;
    }
}

/// Risk: a tool-level error is mistaken for a transport failure (which would tear down a healthy server) or leaves the slot taken.
#[tokio::test]
async fn a_tool_error_is_an_application_error_and_leaves_the_server_usable() {
    let fake = serve_http(Control::new(), false).await;
    let entry = entry_http("srv", &fake, 1, 10).await;
    let err = within("failing call", call_raw(&entry, "fail", json!({"id": "f"})))
        .await
        .expect_err("the tool reports an error");
    assert!(
        !err.is_transport(),
        "a tool error was classified as transport: {err}"
    );
    assert_eq!(
        within("call after a tool error", call(&entry, "echo", "ok")).await,
        Ok("echo:ok".to_string())
    );
    fake.stop().await;
}

// ------------------------------------------------------------------------------------------- stdio

fn stdio_server() -> PathBuf {
    let mut p = std::env::current_exe().expect("current exe");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    let exe = p.join("examples").join(if cfg!(windows) {
        "fake_mcp_stdio.exe"
    } else {
        "fake_mcp_stdio"
    });
    assert!(
        exe.exists(),
        "{} is missing: run these tests with plain `cargo test -p librefang-runtime-mcp` so the example is built",
        exe.display()
    );
    exe
}

async fn entry_stdio(name: &str, limit: usize, timeout_secs: u64) -> Arc<McpConnectionEntry> {
    let conn = within(
        "spawn the stdio fake",
        McpConnection::connect(config(
            name,
            McpTransport::Stdio {
                command: stdio_server().to_string_lossy().into_owned(),
                args: vec![],
            },
            timeout_secs,
        )),
    )
    .await
    .expect("connect to the stdio fake");
    McpConnectionEntry::new(conn, limit)
}

/// Risk: out-of-order answers on a single pipe are matched to the wrong request.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stdio_out_of_order_answers_reach_the_right_callers() {
    let entry = entry_stdio("srv", 8, 30).await;
    let handles: Vec<_> = (0..8u64)
        .map(|i| {
            let e = Arc::clone(&entry);
            tokio::spawn(async move {
                call_raw(
                    &e,
                    "sleep",
                    json!({ "id": i.to_string(), "ms": (8 - i) * 40 }),
                )
                .await
                .map_err(|e| e.message().to_string())
            })
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        let r = within("a stdio call", h).await.unwrap();
        assert_eq!(r, Ok(format!("sleep:{i}")));
    }
    entry.close().await;
}

/// Risk: the stdio child dies with calls in flight. Every call must fail (not wait for its timeout), the slots must come back, later calls must fail fast instead of queueing forever, and closing the entry that still holds the dead connection must return.
/// "Dead server keeps its place in the list" is the existing behaviour; what must not happen is a hang.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stdio_server_dying_mid_call_fails_the_calls_and_frees_the_slots() {
    let entry = entry_stdio("srv", 3, 25).await;
    let started = std::time::Instant::now();

    let hung = {
        let e = Arc::clone(&entry);
        tokio::spawn(async move { call_raw(&e, "hang", json!({"id": "h"})).await })
    };
    let dying = {
        let e = Arc::clone(&entry);
        tokio::spawn(async move { call_raw(&e, "die", json!({"id": "d"})).await })
    };
    let hung = within("the call in flight when the child died", hung)
        .await
        .unwrap();
    let dying = within("the call that killed the child", dying)
        .await
        .unwrap();
    assert!(hung.is_err() && dying.is_err());
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "the calls waited for their {}s timeout instead of failing when the pipe closed: {:?}",
        25,
        started.elapsed()
    );

    for i in 0..4 {
        let r = within(
            "a call to the dead server",
            call_raw(&entry, "echo", json!({"id": i.to_string()})),
        )
        .await;
        let err = r.expect_err("the server is dead");
        assert!(
            err.is_transport(),
            "a dead stdio server must be a transport error, got: {err}"
        );
    }
    // All three slots are free: three simultaneous acquires succeed.
    let a = within("slot 1", entry.acquire()).await.expect("slot");
    let b = within("slot 2", entry.acquire()).await.expect("slot");
    let c = within("slot 3", entry.acquire()).await.expect("slot");
    drop((a, b, c));
    within("closing an entry with a dead connection", entry.close()).await;
}

// ------------------------------------------------------------------------------------------- stress

/// xorshift64*: a seeded generator with no dependency, so a failing seed reproduces the same operation sequence.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4BB5_7F6D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

async fn entry_of(list: &McpConnectionList, name: &str) -> Option<Arc<McpConnectionEntry>> {
    list.lock().await.iter().find(|e| e.name() == name).cloned()
}

async fn stress_once(seed: u64) {
    let sa = serve_http(Control::new(), false).await; // SSE: requests can overlap
    let sb = serve_http(Control::new(), true).await; // plain JSON
    let list: Arc<McpConnectionList> = Arc::new(tokio::sync::Mutex::new(vec![
        entry_http("sa", &sa, 3, 20).await,
        entry_http("sb", &sb, 2, 20).await,
    ]));
    let (url_a, url_b) = (sa.url.clone(), sb.url.clone());

    let mut tasks = tokio::task::JoinSet::new();

    // Callers: random mix of complete calls, abandoned calls and slot holders against both servers.
    for w in 0..24u64 {
        let list = Arc::clone(&list);
        tasks.spawn(async move {
            let mut rng = Rng::new(seed.wrapping_mul(1000).wrapping_add(w));
            let mut abandoned = 0usize;
            for op in 0..25u64 {
                let name = if rng.below(2) == 0 { "sa" } else { "sb" };
                let id = format!("{w}-{op}");
                let Some(entry) = entry_of(&list, name).await else {
                    continue;
                };
                match rng.below(8) {
                    0..=3 => match call(&entry, "echo", &id).await {
                        Ok(v) => assert_eq!(v, format!("echo:{id}"), "seed {seed}: crossed answer"),
                        Err(e) => assert_eq!(e, "closed", "seed {seed}: unexpected error {e}"),
                    },
                    4 => match call_raw(&entry, "sleep", json!({"id": id, "ms": rng.below(20)}))
                        .await
                    {
                        Ok(v) => {
                            assert_eq!(v, format!("sleep:{id}"), "seed {seed}: crossed answer")
                        }
                        Err(e) => {
                            assert_eq!(e.message(), "closed", "seed {seed}: unexpected error {e}")
                        }
                    },
                    5 => {
                        // Abandon a call that is still running on the server. It does finish by itself after a while: over plain JSON the client library keeps the connection busy with an abandoned request until the server answers it, so one that never finished would starve every later call on that server (see `an_abandoned_call_blocks_a_plain_json_connection_but_not_an_sse_one`).
                        let budget = Duration::from_millis(1 + rng.below(15));
                        let running =
                            call_raw(&entry, "sleep", json!({"id": id, "ms": 40 + rng.below(80)}));
                        if let Ok(Err(e)) = tokio::time::timeout(budget, running).await {
                            assert_eq!(e.message(), "closed", "seed {seed}: unexpected error {e}");
                        }
                        abandoned += 1;
                    }
                    6 => {
                        if let Ok(slot) = entry.acquire().await {
                            for _ in 0..rng.below(4) {
                                tokio::task::yield_now().await;
                            }
                            drop(slot);
                        }
                    }
                    _ => tokio::task::yield_now().await,
                }
            }
            abandoned
        });
    }

    // Reconnector: swaps in a fresh connection and closes the old one in the background, as the kernel does.
    {
        let list = Arc::clone(&list);
        tasks.spawn(async move {
            let mut rng = Rng::new(seed ^ 0xABCD);
            let mut closers = Vec::new();
            for _ in 0..6 {
                for _ in 0..rng.below(20) {
                    tokio::task::yield_now().await;
                }
                let (name, url, limit) = if rng.below(2) == 0 {
                    ("sa", &url_a, 3)
                } else {
                    ("sb", &url_b, 2)
                };
                let fresh = McpConnectionEntry::new(connect_http(name, url, 20).await, limit);
                let old: Vec<_> = {
                    let mut g = list.lock().await;
                    let mut old = Vec::new();
                    g.retain(|e| {
                        if e.name() == name {
                            e.retire();
                            old.push(Arc::clone(e));
                            false
                        } else {
                            true
                        }
                    });
                    g.push(fresh);
                    old
                };
                closers.push(tokio::spawn(async move {
                    for e in old {
                        e.close().await;
                    }
                }));
            }
            for c in closers {
                c.await.expect("closer");
            }
            0usize
        });
    }

    let all = async {
        while let Some(done) = tasks.join_next().await {
            done.unwrap_or_else(|e| panic!("seed {seed}: a worker failed: {e}"));
        }
    };
    if tokio::time::timeout(Duration::from_secs(90), all)
        .await
        .is_err()
    {
        panic!("seed {seed}: the stress run did not finish in 90s: deadlock");
    }

    // No leaked permits: every live entry can hand out all of its slots at once.
    let live: Vec<_> = list.lock().await.iter().cloned().collect();
    assert_eq!(
        live.len(),
        2,
        "seed {seed}: the list lost or duplicated a server"
    );
    for e in &live {
        let mut slots = Vec::new();
        for _ in 0..e.max_concurrent_calls() {
            slots.push(
                within(&format!("seed {seed}: a slot of {}", e.name()), e.acquire())
                    .await
                    .unwrap_or_else(|_| panic!("seed {seed}: live entry {} is closed", e.name())),
            );
        }
        drop(slots);
        e.close().await;
    }
    sa.stop().await;
    sb.stop().await;
}

/// Risk: any ordering of calls, abandoned calls, slot holders, closes and swaps that deadlocks, loses a permit, or crosses answers.
/// Each seed is a different, reproducible interleaving of operations (the scheduler adds its own on top); the failing seed is in the message.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stress_random_load_never_deadlocks_or_leaks_slots() {
    for seed in [1u64, 2, 3, 5, 8, 13, 21, 34] {
        stress_once(seed).await;
    }
}
