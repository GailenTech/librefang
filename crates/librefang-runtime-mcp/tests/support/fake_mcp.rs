//! In-process fake MCP server for the concurrency tests.
//!
//! It is a real `rmcp` server, so the client under test speaks the genuine protocol (initialize, session handling, request ids), but every tool is driven by the test:
//!
//! - `echo {id}` answers at once with `echo:<id>`.
//! - `sleep {id, ms}` answers `sleep:<id>` after `ms` milliseconds.
//! - `gate {id}` blocks until the test calls [`Control::open`] for that `id`, then answers `gate:<id>`. This is how a test decides, without sleeping, which of several calls in flight finishes first.
//! - `hang {id}` never answers (until the client cancels the request).
//! - `fail {id}` answers with a tool-level error.
//! - `die {id}` exits the whole process; only honoured when [`Control::allow_exit`] is set (the stdio child), never in the test process.
//!
//! The server counts what it sees (`arrived`, `in_flight`, `max_in_flight`) so a test can wait for "N calls have reached the server" instead of guessing with a sleep.
//!
//! This file is included with `#[path]` by every test target that needs it; it must not depend on any librefang crate.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, InitializeResult,
    JsonObject, ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler};
use tokio::sync::{Notify, Semaphore};

/// Shared state between a test and the fake server it started.
#[derive(Default)]
pub struct Control {
    gates: Mutex<HashMap<String, Arc<Semaphore>>>,
    arrived: AtomicUsize,
    arrived_notify: Notify,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    completed: AtomicUsize,
    cancelled: AtomicUsize,
    allow_exit: AtomicBool,
    extra_tool: AtomicBool,
}

impl Control {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn gate(&self, id: &str) -> Arc<Semaphore> {
        let mut gates = self.gates.lock().unwrap();
        Arc::clone(
            gates
                .entry(id.to_string())
                .or_insert_with(|| Arc::new(Semaphore::new(0))),
        )
    }

    /// Let one `gate` call with this `id` finish (now, or when it arrives).
    pub fn open(&self, id: &str) {
        self.gate(id).add_permits(1);
    }

    /// Let every `gate` call with this `id` finish, present and future.
    pub fn open_for_good(&self, id: &str) {
        self.gate(id).add_permits(1 << 20);
    }

    /// Calls that reached the server so far.
    pub fn arrived(&self) -> usize {
        self.arrived.load(Ordering::SeqCst)
    }

    /// Calls the server is working on right now.
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// The highest number of calls the server was working on at once.
    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight.load(Ordering::SeqCst)
    }

    /// Calls the server answered.
    pub fn completed(&self) -> usize {
        self.completed.load(Ordering::SeqCst)
    }

    /// Calls the client cancelled before the server answered.
    pub fn cancelled(&self) -> usize {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Advertise one more tool (`extra`, which behaves like `echo`) to sessions that start from now on.
    pub fn set_extra_tool(&self, on: bool) {
        self.extra_tool.store(on, Ordering::SeqCst);
    }

    pub fn set_allow_exit(&self) {
        self.allow_exit.store(true, Ordering::SeqCst);
    }

    /// Wait until at least `n` calls have reached the server. Callers wrap this in a timeout.
    pub async fn wait_arrived(&self, n: usize) {
        loop {
            let notified = self.arrived_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.arrived.load(Ordering::SeqCst) >= n {
                return;
            }
            notified.await;
        }
    }

    fn arrive(&self) {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        self.arrived.fetch_add(1, Ordering::SeqCst);
        self.arrived_notify.notify_waiters();
    }

    fn leave(&self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The `rmcp` handler. One instance per MCP session.
#[derive(Clone)]
pub struct FakeServer {
    ctl: Arc<Control>,
}

impl FakeServer {
    pub fn new(ctl: Arc<Control>) -> Self {
        Self { ctl }
    }
}

fn schema() -> Arc<JsonObject> {
    let v = serde_json::json!({
        "type": "object",
        "properties": { "id": { "type": "string" }, "ms": { "type": "integer" } }
    });
    Arc::new(v.as_object().expect("object literal").clone())
}

fn tools() -> Vec<Tool> {
    ["echo", "sleep", "gate", "hang", "fail", "die"]
        .iter()
        .map(|name| Tool::new(*name, "fake tool", schema()))
        .collect()
}

fn text(s: String) -> Result<CallToolResponse, McpError> {
    Ok(CallToolResponse::Complete(CallToolResult::success(vec![
        ContentBlock::text(s),
    ])))
}

impl ServerHandler for FakeServer {
    fn get_info(&self) -> ServerInfo {
        InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let mut all = tools();
        if self.ctl.extra_tool.load(Ordering::SeqCst) {
            all.push(Tool::new("extra", "fake tool", schema()));
        }
        Ok(ListToolsResult::with_all_items(all))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let args = request.arguments.unwrap_or_default();
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let ms = args.get("ms").and_then(|v| v.as_u64()).unwrap_or(0);
        let ctl = Arc::clone(&self.ctl);
        ctl.arrive();
        // `leave` runs however this future ends, including when rmcp drops it because the client went away.
        struct Leave(Arc<Control>);
        impl Drop for Leave {
            fn drop(&mut self) {
                self.0.leave();
            }
        }
        let _leave = Leave(Arc::clone(&ctl));

        let out = match &*request.name {
            "echo" => text(format!("echo:{id}")),
            "extra" if ctl.extra_tool.load(Ordering::SeqCst) => text(format!("echo:{id}")),
            "sleep" => {
                tokio::time::sleep(Duration::from_millis(ms)).await;
                text(format!("sleep:{id}"))
            }
            "gate" => {
                let gate = ctl.gate(&id);
                tokio::select! {
                    permit = gate.acquire_owned() => {
                        permit.expect("gate never closed").forget();
                        text(format!("gate:{id}"))
                    }
                    _ = context.ct.cancelled() => {
                        ctl.cancelled.fetch_add(1, Ordering::SeqCst);
                        Err(McpError::internal_error("cancelled", None))
                    }
                }
            }
            "hang" => {
                context.ct.cancelled().await;
                ctl.cancelled.fetch_add(1, Ordering::SeqCst);
                Err(McpError::internal_error("cancelled", None))
            }
            "fail" => Ok(CallToolResponse::Complete(CallToolResult::error(vec![
                ContentBlock::text(format!("boom:{id}")),
            ]))),
            "die" => {
                if ctl.allow_exit.load(Ordering::SeqCst) {
                    std::process::exit(3);
                }
                Err(McpError::invalid_params(
                    "die is only for the stdio child",
                    None,
                ))
            }
            other => Err(McpError::invalid_params(
                format!("unknown tool {other}"),
                None,
            )),
        };
        ctl.completed.fetch_add(1, Ordering::SeqCst);
        out
    }
}

/// A Streamable HTTP server running in this process.
pub struct HttpFake {
    /// `http://127.0.0.1:<port>/mcp`
    pub url: String,
    /// `127.0.0.1:<port>`
    pub addr: std::net::SocketAddr,
    pub ctl: Arc<Control>,
    stop: tokio_util::sync::CancellationToken,
    task: tokio::task::JoinHandle<()>,
}

impl HttpFake {
    /// Stop accepting connections and end every session.
    pub async fn stop(self) {
        self.stop.cancel();
        self.task.abort();
        let _ = self.task.await;
    }
}

/// Start a Streamable HTTP MCP server on a free loopback port.
///
/// With `json_response` the server answers each POST with a plain JSON body (stateless); without it, with an SSE stream per request (legacy session mode), which is the case where several requests can be in flight on one connection.
pub async fn serve_http(ctl: Arc<Control>, json_response: bool) -> HttpFake {
    serve_http_at(ctl, json_response, "127.0.0.1:0").await
}

/// Like [`serve_http`], on a given address (to bring a stopped server back on the port a client is configured for).
pub async fn serve_http_at(ctl: Arc<Control>, json_response: bool, bind: &str) -> HttpFake {
    use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
    use rmcp::transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService,
    };

    let stop = tokio_util::sync::CancellationToken::new();
    let mut config = StreamableHttpServerConfig::default();
    config.cancellation_token = stop.child_token();
    if json_response {
        config.legacy_session_mode = false;
        config.json_response = true;
    }
    let factory_ctl = Arc::clone(&ctl);
    let service = StreamableHttpService::new(
        move || Ok(FakeServer::new(Arc::clone(&factory_ctl))),
        Arc::new(LocalSessionManager::default()),
        config,
    );
    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .expect("bind a loopback port");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    HttpFake {
        url: format!("http://{addr}/mcp"),
        addr,
        ctl,
        stop,
        task,
    }
}
