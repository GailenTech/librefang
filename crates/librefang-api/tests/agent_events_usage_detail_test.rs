//! Integration test: `GET /api/agents/{id}/events` carries per-turn detail and
//! can be polled with `since`.
//!
//! Two things are pinned here:
//!
//! 1. A turn sent with a cron-shaped `SenderContext` (`channel = "cron"`; it
//!    does NOT go through `cron_tick`, that path is covered by
//!    `dispatched_cron_turn_writes_usage_row_with_channel_iterations_and_cache`
//!    in the kernel) leaves a `usage_events` row with `channel = "cron"`,
//!    `iterations` and the prompt-cache token counts, and the route returns
//!    them. The kernel points at a local OpenAI-compatible stub so the whole
//!    turn runs (same technique as `auto_reply_sender_identity_test`).
//! 2. The route's `since` cursor: strictly newer rows only, oldest first, capped
//!    by `limit`, and the unchanged newest-first order without it.
//!
//! Run: cargo test -p librefang-api --test agent_events_usage_detail_test

use axum::body::{to_bytes, Body};
use axum::http::{header, Method, Request, StatusCode};
use librefang_api::server;
use librefang_channels::types::SenderContext;
use librefang_kernel::KernelApi;
use librefang_kernel::LibreFangKernel;
use librefang_memory::usage::UsageRecord;
use librefang_types::agent::{AgentId, AgentManifest};
use librefang_types::config::{DefaultModelConfig, KernelConfig};
use std::sync::Arc;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const API_KEY: &str = "test-secret-key";
const PROVIDER_KEY_ENV: &str = "AGENT_EVENTS_USAGE_DETAIL_TEST_KEY";

struct Harness {
    app: axum::Router,
    state: Arc<librefang_api::routes::AppState>,
    _tmp: tempfile::TempDir,
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.state.kernel.shutdown();
    }
}

async fn mount_completion(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "done" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 3,
                "total_tokens": 53,
                "prompt_tokens_details": { "cached_tokens": 20 }
            }
        })))
        .mount(server)
        .await;
}

async fn boot(server: &MockServer) -> Harness {
    std::env::set_var(PROVIDER_KEY_ENV, "sk-test-agent-events-usage-detail");
    let tmp = tempfile::tempdir().expect("tempdir");
    librefang_kernel::registry_sync::seed_registry_fixture_for_tests(tmp.path());
    let config = KernelConfig {
        home_dir: tmp.path().to_path_buf(),
        data_dir: tmp.path().join("data"),
        api_key: API_KEY.to_string(),
        default_model: DefaultModelConfig {
            provider: "openai".to_string(),
            model: "gpt-4o-mini".to_string(),
            api_key_env: PROVIDER_KEY_ENV.to_string(),
            base_url: Some(server.uri()),
            ..DefaultModelConfig::default()
        },
        ..KernelConfig::default()
    };
    let kernel = Arc::new(LibreFangKernel::boot_with_config(config).expect("kernel boot"));
    kernel.clone().set_self_handle();
    let (app, state) = server::build_router(kernel, "127.0.0.1:0".parse().expect("addr")).await;
    Harness {
        app,
        state,
        _tmp: tmp,
    }
}

fn spawn_agent(h: &Harness, name: &str) -> AgentId {
    h.state
        .kernel
        .spawn_agent_typed(AgentManifest {
            name: name.to_string(),
            source_template: None,
            ..AgentManifest::default()
        })
        .expect("spawn agent")
}

async fn get(h: &Harness, uri: &str) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method(Method::GET)
        .uri(uri)
        .header(header::AUTHORIZATION, format!("Bearer {API_KEY}"))
        .body(Body::empty())
        .unwrap();
    let resp = h.app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 4 * 1024 * 1024).await.unwrap();
    let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, body)
}

fn seed(h: &Harness, agent: AgentId, input_tokens: u64) {
    h.state
        .kernel
        .memory_substrate()
        .usage()
        .record(&UsageRecord {
            agent_id: agent,
            provider: "p".to_string(),
            model: "m".to_string(),
            input_tokens,
            output_tokens: 1,
            cost_usd: 0.01,
            latency_ms: 4,
            channel: Some("telegram".to_string()),
            iterations: Some(2),
            cache_read_input_tokens: Some(0),
            cache_creation_input_tokens: Some(9),
            ..Default::default()
        })
        .expect("record");
}

fn inputs(body: &serde_json::Value) -> Vec<u64> {
    body["events"]
        .as_array()
        .expect("events array")
        .iter()
        .map(|e| e["input_tokens"].as_u64().unwrap())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn turn_with_cron_sender_context_records_channel_iterations_and_cache_tokens() {
    let server = MockServer::start().await;
    mount_completion(&server).await;
    let h = boot(&server).await;
    let agent = spawn_agent(&h, "cron-usage-agent");

    let cron = SenderContext {
        channel: "cron".to_string(),
        user_id: String::new(),
        display_name: "cron".to_string(),
        is_internal_cron: true,
        is_internal_system: true,
        ..Default::default()
    };
    h.state
        .kernel
        .send_message_with_sender_context(agent, "tick", cron, None)
        .await
        .expect("cron-shaped turn runs");

    let (status, body) = get(&h, &format!("/api/agents/{agent}/events")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = body["events"].as_array().unwrap();
    assert_eq!(events.len(), 1, "one turn, one row: {body}");
    let row = &events[0];
    assert_eq!(row["channel"], "cron");
    assert!(row["session_id"].is_string(), "{row}");
    assert_eq!(row["iterations"], 1);
    assert_eq!(row["input_tokens"], 50);
    assert_eq!(row["cache_read_input_tokens"], 20);
    assert_eq!(row["cache_creation_input_tokens"], 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn events_row_has_the_documented_shape_and_old_rows_are_null() {
    let server = MockServer::start().await;
    let h = boot(&server).await;
    let agent = spawn_agent(&h, "shape-agent");
    // A record from a writer that does not know the new fields.
    h.state
        .kernel
        .memory_substrate()
        .usage()
        .record(&UsageRecord::anonymous(agent, "p", "m", 7, 1, 0.5, 2, 30))
        .unwrap();

    let (status, body) = get(&h, &format!("/api/agents/{agent}/events")).await;
    assert_eq!(status, StatusCode::OK);
    let row = body["events"][0].as_object().expect("row object");
    let mut keys: Vec<&str> = row.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "channel",
            "cost_usd",
            "input_tokens",
            "iterations",
            "latency_ms",
            "model",
            "output_tokens",
            "provider",
            "session_id",
            "timestamp",
            "tool_calls",
        ]
    );
    for k in [
        "channel",
        "session_id",
        "iterations",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ] {
        assert!(row[k].is_null(), "{k} must be null for an unattributed row");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn since_filters_strictly_orders_ascending_and_caps() {
    let server = MockServer::start().await;
    let h = boot(&server).await;
    let agent = spawn_agent(&h, "since-agent");
    let other = spawn_agent(&h, "other-agent");
    for i in 1..=5u64 {
        seed(&h, agent, i);
        seed(&h, other, 900 + i);
        std::thread::sleep(std::time::Duration::from_millis(3));
    }

    // No `since`: unchanged, newest first.
    let (_, body) = get(&h, &format!("/api/agents/{agent}/events")).await;
    assert_eq!(inputs(&body), vec![5, 4, 3, 2, 1]);

    // `since` in the past: oldest first, the first `limit` rows.
    let (status, page1) = get(
        &h,
        &format!("/api/agents/{agent}/events?since=2000-01-01T00:00:00Z&limit=2"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(inputs(&page1), vec![1, 2]);

    // The cursor is the last timestamp seen (as the route returned it,
    // URL-encoded because it carries a `+`): strictly newer rows only.
    let cursor = page1["events"][1]["timestamp"].as_str().unwrap();
    let enc = cursor.replace('+', "%2B");
    let (_, page2) = get(
        &h,
        &format!("/api/agents/{agent}/events?since={enc}&limit=2"),
    )
    .await;
    assert_eq!(inputs(&page2), vec![3, 4]);

    let cursor = page2["events"][1]["timestamp"].as_str().unwrap();
    let enc = cursor.replace('+', "%2B");
    let (_, page3) = get(
        &h,
        &format!("/api/agents/{agent}/events?since={enc}&limit=2"),
    )
    .await;
    assert_eq!(inputs(&page3), vec![5]);

    let cursor = page3["events"][0]["timestamp"].as_str().unwrap();
    let enc = cursor.replace('+', "%2B");
    let (_, empty) = get(&h, &format!("/api/agents/{agent}/events?since={enc}")).await;
    assert!(empty["events"].as_array().unwrap().is_empty());

    // An oversized limit is clamped (to 200), not rejected; only 5 rows exist,
    // so this checks acceptance, not the ceiling itself.
    let (_, capped) = get(
        &h,
        &format!("/api/agents/{agent}/events?since=2000-01-01T00:00:00Z&limit=100000"),
    )
    .await;
    assert_eq!(inputs(&capped), vec![1, 2, 3, 4, 5]);

    // A `since` that is not RFC 3339 is a client error, not an empty list.
    let (status, _) = get(&h, &format!("/api/agents/{agent}/events?since=yesterday")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
