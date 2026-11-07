//! Time a tool call spends queued is not time the tool spent running.
//!
//! The agent loop bounds every tool call with `tool_timeout_secs`. For an MCP call that clock used to start before the call had even got its turn on the server's connection, so a tool that needs 4 s with a 6 s limit was cut off when it queued behind another 4 s call. This module lets the dispatcher mark the stretches it spends *waiting for a slot* and lets the timeout leave them out.
//!
//! Mechanism: [`timeout_excluding_waits`] runs the future under a [`ToolWait`] stored in a task-local. Code inside the future calls [`begin_wait`] around the queueing and the returned guard records how long it lasted. The timeout's deadline is `start + budget + time spent waiting`, recomputed whenever the timer fires, so a wait in progress also holds the deadline back. Code outside such a scope (tests, other callers) sees no task-local and behaves as before.
//!
//! The wait itself must be bounded by whoever waits (the MCP dispatcher uses the server's `timeout_secs`), otherwise a stuck queue would never time out.

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
// The tokio clock, not `std::time::Instant`: the deadline is handed to a tokio timer, and the two clocks only agree until a test pauses tokio's.
use tokio::time::Instant;

/// Accumulated time spent waiting inside one tool call.
#[derive(Default)]
pub struct ToolWait {
    /// Completed waits, in nanoseconds.
    waited_ns: AtomicU64,
    /// Start of the wait in progress, if any.
    waiting_since: Mutex<Option<Instant>>,
}

impl ToolWait {
    /// Total time spent waiting so far, including a wait still in progress.
    pub fn total(&self) -> Duration {
        let done = Duration::from_nanos(self.waited_ns.load(Ordering::Acquire));
        let current = self
            .waiting_since
            .lock()
            .ok()
            .and_then(|g| g.map(|t| t.elapsed()))
            .unwrap_or_default();
        done + current
    }
}

/// Marks a wait in progress; dropping it adds the elapsed time to the call's total.
pub struct WaitGuard {
    wait: Arc<ToolWait>,
    started: Instant,
}

impl WaitGuard {
    /// How long this wait has lasted so far.
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

impl Drop for WaitGuard {
    fn drop(&mut self) {
        // Order matters for `total()`: add to the completed total first, then clear the in-progress marker, so a concurrent reader may briefly count the wait twice (a longer deadline, never a spurious timeout) but never zero times.
        self.wait
            .waited_ns
            .fetch_add(self.started.elapsed().as_nanos() as u64, Ordering::AcqRel);
        if let Ok(mut g) = self.wait.waiting_since.lock() {
            *g = None;
        }
    }
}

tokio::task_local! {
    static CURRENT: Arc<ToolWait>;
}

/// Start a wait for the current tool call. `None` outside a [`timeout_excluding_waits`] scope.
pub fn begin_wait() -> Option<WaitGuard> {
    CURRENT
        .try_with(|w| {
            if let Ok(mut g) = w.waiting_since.lock() {
                *g = Some(Instant::now());
            }
            WaitGuard {
                wait: Arc::clone(w),
                started: Instant::now(),
            }
        })
        .ok()
}

/// Like `tokio::time::timeout(budget, fut)`, but time `fut` spends inside [`begin_wait`] guards does not count against `budget`.
///
/// A `budget` too large for the clock to represent (`u64::MAX` seconds as "no timeout") means no deadline, as with `tokio::time::timeout`, rather than an overflow panic.
pub async fn timeout_excluding_waits<F: Future>(
    budget: Duration,
    fut: F,
) -> Result<F::Output, tokio::time::error::Elapsed> {
    let wait = Arc::new(ToolWait::default());
    let started = Instant::now();
    let scoped = CURRENT.scope(Arc::clone(&wait), fut);
    tokio::pin!(scoped);
    // `None`: the deadline lies beyond what the clock can represent.
    let deadline = |wait: &ToolWait| started.checked_add(budget.saturating_add(wait.total()));
    loop {
        let Some(due) = deadline(&wait) else {
            return Ok(scoped.await);
        };
        tokio::select! {
            out = &mut scoped => return Ok(out),
            _ = tokio::time::sleep_until(due) => {
                // Waits may have grown while the timer ran: only expire if the deadline, recomputed, has really passed.
                if deadline(&wait).is_some_and(|d| Instant::now() >= d) {
                    // Same error type as `tokio::time::timeout`, so callers keep their `Err(_)` arm.
                    return Err(elapsed().await);
                }
            }
        }
    }
}

/// A `tokio::time::error::Elapsed` (the type has no public constructor).
async fn elapsed() -> tokio::time::error::Elapsed {
    match tokio::time::timeout(Duration::ZERO, std::future::pending::<()>()).await {
        Err(e) => e,
        Ok(()) => unreachable!("a pending future never completes"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn work(wait: Duration, run: Duration) {
        let g = begin_wait();
        tokio::time::sleep(wait).await;
        drop(g);
        tokio::time::sleep(run).await;
    }

    #[tokio::test]
    async fn waiting_is_not_charged_to_the_budget() {
        // 400 ms queued + 200 ms running under a 500 ms budget: 600 ms of wall clock, 200 ms of tool time.
        let r = timeout_excluding_waits(
            Duration::from_millis(500),
            work(Duration::from_millis(400), Duration::from_millis(200)),
        )
        .await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn running_longer_than_the_budget_still_times_out() {
        let t = Instant::now();
        let r = timeout_excluding_waits(
            Duration::from_millis(300),
            work(Duration::from_millis(200), Duration::from_millis(2000)),
        )
        .await;
        assert!(r.is_err());
        // 200 ms waited + 300 ms budget: expires near 500 ms, not at 300 and not at 2200.
        let e = t.elapsed();
        assert!(
            e >= Duration::from_millis(480) && e < Duration::from_millis(1200),
            "{e:?}"
        );
    }

    #[tokio::test]
    async fn without_a_wait_it_behaves_like_a_plain_timeout() {
        let t = Instant::now();
        let r = timeout_excluding_waits(
            Duration::from_millis(200),
            tokio::time::sleep(Duration::from_secs(5)),
        )
        .await;
        assert!(r.is_err());
        assert!(t.elapsed() < Duration::from_millis(900));
    }

    #[tokio::test]
    async fn a_wait_in_progress_holds_the_deadline_back() {
        // Still queued when the nominal budget runs out: must not expire while waiting.
        let r = timeout_excluding_waits(
            Duration::from_millis(150),
            work(Duration::from_millis(500), Duration::from_millis(50)),
        )
        .await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn begin_wait_outside_a_scope_is_none() {
        assert!(begin_wait().is_none());
    }
}
