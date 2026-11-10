Record the agent-loop iteration count and the prompt-cache read/write token counts on each `usage_events` row, and return them from `GET /api/agents/{id}/events` together with `channel` and `session_id`.
The columns are added by an idempotent step that runs at the end of every migration pass and does not use a schema version number, so `user_version` stays where it was and the database remains readable by the previous binary.
NULL means "not recorded" (rows written before the columns existed, or paths that do not run the agent loop); 0 in the cache columns can mean the provider does not report cache usage.
The same route accepts `since=<RFC3339>` to return only rows strictly newer than that instant, oldest first and capped by `limit`; without it rows stay newest first. (@jorgeuriarte)
