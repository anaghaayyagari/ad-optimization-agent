# Scaling Note — Running BudgetPilot in the Cloud

To run reliably in the cloud, the same code becomes a scheduled daily job with data checks, retries,
monitoring and hard spend limits.

- **How it runs.** Package the code in a container and run it once a day (e.g. 6 a.m., after ad platforms finalize yesterday's data) with Cloud Scheduler + Cloud Run Jobs or AWS EventBridge + Lambda. Read from the ad platforms' reporting APIs or a warehouse table (BigQuery/Snowflake) instead of a CSV. Start in **recommend-only** mode, where a marketer approves the budgets, then switch to auto-apply via the platforms' budget APIs.
- **Reliability.** Retry API calls 3 times with exponential backoff. Key each run by date so a re-run never applies a change twice. Check the data first: all channels present, no missing days, no impossible values. If a check fails, keep yesterday's budgets and alert. LLM calls get a 30-second timeout and fall back to the rules.
- **Monitoring and logging.** Send `decision_log.jsonl` (budgets + reason per channel) to Cloud Logging/CloudWatch as structured logs. A dashboard tracks daily spend vs. cap, conversions, CPA by channel, shifts made and guardrail hits. Alerts fire on job failure, missing data, spend above the cap, or CPA up more than 30% week over week.
- **Cost controls.** The $1,500/day cap is enforced in code *and* as account-level spending limits on each ad platform, plus a monthly ceiling. LLM use is one call a day capped at 300 output tokens, and compute is one short job a day. A kill-switch flag pauses all changes instantly.
- **Governance.** Guardrail settings live in version control, every change is in the audit log, and the team reviews the agent's decisions weekly.
