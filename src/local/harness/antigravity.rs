//! Google Antigravity harness.
//!
//! Chat: one `agy --output-format stream-json` child per turn. Multi-turn continues
//! via `--conversation <conversation_id>` from the init/result `conversation_id`. Isolated
//! ORX worktrees are the child's current working directory.
//!
//! The playbook is pointed at on the first turn (the file is already in the
//! worktree via [`ensure_playbook`]); session skills land in `.agents/skills`.
//!
//! Headless approval requests are denied; permission choices expose the CLI policy.
//! Bypass explicitly passes `--dangerously-skip-permissions`.
//!
//! Detection: `agy` on PATH or in `~/.local/bin` / `~/.gemini/bin` /
//! `~/.gemini/antigravity-cli/bin`;
//! `agy models` for catalog and authentication verification.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use super::detect::{probe_bin, resolve_symlinks, HarnessAuthState, HarnessInfo, ModelInfo};
use super::options::{
    resolve_reasoning, HarnessOptions, OptionChoice, PermissionMode, PlanActivation,
    REASONING_DEFAULT_ID,
};
use super::{Harness, ResumeAction, TurnFailure, TurnOutcome, TurnResult, TURN_WATCHDOG};
use crate::error::{anyhow, Result};
use crate::local::chat::{
    find_part_mut, harness_log, prepare_env, set_chat_session_env, DeliveryState, PromptAnswer,
    ResumeCtx, TurnCtx, WirePart, WirePrompt, WireToolState,
};
use crate::local::opencode::{ensure_playbook, PLAYBOOK_REL};
use crate::local::shell_env::{find_in_dir, find_on_path};

const AGY_REINSTALL: &str =
    "Reinstall Antigravity CLI via curl -fsSL https://antigravity.google/cli/install.sh | bash";
const MODELS_TIMEOUT: Duration = Duration::from_secs(15);

pub struct Antigravity;

#[async_trait]
impl Harness for Antigravity {
    fn id(&self) -> &'static str {
        "antigravity"
    }

    fn name(&self) -> &'static str {
        "Google Antigravity"
    }

    fn supports_chat(&self) -> bool {
        true
    }

    async fn detect(&self) -> Option<HarnessInfo> {
        let mut info = HarnessInfo::new(self.id(), self.name());
        if let Some(bin) = find_agy() {
            info.record_bin(&bin, probe_bin(&bin).await);
        }
        if info.installed && !info.install_broken {
            if let Some(bin) = info.bin_path.as_deref().map(Path::new) {
                match agy_model_list(bin).await {
                    Ok(models) => {
                        info.authenticated = true;
                        info.auth_state = HarnessAuthState::Ready;
                        info.auth_method = Some("cli");
                        info = info.with_models(models);
                    }
                    Err(error) => {
                        let message = error.to_string();
                        info.auth_state = if message.to_lowercase().contains("sign in") {
                            HarnessAuthState::NeedsLogin
                        } else {
                            HarnessAuthState::Unknown
                        };
                        info.agent_note = Some(message);
                    }
                }
            }
        }
        info.agent_ready = info.ready();
        if info.agent_ready {
            return Some(info);
        } else if info.install_broken {
            info.agent_note = Some(info.broken_note(AGY_REINSTALL));
        } else if info.installed && info.agent_note.is_none() {
            info.agent_note = Some(
                "Sign in by running `agy` in your terminal, then re-check this harness."
                    .to_string(),
            );
        } else if !info.installed {
            info.agent_note = Some(
                "Install Antigravity CLI with `curl -fsSL https://antigravity.google/cli/install.sh | bash`, then sign in with `agy`."
                    .to_string(),
            );
        }
        Some(info)
    }

    async fn run_turn(&self, ctx: &mut TurnCtx) -> TurnResult {
        run_turn(ctx)
            .await
            .map(|()| TurnOutcome::Completed)
            .map_err(|error| TurnFailure::adapter(error, ctx.delivery_state()))
    }

    fn options(&self) -> HarnessOptions {
        HarnessOptions::none()
            .with_permission_choices(
                vec![
                    OptionChoice::described(
                        "default",
                        "Default",
                        "Use Antigravity permission rules; actions needing approval are denied",
                    ),
                    OptionChoice::described(
                        "accept-edits",
                        "Accept edits",
                        "Allow file edits; commands still follow Antigravity permission rules",
                    ),
                    OptionChoice::described(
                        "bypass",
                        "Bypass",
                        "Allow commands and skip tool confirmation prompts",
                    ),
                ],
                "default",
                PlanActivation::Command,
            )
            .with_reasoning_levels(&["low", "medium", "high"])
    }

    async fn resume_from_prompt(
        &self,
        _ctx: &ResumeCtx,
        prompt: &WirePrompt,
        answer: &PromptAnswer,
    ) -> Result<ResumeAction> {
        if prompt.kind != "plan" {
            return Ok(ResumeAction::Nothing);
        }
        if !answer.approve && answer.note.as_deref().is_none_or(|s| s.trim().is_empty()) {
            return Ok(ResumeAction::Nothing);
        }
        Ok(ResumeAction::SendMessage {
            text: super::synthesize_resume("plan", answer).0,
            mode: None,
            plan_mode: Some(!answer.approve),
        })
    }

    fn config_home(&self) -> Option<PathBuf> {
        Some(dirs::home_dir()?.join(".gemini").join("antigravity-cli"))
    }

    fn skill_target(&self) -> Option<PathBuf> {
        Some(
            self.config_home()?
                .join("skills")
                .join("orx")
                .join("SKILL.md"),
        )
    }

    fn skill_shim(&self) -> Option<&'static str> {
        Some(super::CLAUDE_SKILL)
    }

    fn session_skills_dir(&self) -> Option<&'static str> {
        Some(".agents/skills")
    }
}

/// `agy` on PATH, else search common install locations under `~/.local/bin`,
/// `~/.gemini/bin`, or `~/.gemini/antigravity-cli/bin`.
pub(crate) fn find_agy() -> Option<PathBuf> {
    find_on_path("agy")
        .or_else(|| {
            let home = dirs::home_dir()?;
            let local = home.join(".local").join("bin");
            let gemini = home.join(".gemini");
            find_in_dir(&local, "agy")
                .or_else(|| find_in_dir(&gemini.join("bin"), "agy"))
                .or_else(|| find_in_dir(&gemini.join("antigravity-cli").join("bin"), "agy"))
        })
        .or_else(|| find_in_dir(&dirs::data_local_dir()?.join("agy").join("bin"), "agy"))
        .map(resolve_symlinks)
}

async fn agy_model_list(bin: &Path) -> Result<Vec<ModelInfo>> {
    let mut cmd = Command::new(bin);
    cmd.arg("models")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    prepare_env(&mut cmd);
    cmd.env("NO_COLOR", "1");
    let out = tokio::time::timeout(MODELS_TIMEOUT, cmd.output())
        .await
        .map_err(|_| {
            anyhow!("Antigravity model discovery timed out. Re-check when connected.")
        })??;
    if !out.status.success() {
        let error = String::from_utf8_lossy(&out.stderr);
        return Err(anyhow!(
            "Antigravity model discovery failed: {}",
            error.trim()
        ));
    }
    let models = parse_agy_model_list(&String::from_utf8_lossy(&out.stdout));
    if models.is_empty() {
        return Err(anyhow!(
            "Antigravity returned no available models. Re-check your account."
        ));
    }
    Ok(models)
}

/// Parse the whitespace-separated model catalog, ignoring
/// informational banner lines such as `Fetching available models...`.
fn parse_agy_model_list(text: &str) -> Vec<ModelInfo> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty()
                || line.starts_with("Fetching")
                || line.starts_with("Available")
                || line.starts_with("Listing")
            {
                return None;
            }
            let (id, label) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
            let id = id.trim();
            let label = label.trim();
            if id.is_empty()
                || id == REASONING_DEFAULT_ID
                || !id.chars().all(|c| {
                    c.is_ascii_alphanumeric()
                        || matches!(c, '-' | '_' | '.' | '[' | ']' | '=' | ',')
                })
            {
                return None;
            }
            Some(ModelInfo::new(id).with_label((!label.is_empty()).then_some(label), None))
        })
        .collect()
}

fn first_turn_prompt(text: &str) -> String {
    format!(
        "Read and follow `{PLAYBOOK_REL}` before acting. It is the OpenResearch session playbook for this worktree.\n\n{text}"
    )
}

async fn run_turn(ctx: &mut TurnCtx) -> Result<()> {
    let bin = find_agy().ok_or_else(|| {
        anyhow!("agy not found on PATH — install Antigravity CLI and sign in first")
    })?;
    let project = ctx.project.clone();
    let session_id = ctx.session_id.clone();
    let skills_dir = Antigravity.session_skills_dir();
    let (repo, _playbook) =
        tokio::task::spawn_blocking(move || ensure_playbook(&project, &session_id, skills_dir))
            .await
            .map_err(|e| anyhow!("playbook task failed: {e}"))??;

    let resume = ctx.native_session_id.clone();
    let mut prompt = ctx.text.clone();
    if resume.is_none() {
        prompt = first_turn_prompt(&prompt);
    }

    let mut cmd = Command::new(&bin);
    cmd.args(["--output-format", "stream-json"]);

    if let Some(model) = ctx.model.as_deref().filter(|model| !model.is_empty()) {
        cmd.args(["--model", model]);
    }

    if let Some(effort) =
        resolve_reasoning(ctx.reasoning_level.as_deref(), &["low", "medium", "high"])
    {
        cmd.args(["--effort", effort]);
    }
    cmd.args(permission_args(ctx.permission_mode, ctx.plan_mode));
    cmd.args(["--print-timeout", "30m"]);

    if let Some(native_id) = &resume {
        cmd.args(["--conversation", native_id]);
    }

    cmd.arg(format!("--print={prompt}"));
    cmd.current_dir(&repo);

    let log_name = format!("antigravity-{}", uuid::Uuid::new_v4());
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(harness_log(&log_name)?))
        .kill_on_drop(true);

    prepare_env(&mut cmd);
    cmd.env("NO_COLOR", "1");
    set_chat_session_env(&mut cmd, &ctx.session_id, "antigravity", ctx.host.up_port());

    ctx.persist_delivery(DeliveryState::Unknown)?;
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(error) => {
            ctx.mark_delivery(DeliveryState::NotSent);
            return Err(anyhow!("Could not spawn {}: {}", bin.display(), error));
        }
    };
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let mut lines = BufReader::new(stdout).lines();
    let mut state = TurnState::default();

    loop {
        match tokio::time::timeout(TURN_WATCHDOG, lines.next_line()).await {
            Ok(Ok(Some(line))) => {
                let Ok(event) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                if matches!(
                    event.get("event").and_then(Value::as_str),
                    Some("step_update")
                ) {
                    ctx.mark_delivery(DeliveryState::Accepted);
                }
                let terminal = apply_event(ctx, &mut state, &event);
                if let Some(sid) = state.conversation_id.as_deref() {
                    ctx.set_native_session_id(sid);
                }
                ctx.maybe_flush();
                if terminal {
                    break;
                }
            }
            Ok(Ok(None)) => break,
            Ok(Err(error)) => {
                return Err(anyhow!("antigravity stdout: {error}"));
            }
            Err(_) => {
                return Err(anyhow!(
                    "Antigravity went silent for {} minutes and was interrupted.",
                    TURN_WATCHDOG.as_secs() / 60
                ));
            }
        }
    }

    let status = tokio::time::timeout(TURN_WATCHDOG, child.wait())
        .await
        .map_err(|_| anyhow!("Antigravity did not exit after its response"))??;
    let log_path = crate::store::data_dir().join(format!("agent-{log_name}.log"));
    if !state.saw_result {
        return Err(anyhow!(
            "Antigravity ended without a result ({status}); see {}",
            log_path.display()
        ));
    }
    if !status.success() && !state.turn_errored {
        return Err(anyhow!("Antigravity ended with error ({status})"));
    }
    if ctx.plan_mode && !state.turn_errored {
        if let Some(card) = plan_card(&ctx.assistant.parts, &ctx.assistant.id) {
            ctx.upsert_part(card);
        }
    }
    if !state.turn_errored {
        let _ = std::fs::remove_file(log_path);
    }
    Ok(())
}

fn permission_args(mode: Option<PermissionMode>, plan: bool) -> Vec<&'static str> {
    let mut args = if plan {
        vec!["--mode=plan"]
    } else if matches!(
        mode,
        Some(PermissionMode::AcceptEdits | PermissionMode::Bypass)
    ) {
        vec!["--mode=accept-edits"]
    } else {
        Vec::new()
    };
    if mode == Some(PermissionMode::Bypass) && !plan {
        args.push("--dangerously-skip-permissions");
    }
    args
}

fn normalize_tool<'a>(name: &'a str, params: Option<&Value>) -> (&'a str, Option<Value>) {
    let (tool, aliases): (&str, &[(&str, &str)]) = match name {
        "run_command" => ("Bash", &[("CommandLine", "command")]),
        "view_file" => ("Read", &[("AbsolutePath", "file_path")]),
        "write_to_file" => (
            "Write",
            &[("TargetFile", "file_path"), ("CodeContent", "content")],
        ),
        "replace_file_content" | "multi_replace_file_content" => {
            ("Edit", &[("TargetFile", "file_path")])
        }
        "list_dir" => ("Glob", &[("DirectoryPath", "path")]),
        "grep_search" | "code_search" => ("Grep", &[("Query", "pattern"), ("SearchPath", "path")]),
        "find_by_name" => (
            "Glob",
            &[("Pattern", "pattern"), ("SearchDirectory", "path")],
        ),
        "read_url_content" => ("WebFetch", &[("Url", "url")]),
        "search_web" => ("WebSearch", &[]),
        _ => (name, &[]),
    };
    let mut input = params.cloned();
    if let Some(object) = input.as_mut().and_then(Value::as_object_mut) {
        for &(native, normalized) in aliases {
            if let Some(value) = object.get(native).cloned() {
                object.insert(normalized.into(), value);
            }
        }
    }
    (tool, input)
}

fn error_text(value: &Value) -> Option<&str> {
    value.as_str().or_else(|| value.get("message")?.as_str())
}

fn step_is_terminal(state: &str) -> bool {
    matches!(state, "DONE" | "ERROR" | "CANCELED")
}

fn denied_actions_error(result: &Value) -> Option<String> {
    if result
        .get("response")
        .and_then(Value::as_str)
        .is_some_and(|response| !response.trim().is_empty())
    {
        return None;
    }
    let actions = result.get("denied_actions")?.as_array()?;
    if actions.is_empty() {
        return None;
    }
    let names = actions
        .iter()
        .filter_map(|action| {
            action
                .get("display_name")
                .or_else(|| action.get("action"))
                .and_then(Value::as_str)
        })
        .collect::<Vec<_>>();
    Some(if names.is_empty() {
        "Antigravity denied one or more required actions".into()
    } else {
        format!(
            "Antigravity denied required action(s): {}",
            names.join(", ")
        )
    })
}

#[derive(Default)]
struct TurnState {
    conversation_id: Option<String>,
    text_part_id: Option<String>,
    text_seq: usize,
    saw_result: bool,
    turn_errored: bool,
}

fn apply_event(ctx: &mut TurnCtx, state: &mut TurnState, event: &Value) -> bool {
    let event_type = event.get("event").and_then(Value::as_str).unwrap_or("");
    match event_type {
        "init" => {
            if let Some(cid) = event
                .get("conversation_id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            {
                state.conversation_id = Some(cid.to_string());
            }
            false
        }
        "step_update" => {
            if let Some(step) = event.get("step_update") {
                if let Some(cid) = step
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                {
                    state.conversation_id = Some(cid.to_string());
                }
                let step_type = step.get("step_type").and_then(Value::as_str).unwrap_or("");
                let step_state = step.get("state").and_then(Value::as_str).unwrap_or("");

                match step_type {
                    "agent_response" => {
                        if let Some(delta) = step.get("text_delta").and_then(Value::as_str) {
                            if !delta.is_empty() {
                                let id = match &state.text_part_id {
                                    Some(id) => id.clone(),
                                    None => {
                                        state.text_seq += 1;
                                        let id = format!("text-{}", state.text_seq);
                                        ctx.upsert_part(WirePart::text(id.clone(), ""));
                                        state.text_part_id = Some(id.clone());
                                        id
                                    }
                                };
                                ctx.append_part_text(&id, delta);
                            }
                        }
                        if step_is_terminal(step_state) {
                            state.text_part_id = None;
                        }
                    }
                    "tool" => {
                        state.text_part_id = None;
                        let tool_name = step
                            .get("tool_name")
                            .and_then(Value::as_str)
                            .unwrap_or("tool");
                        let tool_info = step.get("tool_info").unwrap_or(&Value::Null);
                        let step_index =
                            step.get("step_index").and_then(Value::as_i64).unwrap_or(0);
                        let call_id = format!("tool-{step_index}-{tool_name}");

                        let (tool, params) = normalize_tool(tool_name, tool_info.get("parameters"));
                        let output = tool_info.get("output").and_then(Value::as_str);
                        let error = tool_info
                            .get("error")
                            .and_then(error_text)
                            .map(str::to_string)
                            .or_else(|| match step_state {
                                "ERROR" => Some("Antigravity tool failed".into()),
                                "CANCELED" => Some("Antigravity tool was canceled".into()),
                                _ => None,
                            });
                        let is_terminal = step_is_terminal(step_state) || error.is_some();
                        let is_failed =
                            matches!(step_state, "ERROR" | "CANCELED") || error.is_some();

                        if let Some(part) = find_part_mut(&mut ctx.assistant.parts, &call_id) {
                            if let Some(part_state) = part.state.as_mut() {
                                if params.is_some() {
                                    part_state.input = params;
                                }
                                if is_terminal {
                                    part_state.status =
                                        if is_failed { "error" } else { "completed" }.into();
                                    if let Some(out) = output {
                                        part_state.output = Some(out.to_string());
                                    }
                                    if let Some(err) = &error {
                                        part_state.error = Some(err.clone());
                                    }
                                }
                            }
                        } else {
                            let status = if is_terminal {
                                if is_failed {
                                    "error"
                                } else {
                                    "completed"
                                }
                            } else {
                                "running"
                            };
                            ctx.upsert_part(WirePart {
                                id: call_id,
                                kind: "tool".into(),
                                text: None,
                                tool: Some(tool.to_string()),
                                state: Some(WireToolState {
                                    status: status.into(),
                                    input: params,
                                    output: output.map(str::to_string),
                                    error,
                                    title: None,
                                }),
                                prompt: None,
                                phase: None,
                                children: Vec::new(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            false
        }
        "result" => {
            state.saw_result = true;
            let res = event.get("result").unwrap_or(&Value::Null);
            if let Some(cid) = res
                .get("conversation_id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            {
                state.conversation_id = Some(cid.to_string());
            }
            let status = res
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("INVALID");
            if status != "SUCCESS" {
                state.turn_errored = true;
                let error = res.get("error").and_then(error_text).unwrap_or(status);
                ctx.mark_terminal_failure("antigravity_terminal", format!("Antigravity: {error}"));
            } else if let Some(error) = denied_actions_error(res) {
                ctx.mark_delivery(DeliveryState::Accepted);
                state.turn_errored = true;
                ctx.mark_terminal_failure("antigravity_permission_denied", error);
            } else {
                ctx.mark_delivery(DeliveryState::Accepted);
                ctx.mark_final_text_tail();
            }
            true
        }
        _ => false,
    }
}

fn plan_card(parts: &[WirePart], assistant_id: &str) -> Option<WirePart> {
    let last_text = parts.iter().rev().find_map(|part| {
        (part.kind == "text")
            .then_some(part.text.as_deref())
            .flatten()
            .filter(|text| !text.trim().is_empty())
    })?;
    if !super::should_synthesize_plan(true, false, false, last_text) {
        return None;
    }
    Some(WirePart::prompt(
        format!("plan-synth-{assistant_id}"),
        WirePrompt {
            kind: "plan".into(),
            plan: Some(last_text.to_string()),
            synthesized: true,
            ..Default::default()
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fold(events: &[Value]) -> (TurnCtx, TurnState) {
        let mut ctx = TurnCtx::test_stub();
        let mut state = TurnState::default();
        for event in events {
            apply_event(&mut ctx, &mut state, event);
        }
        (ctx, state)
    }

    #[test]
    fn stream_folds_init_text_tools_and_result() {
        let (mut ctx, mut state) = fold(&[
            json!({
                "event": "init",
                "conversation_id": "test-conv-123",
                "init": {"tools": ["run_command", "view_file"]}
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 0,
                    "state": "DONE",
                    "step_type": "user_input"
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "state": "ACTIVE",
                    "step_type": "agent_response",
                    "text_delta": "Checking directory "
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "state": "DONE",
                    "step_type": "agent_response",
                    "text_delta": "contents..."
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 2,
                    "state": "ACTIVE",
                    "step_type": "tool",
                    "tool_name": "run_command",
                    "tool_info": {
                        "name": "run_command",
                        "parameters": {"CommandLine": "ls -la"}
                    }
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 2,
                    "state": "DONE",
                    "step_type": "tool",
                    "tool_name": "run_command",
                    "tool_info": {
                        "name": "run_command",
                        "parameters": {"CommandLine": "ls -la"},
                        "output": "file.txt\n"
                    }
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 3,
                    "state": "ACTIVE",
                    "step_type": "agent_response",
                    "text_delta": "Found file.txt"
                }
            }),
            json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 3,
                    "state": "DONE",
                    "step_type": "agent_response",
                    "text_delta": "."
                }
            }),
        ]);

        assert_eq!(state.conversation_id.as_deref(), Some("test-conv-123"));
        assert_eq!(ctx.assistant.parts.len(), 3);
        assert_eq!(ctx.assistant.parts[0].kind, "text");
        assert_eq!(
            ctx.assistant.parts[0].text.as_deref(),
            Some("Checking directory contents...")
        );

        assert_eq!(ctx.assistant.parts[1].kind, "tool");
        let tool_state = ctx.assistant.parts[1].state.as_ref().unwrap();
        assert_eq!(tool_state.status, "completed");
        assert_eq!(ctx.assistant.parts[1].tool.as_deref(), Some("Bash"));
        assert_eq!(tool_state.input.as_ref().unwrap()["command"], "ls -la");
        assert_eq!(tool_state.output.as_deref(), Some("file.txt\n"));

        assert_eq!(ctx.assistant.parts[2].kind, "text");
        assert_eq!(
            ctx.assistant.parts[2].text.as_deref(),
            Some("Found file.txt.")
        );

        let done = apply_event(
            &mut ctx,
            &mut state,
            &json!({
                "event": "result",
                "result": {
                    "conversation_id": "test-conv-123",
                    "status": "SUCCESS",
                    "response": "Done"
                }
            }),
        );
        assert!(done);
        assert!(state.saw_result);
        assert!(!state.turn_errored);
    }

    #[test]
    fn non_success_results_never_finish_the_answer() {
        for status in [
            "ERROR",
            "CANCELED",
            "INTERRUPTED",
            "INVALID",
            "WAITING",
            "RUNNING",
        ] {
            let (ctx, state) = fold(&[json!({
                "event": "result",
                "result": {"conversation_id": "", "status": status, "error": "Quota limit exceeded"}
            })]);
            assert!(state.saw_result);
            assert!(state.turn_errored, "{status}");
            assert!(state.conversation_id.is_none());
            assert!(ctx.assistant.parts.is_empty());
        }
    }

    #[test]
    fn tool_updates_preserve_inputs_and_surface_object_errors() {
        let (ctx, _) = fold(&[
            json!({"event":"step_update","step_update":{"step_index":1,"state":"ACTIVE","step_type":"tool","tool_name":"view_file","tool_info":{"parameters":{"AbsolutePath":"/repo/file.rs"}}}}),
            json!({"event":"step_update","step_update":{"step_index":1,"state":"DONE","step_type":"tool","tool_name":"view_file","tool_info":{"error":{"type":"permission","message":"Denied"}}}}),
        ]);
        let part = &ctx.assistant.parts[0];
        assert_eq!(part.tool.as_deref(), Some("Read"));
        let state = part.state.as_ref().unwrap();
        assert_eq!(state.status, "error");
        assert_eq!(state.error.as_deref(), Some("Denied"));
        assert_eq!(state.input.as_ref().unwrap()["file_path"], "/repo/file.rs");
        for name in [
            "write_to_file",
            "replace_file_content",
            "multi_replace_file_content",
        ] {
            let (_, input) = normalize_tool(name, Some(&json!({"TargetFile":"/repo/file.rs"})));
            assert_eq!(input.unwrap()["file_path"], "/repo/file.rs");
        }
    }

    #[test]
    fn failed_tool_states_are_terminal_even_without_error_payloads() {
        for (step_state, expected_error) in [
            ("ERROR", "Antigravity tool failed"),
            ("CANCELED", "Antigravity tool was canceled"),
        ] {
            let (ctx, _) = fold(&[json!({
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "state": step_state,
                    "step_type": "tool",
                    "tool_name": "run_command",
                    "tool_info": {"parameters": {"CommandLine": "false"}}
                }
            })]);
            let tool = ctx.assistant.parts[0].state.as_ref().unwrap();
            assert_eq!(tool.status, "error", "{step_state}");
            assert_eq!(tool.error.as_deref(), Some(expected_error), "{step_state}");
        }
    }

    #[test]
    fn an_error_payload_terminates_an_active_tool() {
        let (ctx, _) = fold(&[json!({
            "event": "step_update",
            "step_update": {
                "step_index": 1,
                "state": "ACTIVE",
                "step_type": "tool",
                "tool_name": "view_file",
                "tool_info": {"error": {"message": "Denied"}}
            }
        })]);
        let tool = ctx.assistant.parts[0].state.as_ref().unwrap();
        assert_eq!(tool.status, "error");
        assert_eq!(tool.error.as_deref(), Some("Denied"));
    }

    #[test]
    fn terminal_agent_response_clears_the_streamed_text_part() {
        for step_state in ["ERROR", "CANCELED"] {
            let (_, state) = fold(&[
                json!({"event":"step_update","step_update":{"step_index":1,"state":"ACTIVE","step_type":"agent_response","text_delta":"partial"}}),
                json!({"event":"step_update","step_update":{"step_index":1,"state":step_state,"step_type":"agent_response"}}),
            ]);
            assert!(state.text_part_id.is_none(), "{step_state}");
        }
    }

    #[test]
    fn empty_success_with_denied_actions_is_a_failed_turn() {
        let (ctx, state) = fold(&[json!({
            "event": "result",
            "result": {
                "status": "SUCCESS",
                "response": "",
                "denied_actions": [{"action": "command", "display_name": "RunCommand"}]
            }
        })]);
        assert!(state.saw_result);
        assert!(state.turn_errored);
        assert_eq!(ctx.delivery_state(), DeliveryState::Accepted);
        assert_eq!(
            denied_actions_error(&json!({
                "response": "",
                "denied_actions": [{"display_name": "RunCommand"}]
            }))
            .as_deref(),
            Some("Antigravity denied required action(s): RunCommand")
        );
    }

    #[test]
    fn denied_actions_do_not_discard_a_nonempty_response() {
        assert!(denied_actions_error(&json!({
            "response": "I could not run it, but here is an explanation.",
            "denied_actions": [{"display_name": "RunCommand"}]
        }))
        .is_none());
    }

    #[test]
    fn permissions_match_advertised_native_modes() {
        assert!(permission_args(None, false).is_empty());
        assert!(permission_args(Some(PermissionMode::Ask), false).is_empty());
        assert_eq!(
            permission_args(Some(PermissionMode::AcceptEdits), false),
            ["--mode=accept-edits"]
        );
        assert_eq!(
            permission_args(Some(PermissionMode::Bypass), false),
            ["--mode=accept-edits", "--dangerously-skip-permissions"]
        );
        assert_eq!(
            permission_args(Some(PermissionMode::Bypass), true),
            ["--mode=plan"]
        );
    }

    #[test]
    fn parses_agy_model_list_output() {
        let sample = "Fetching available models...\n\
                      gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n\
                      gemini-3.1-pro-high    Gemini 3.1 Pro (High)\n\
                      claude-sonnet-4-6\tClaude Sonnet 4.6 (Thinking)\n";
        let models = parse_agy_model_list(sample);
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].id, "gemini-3.8-flash-high");
        assert_eq!(
            models[0].display_name.as_deref(),
            Some("Gemini 3.8 Flash (High)")
        );
        assert_eq!(models[1].id, "gemini-3.1-pro-high");
        assert_eq!(models[2].id, "claude-sonnet-4-6");
    }
}
