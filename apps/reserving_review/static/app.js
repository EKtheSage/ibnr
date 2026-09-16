"use strict";

(() => {
  const SESSION_KEY = "reserve-review-authorization";
  const $ = (id) => document.getElementById(id);
  const state = { auth: null, me: null, demo: false, runs: [], run: null, csv: "", busy: false, exportRunId: null };
  const amountFormat = new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const numberFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 6 });
  const statusNames = { DRAFT: "Draft", SUBMITTED: "In review", APPROVED: "Approved", REJECTED: "Rejected", UNVERIFIABLE: "Cannot verify" };
  const actionNames = { create: "Analysis created", analyze: "Analysis created", override: "Reserve adjusted", submit: "Submitted for review", approve: "Approved", reject: "Rejected", revise: "Revision created", export: "Record exported" };
  const labels = {
    candidate: "Candidate", eligible: "Eligible", mean_rmse: "Mean RMSE", n_scored: "Scored intervals", n_required: "Required intervals",
    reason: "Reason", origin_period: "Origin period", latest: "Latest observed", ultimate: "Modeled ultimate", reserve: "Modeled reserve",
    booked: "Booked reserve", adjustment: "Adjustment", actor: "Adjusted by", as_of: "Information date", eval_date: "Outcome date",
    metric: "Metric", rmse: "RMSE", n_cells: "Origins", weight_sum: "Actual movement weight", status: "Status",
    dev_lag: "Age · months", next_dev_lag: "Next age · months", factor: "Factor", selected: "Selected", ratio: "Observed ratio",
    method: "Method", history_periods: "History periods", average: "Average", drop_high: "Drop highest", drop_low: "Drop lowest",
    expected_loss_ratio: "Prior loss ratio", decay: "Decay", horizon: "Horizon · months", unsupported_factor: "Unsupported factors",
    exhausted_exclusions: "Exhausted exclusions", exclude: "Explicit exclusions", n_pairs: "Observation pairs", n_selected: "Selected pairs",
  };

  function node(tag, text, className) {
    const result = document.createElement(tag);
    if (text !== undefined && text !== null) result.textContent = String(text);
    if (className) result.className = className;
    return result;
  }

  function display(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "number") return Number.isFinite(value) ? numberFormat.format(value) : "Undefined";
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function amount(value) {
    return typeof value === "number" && Number.isFinite(value) ? amountFormat.format(value) : "—";
  }

  function timestamp(value) {
    if (!value) return "—";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  }

  function label(key) { return labels[key] || key.replaceAll("_", " "); }
  function settingValue(key, value) {
    if (key === "history_periods" && value === null) return "All available";
    if (key === "exclude" && Array.isArray(value) && value.length === 0) return "None";
    const names = {
      method: { cl: "Chain ladder (CL)", bf: "Bornhuetter–Ferguson (BF)", gcc: "Generalized Cape Cod (GCC)" },
      unsupported_factor: { raise: "Require estimable factors", unity: "Use factor 1 when unsupported" },
      exhausted_exclusions: { raise: "Stop if extreme exclusions exhaust observations", keep: "Retain observations if extreme exclusions exhaust them" },
    };
    return names[key]?.[value] ?? value;
  }
  function hasRole(role) { return Boolean(state.me?.roles?.includes(role)); }
  function ownRun() { return state.run?.created_by === state.me?.name; }
  function runPath(suffix = "") { return `/api/runs/${encodeURIComponent(state.run.id)}${suffix}`; }

  function notice(message, error = false) {
    $("notice").textContent = message;
    $("notice").className = error ? "notice error" : "notice";
    $("notice").setAttribute("role", error ? "alert" : "status");
    $("notice").hidden = !message;
  }

  function busy(value, message = "") {
    state.busy = value;
    $("main").setAttribute("aria-busy", String(value));
    $("busy-message").hidden = !value;
    $("busy-message").textContent = message;
    document.querySelectorAll("button, input, select, textarea").forEach((control) => { control.disabled = value; });
  }

  async function operate(message, work) {
    if (state.busy) return;
    notice("");
    busy(true, message);
    try { await work(); }
    catch (error) { notice(error.message || "The action could not be completed.", true); }
    finally { busy(false); }
  }

  function clearSession() {
    try { sessionStorage.removeItem(SESSION_KEY); } catch { /* No other persistence is used. */ }
    state.auth = null;
    state.me = null;
    state.run = null;
    state.runs = [];
    state.csv = "";
    clearExportPreview();
    $("login-form").reset();
    $("analysis-form").reset();
    updateHorizonGrain();
    $("csv-status").textContent = "No data loaded.";
    $("run-list").replaceChildren();
    $("login-view").hidden = false;
    $("workspace").hidden = true;
    $("identity").hidden = true;
    $("logout").hidden = true;
  }

  async function api(path, { method = "GET", body, authenticated = true, download = false } = {}) {
    const headers = { Accept: "application/json" };
    if (authenticated && state.auth) headers.Authorization = state.auth;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    let response;
    try {
      response = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), credentials: "omit", cache: "no-store" });
    } catch {
      throw new Error("The server could not be reached. If you were saving a change, refresh the run before trying again.");
    }
    if (!response.ok) {
      const result = await response.json().catch(() => ({}));
      if (response.status === 401 && authenticated) clearSession();
      const message = typeof result.error === "string" ? result.error : `Request failed (${response.status}).`;
      throw new Error(response.status === 409 ? `${message} Refresh the run to inspect the latest revision before trying again.` : message);
    }
    return download ? response.blob() : response.json();
  }

  function showWorkspace() {
    $("login-view").hidden = true;
    $("workspace").hidden = false;
    $("identity").hidden = false;
    $("identity").textContent = `${state.me.name} · ${state.me.roles.join(" / ")}`;
    $("logout").hidden = false;
    $("new-analysis").hidden = !hasRole("analyst");
    $("load-demo").hidden = !hasRole("analyst");
  }

  function showPage(page) {
    for (const name of ["create", "run", "empty"]) $(`${name}-view`).hidden = name !== page;
  }

  function newAnalysis(focus = true) {
    state.run = null;
    clearExportPreview();
    showPage(hasRole("analyst") ? "create" : "empty");
    renderRuns();
    if (focus && hasRole("analyst")) $("create-title").focus();
  }

  function statusBadge(status) {
    return node("span", statusNames[status] || status, `status ${Object.hasOwn(statusNames, status) ? status.toLowerCase() : ""}`);
  }

  function renderRuns() {
    $("run-count").textContent = state.runs.length;
    $("run-list").replaceChildren();
    if (!state.runs.length) $("run-list").append(node("p", "No analyses in this workspace yet.", "empty-message"));
    for (const run of state.runs) {
      const active = state.run?.id === run.id;
      const damaged = run.status === "UNVERIFIABLE";
      const button = node("button", null, `run-item${active ? " active" : ""}`);
      button.type = "button";
      if (active) button.setAttribute("aria-current", "true");
      button.append(node("span", damaged ? "Record that failed verification" : run.title || run.snapshot?.title || "Untitled analysis", "run-item-title"));
      const meta = node("span", null, "run-item-meta");
      meta.append(node("span", damaged ? run.id : run.as_of || run.snapshot?.as_of || "—"), statusBadge(run.status));
      // An approved run that already has a revision is not the current position.
      if (run.superseded_by) meta.append(node("span", "Superseded", "status superseded"));
      button.append(meta);
      button.addEventListener("click", () => operate("Loading analysis…", () => openRun(run.id)));
      $("run-list").append(button);
    }
  }

  async function loadRuns() {
    state.runs = await api("/api/runs");
    renderRuns();
  }

  async function openRun(id, focus = true) {
    state.run = await api(`/api/runs/${encodeURIComponent(id)}`);
    renderRun();
    renderRuns();
    if (focus) $("run-title").focus();
  }

  function renderTable(id, rows, { columns, caption, highlight, money = [], pageSize = 50 } = {}) {
    const destination = $(id);
    destination.replaceChildren();
    if (!Array.isArray(rows) || !rows.length) {
      destination.append(node("p", "No rows recorded.", "empty-message"));
      return;
    }
    const keys = columns || [...new Set(rows.flatMap((row) => Object.keys(row)))];
    let expanded = false;
    const wrap = node("div", null, "table-wrap");
    wrap.tabIndex = 0;
    wrap.setAttribute("role", "region");
    wrap.setAttribute("aria-label", caption || "Evidence table");
    const table = node("table", null, "data-table");
    if (caption) table.append(node("caption", caption));
    const head = node("thead");
    const header = node("tr");
    for (const key of keys) {
      const th = node("th", label(key));
      th.scope = "col";
      if (money.includes(key) || rows.some((row) => typeof row[key] === "number")) th.classList.add("numeric");
      header.append(th);
    }
    head.append(header);
    table.append(head);
    const tbody = node("tbody");
    table.append(tbody);
    wrap.append(table);
    destination.append(wrap);
    const footer = node("div", null, "table-footer");
    const count = node("span");
    footer.append(count);
    const toggle = node("button", "Show all rows", "button quiet compact");
    toggle.type = "button";
    if (rows.length > pageSize) footer.append(toggle);
    destination.append(footer);
    function draw() {
      tbody.replaceChildren();
      const visible = expanded ? rows : rows.slice(0, pageSize);
      for (const row of visible) {
        const tr = node("tr");
        if (highlight && highlight(row)) tr.classList.add("selected-row");
        for (const key of keys) {
          const cell = node("td", money.includes(key) ? amount(row[key]) : display(row[key]));
          if (typeof row[key] === "number" || money.includes(key)) cell.classList.add("numeric");
          if (key === "origin_period") cell.classList.add("origin-cell");
          if (typeof row[key] === "object" || key === "reason" || (typeof row[key] === "string" && row[key].length > 45)) cell.classList.add("wrap-cell");
          tr.append(cell);
        }
        tbody.append(tr);
      }
      count.textContent = `${visible.length.toLocaleString()} of ${rows.length.toLocaleString()} rows`;
      toggle.textContent = expanded ? `Show first ${pageSize} rows` : `Show all ${rows.length.toLocaleString()} rows`;
      toggle.setAttribute("aria-expanded", String(expanded));
    }
    toggle.addEventListener("click", () => { expanded = !expanded; draw(); });
    draw();
  }

  function definitionList(id, entries) {
    $(id).replaceChildren();
    for (const [key, value] of entries) {
      const pair = node("div");
      pair.append(node("dt", key), node("dd", display(value)));
      $(id).append(pair);
    }
  }

  function renderWorkflow(status) {
    $("workflow").replaceChildren();
    const current = status === "DRAFT" ? 0 : status === "SUBMITTED" ? 1 : 2;
    const titles = ["Draft", "In review", status === "REJECTED" ? "Rejected" : "Approved"];
    titles.forEach((title, index) => {
      const item = node("li", null, index === current ? "active" : index < current ? "completed" : "");
      if (index === current) item.setAttribute("aria-current", "step");
      item.append(node("span", String(index + 1)), node("strong", title));
      $("workflow").append(item);
    });
  }

  function renderActivity(events) {
    $("activity-list").replaceChildren();
    if (!events.length) $("activity-list").append(node("li", "No actions recorded.", "empty-message"));
    for (const event of [...events].reverse()) {
      const item = node("li", null, "activity-item");
      const heading = node("div", null, "activity-heading");
      const eventAction = String(event.action || "").toLowerCase();
      const time = node("time", timestamp(event.at));
      if (event.at) time.dateTime = event.at;
      heading.append(node("strong", actionNames[eventAction] || display(event.action)), time);
      item.append(heading, node("p", `${event.actor || "—"} · Revision ${display(event.revision)}`, "activity-meta"));
      if (event.reason) item.append(node("p", event.reason, "activity-reason"));
      const details = node("details");
      details.append(node("summary", "Inspect event record"), node("pre", JSON.stringify(event, null, 2)));
      item.append(details);
      $("activity-list").append(item);
    }
  }

  function renderRun() {
    const run = state.run;
    const snapshot = run.snapshot;
    const selected = snapshot.selected || {};
    const origins = snapshot.origins || [];
    const overrides = run.overrides || {};
    const status = run.status;
    if (state.exportRunId !== run.id || status !== "APPROVED") clearExportPreview();
    const editable = hasRole("analyst") && ownRun() && status === "DRAFT";
    const reviewable = hasRole("reviewer") && !ownRun() && status === "SUBMITTED";
    showPage("run");
    $("run-title").textContent = snapshot.title;
    $("run-subtitle").textContent = `Created by ${run.created_by} · ${timestamp(run.created_at)} · Revision ${run.revision}`;
    $("run-status").replaceWith(Object.assign(statusBadge(status), { id: "run-status" }));
    renderWorkflow(status);
    $("model-reserve").textContent = amount(run.model_reserve);
    $("booked-reserve").textContent = amount(run.booked_reserve);
    const adjustment = run.booked_reserve - run.model_reserve;
    $("reserve-adjustment").textContent = `${adjustment > 0 ? "+" : ""}${amount(adjustment)}`;
    document.querySelectorAll(".reserve-units").forEach((span) => { span.textContent = snapshot.units || "Source units"; });
    const overrideCount = Object.keys(overrides).length;
    $("override-count").textContent = `${overrideCount} ${overrideCount === 1 ? "origin with an adjustment" : "origins with adjustments"}`;
    $("selected-name").textContent = selected.name || "—";
    $("cutoff-value").textContent = snapshot.as_of;
    $("horizon-value").textContent = `${snapshot.horizon} months · ${snapshot.grain === "Y" ? "Annual" : snapshot.grain === "Q" ? "Quarterly" : "Monthly"}`;
    $("score-value").textContent = `${display(selected.mean_rmse)} · ${String(snapshot.metric).toUpperCase()} RMSE`;
    $("warnings-section").hidden = !snapshot.warnings?.length;
    $("warnings-list").replaceChildren();
    for (const warning of snapshot.warnings || []) $("warnings-list").append(node("li", display(warning)));
    const originRows = origins.map((origin) => {
      const override = overrides[origin.origin_period];
      return { origin_period: origin.origin_period, latest: origin.latest, ultimate: origin.ultimate, reserve: origin.reserve, booked: override ? override.reserve : origin.reserve, adjustment: override ? override.reserve - origin.reserve : 0, reason: override?.reason || "—", actor: override?.actor || "—" };
    });
    renderTable("origin-table", originRows, { caption: `Reserve amounts in ${snapshot.units || "source units"}`, columns: ["origin_period", "latest", "ultimate", "reserve", "booked", "adjustment", "reason", "actor"], money: ["latest", "ultimate", "reserve", "booked", "adjustment"] });
    $("override-form").hidden = !editable;
    const previousOrigin = $("override-origin").value;
    $("override-origin").replaceChildren();
    for (const origin of origins) {
      const option = node("option", origin.origin_period);
      option.value = origin.origin_period;
      $("override-origin").append(option);
    }
    if (origins.some((origin) => origin.origin_period === previousOrigin)) $("override-origin").value = previousOrigin;
    fillOverride();
    $("evidence-description").textContent = `Replay history begins ${snapshot.history_start}. Selection uses ${String(snapshot.metric).toUpperCase()} outcomes available by ${snapshot.as_of}, in ${snapshot.units || "source units"}. Historical scores are evidence about past forecasting performance.`;
    definitionList("settings-list", Object.entries(selected.settings || {}).map(([key, value]) => [label(key), settingValue(key, value)]));
    $("ranking-summary").textContent = `Candidate ranking · ${(snapshot.ranking || []).length} candidates`;
    renderTable("ranking-table", snapshot.ranking, { caption: "Historical candidate ranking; selected candidate shaded", highlight: (row) => row.candidate === selected.name });
    renderTable("history-table", snapshot.history_scores, { caption: "Historical scores by candidate and outcome date" });
    renderTable("factor-table", snapshot.factor_summary, { caption: "Fitted development factors and support" });
    renderTable("factor-selection-table", snapshot.factor_selection, { caption: "Observations included or excluded from each factor" });
    definitionList("provenance-list", [["Run ID", run.id], ["Parent run ID", run.parent_id], ["Superseded by", run.superseded_by],["Source SHA-256", snapshot.source_hash], ["Snapshot SHA-256", run.snapshot_hash], ["Engine", snapshot.engine], ["Loss field", snapshot.loss_field], ["Premium field", snapshot.premium_field], ["Units", snapshot.units], ["Segment", snapshot.segment], ["Saved source rows", snapshot.source_rows], ["History starts", snapshot.history_start], ["Information cutoff", snapshot.as_of], ["Created at", run.created_at], ["Created by", run.created_by], ["Revision", run.revision]]);
    $("saved-source-csv").textContent = typeof snapshot.source_csv === "string" ? snapshot.source_csv : "No source history recorded.";
    $("submit-form").hidden = !editable;
    $("decision-form").hidden = !reviewable;
    $("submit-form").reset();
    $("decision-form").reset();
    $("export-run").hidden = status !== "APPROVED";
    // A decided run allows one revision; the server refuses a second one.
    $("revise-run").hidden = !hasRole("analyst") || !ownRun() || Boolean(run.superseded_by) || !["APPROVED", "REJECTED"].includes(status);
    const descriptions = {
      DRAFT: editable ? "Review the evidence and adjustments, then submit this reserve position to a separate reviewer." : "The analyst is preparing this reserve position. It is not yet submitted for review.",
      SUBMITTED: reviewable ? "Review the evidence and recorded adjustments. Your decision and rationale will be preserved with this run." : ownRun() ? "This run is awaiting a separate reviewer's decision. The submitted reserve position is locked for review." : "This run is awaiting a reviewer's decision.",
      APPROVED: hasRole("analyst") && ownRun() ? "This reserve position is approved and read-only. Export the record, or create a linked draft revision to propose a new decision." : "This reserve position is approved and read-only. You can export the complete approved record. Further changes require a linked analyst revision.",
      REJECTED: "This reserve position was rejected and is read-only. The analyst can create a linked draft revision to address the review.",
    };
    $("review-description").textContent = run.superseded_by
      ? `This run was revised as ${run.superseded_by}, which now carries the current reserve position. This record stays read-only, and no further revision can be created from it.`
      : descriptions[status] || "This run is read-only.";
    renderActivity(run.events || []);
  }

  function fillOverride() {
    if (!state.run) return;
    const origin = $("override-origin").value;
    const source = state.run.snapshot.origins.find((row) => row.origin_period === origin);
    const override = state.run.overrides?.[origin];
    $("override-reserve").value = override?.reserve ?? source?.reserve ?? "";
    $("override-reason").value = override?.reason || "";
  }

  async function savedRun(path, body, message) {
    state.run = await api(path, { method: "POST", body });
    renderRun();
    notice(message);
    try { await loadRuns(); }
    catch (error) { notice(`${message} The analysis list could not be refreshed: ${error.message}`, true); }
  }

  $("login-form").addEventListener("submit", (event) => {
    event.preventDefault();
    operate("Signing in…", async () => {
      const raw = `${$("username").value}:${$("password").value}`;
      state.auth = `Basic ${btoa(Array.from(new TextEncoder().encode(raw), (byte) => String.fromCharCode(byte)).join(""))}`;
      try {
        state.me = await api("/api/me");
        let stored = true;
        try { sessionStorage.setItem(SESSION_KEY, state.auth); }
        catch { stored = false; }
        showWorkspace();
        await loadRuns();
        if (state.runs.length) await openRun(state.runs[0].id);
        else newAnalysis();
        if (!stored) notice("Signed in for this page only. Browser storage is unavailable, so reloading will require signing in again.");
      } finally { $("password").value = ""; }
    });
  });

  $("logout").addEventListener("click", () => {
    if (state.busy) return;
    clearSession();
    notice("You have signed out.");
    $("username").focus();
  });

  $("new-analysis").addEventListener("click", () => { if (!state.busy) { notice(""); newAnalysis(); } });
  $("refresh").addEventListener("click", () => operate("Refreshing workspace…", async () => {
    state.me = await api("/api/me");
    showWorkspace();
    await loadRuns();
    if (state.run) await openRun(state.run.id, false);
    else if (!hasRole("analyst") && state.runs.length) await openRun(state.runs[0].id);
    notice("Workspace refreshed.");
  }));

  $("csv-file").addEventListener("change", () => operate("Reading CSV…", async () => {
    const file = $("csv-file").files[0];
    state.csv = "";
    if (file && file.size > 1500000) {
      $("csv-status").textContent = "This file is larger than the 1.5 MB limit. Choose a smaller CSV.";
      throw new Error("Choose a CSV no larger than 1.5 MB.");
    }
    state.csv = file ? await file.text() : "";
    $("csv-status").textContent = file ? `${file.name} · ${file.size.toLocaleString()} bytes loaded` : "No data loaded.";
  }));

  $("load-demo").addEventListener("click", () => operate("Loading example data…", async () => {
    const example = await api("/api/demo");
    for (const [key, value] of Object.entries(example)) {
      const control = $("analysis-form").elements.namedItem(key);
      if (!control) continue;
      if (control.type === "checkbox") control.checked = Boolean(value);
      else control.value = value ?? "";
    }
    state.csv = example.csv || "";
    updateHorizonGrain();
    $("csv-file").value = "";
    $("csv-status").textContent = "Example cumulative triangle loaded. Choose a file to replace it.";
    notice("Example data and analysis settings loaded. Run the analysis to create a draft.");
  }));

  function updateHorizonGrain() {
    const step = { Y: 12, Q: 3, M: 1 }[$("grain").value];
    $("horizon").min = step;
    $("horizon").step = step;
    $("horizon").max = 720;
  }

  $("grain").addEventListener("change", updateHorizonGrain);
  updateHorizonGrain();

  $("analysis-form").addEventListener("submit", (event) => {
    event.preventDefault();
    // FormData omits disabled controls; capture before the busy state disables them.
    const payload = Object.fromEntries(new FormData($("analysis-form")));
    operate("Replaying historical forecasts and fitting the selected candidate. This may take a moment…", async () => {
      if (!state.csv.trim()) throw new Error("Choose a triangle CSV or load the example data before running the analysis.");
      payload.csv = state.csv;
      payload.horizon = Number(payload.horizon);
      payload.allow_unity = $("allow-unity").checked;
      await savedRun("/api/analyze", payload, "Analysis created. Inspect the evidence and reserve position before submitting it for review.");
      $("run-title").focus();
    });
  });

  $("override-origin").addEventListener("change", fillOverride);
  $("override-form").addEventListener("submit", (event) => {
    event.preventDefault();
    operate("Saving reserve adjustment…", async () => {
      const reserve = Number($("override-reserve").value);
      const reason = $("override-reason").value.trim();
      if (!Number.isFinite(reserve) || !reason) throw new Error("Provide a finite reserve amount and a reason for the adjustment.");
      await savedRun(runPath("/override"), { origin_period: $("override-origin").value, reserve, reason, expected_revision: state.run.revision }, "Reserve adjustment saved with its reason.");
    });
  });

  $("submit-form").addEventListener("submit", (event) => {
    event.preventDefault();
    operate("Submitting reserve position…", async () => {
      const reason = $("submit-reason").value.trim();
      if (!reason) throw new Error("Add a submission note before sending this run for review.");
      await savedRun(runPath("/submit"), { reason, expected_revision: state.run.revision }, "Submitted for review. The reserve position is locked while a separate reviewer decides.");
    });
  });

  $("decision-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const decision = event.submitter?.value;
    operate("Recording review decision…", async () => {
      const reason = $("decision-reason").value.trim();
      if (!reason || !["approve", "reject"].includes(decision)) throw new Error("Add a rationale and choose Approve or Reject.");
      await savedRun(runPath("/decision"), { decision, reason, expected_revision: state.run.revision }, decision === "approve" ? "Reserve position approved. The approved record is ready to export." : "Reserve position rejected. The analyst can create a linked revision.");
    });
  });

  $("revise-run").addEventListener("click", () => operate("Creating linked draft revision…", async () => {
    await savedRun(runPath("/revise"), { expected_revision: state.run.revision }, "Linked draft revision created. The prior decision remains preserved.");
    $("run-title").focus();
  }));

  function clearExportPreview() {
    state.exportRunId = null;
    $("export-json").textContent = "";
    $("export-preview").hidden = true;
    $("export-preview").open = false;
  }

  $("export-run").addEventListener("click", () => operate("Preparing approved record…", async () => {
    const record = await api(runPath("/export"), { download: true });
    $("export-json").textContent = await record.text();
    state.exportRunId = state.run.id;
    $("export-preview").hidden = false;
    $("export-preview").open = false;
    const url = URL.createObjectURL(record);
    const link = node("a");
    link.href = url;
    link.download = `reserve-review-${state.run.id}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    notice("Approved JSON export prepared. If your browser did not save it, open the JSON below.");
  }));

  operate("Connecting to review workspace…", async () => {
    const config = await api("/api/config", { authenticated: false });
    state.demo = Boolean(config.demo);
    $("demo-badge").hidden = !state.demo;
    $("demo-hint").hidden = !state.demo;
    try { state.auth = sessionStorage.getItem(SESSION_KEY); } catch { state.auth = null; }
    if (!state.auth) return;
    state.me = await api("/api/me");
    showWorkspace();
    await loadRuns();
    if (state.runs.length) await openRun(state.runs[0].id, false);
    else newAnalysis(false);
  });
})();
