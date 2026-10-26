// Sim <-> Real paired evaluation UI. Served by sim_to_real_so101/utils/sim_real_web_gui.py:
// state arrives on GET /events (Server-Sent Events), commands go to POST /command.

const { useEffect, useState, useRef } = React;

const STAGE_ORDER = ["none", "reached", "grasped", "lifted", "over_box", "placed", "success"];
const STAGES = STAGE_ORDER.slice(1);
const STAGE_LABELS = {
  none: "Nothing", reached: "Reached", grasped: "Grasped", lifted: "Lifted",
  over_box: "Over box", placed: "Placed", success: "Success",
};
const SIDE_M = 0.02; // side of the start-pose square on the map

const jpeg = (frame) => `data:image/jpeg;base64,${frame}`;
const show = (value) => (value === null || value === undefined || value === "" ? "—" : String(value));
const fmtSec = (value) => (typeof value === "number" ? `${value.toFixed(1)} s` : show(value));
const fmtPct = (rate) => (typeof rate === "number" ? (100 * rate).toFixed(1) : "—");
const fmtPp = (gap) => (typeof gap === "number" ? (100 * gap).toFixed(1) : "—");
const fmtP = (p) => (typeof p === "number" ? p.toFixed(3) : "—");
const fmtM = (v) => (typeof v === "number" ? v.toFixed(3) : "—");
const toDeg = (rad) => (typeof rad === "number" ? ((rad * 180) / Math.PI).toFixed(1) : "—");
const round3 = (v) => Math.round(v * 1000) / 1000;
const parseField = (text) => (text.trim() === "" ? NaN : Number(text));
const labelText = (label) => {
  if (label === null || label === undefined) return "—";
  return typeof label === "string" ? label : show(label.stage);
};

function useSimRealStream() {
  const [payload, setPayload] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    // EventSource reconnects on its own if the agent restarts.
    const events = new EventSource("/events");
    events.onopen = () => setConnected(true);
    events.onerror = () => setConnected(false);
    events.onmessage = (event) => {
      let next = null;
      try {
        next = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      setPayload(next);
      if (next.closed) events.close();
      setConnected(true);
    };
    return () => events.close();
  }, []);

  return [payload, connected];
}

async function postCommand(body) {
  const response = await fetch("/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = (await response.text()).trim();
    throw new Error(`HTTP ${response.status}${detail ? `: ${detail}` : ""}`);
  }
}

function mapViewBox(points) {
  if (points.length === 0) return { x: -0.5, y: -0.5, w: 1, h: 1 };
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const w = Math.max(maxX - minX, 0.1);
  const h = Math.max(maxY - minY, 0.1);
  const mx = w * 0.05;
  const my = h * 0.05;
  // y is flipped for display, so the top edge of the view sits at -maxY in SVG units.
  return { x: minX - mx, y: -maxY - my, w: w + 2 * mx, h: h + 2 * my };
}

// SVG user units are (x, -y) so that +y is drawn upwards.
const polyPoints = (points) => points.map(([x, y]) => `${x},${-y}`).join(" ");

function squarePoints({ x, y, yaw }, side) {
  const half = side / 2;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return [[-half, -half], [half, -half], [half, half], [-half, half]]
    .map(([u, v]) => `${x + u * c - v * s},${-(y + u * s + v * c)}`)
    .join(" ");
}

function StartMap({ info, current, disabled, onPick }) {
  const svgRef = useRef(null);
  const scene = info?.scene ?? {};
  const outline = scene.table_outline_base_xy ?? [];
  const footprint = scene.box_footprint_base_xy ?? [];
  const starts = info?.starts ?? [];
  const view = mapViewBox(outline);
  const dot = view.w * 0.008;
  const robot = view.w * 0.016;

  const onClick = (event) => {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM ? svg.getScreenCTM() : null;
    if (disabled || !ctm) return;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const local = point.matrixTransform(ctm.inverse());
    onPick(round3(local.x), round3(-local.y));
  };

  return (
    <svg ref={svgRef} className="map" viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
      role="img" aria-label="Top-down map in the robot base frame, click to set x and y" onClick={onClick}>
      {outline.length > 0 && <polygon className="map-table" points={polyPoints(outline)} />}
      {footprint.length > 0 && <polygon className="map-box" points={polyPoints(footprint)} />}
      {starts.map((s, i) => <circle key={`${s.start_id}-${i}`} className="map-start" cx={s.x} cy={-s.y} r={dot} />)}
      {current && <polygon className="map-current" points={squarePoints(current, SIDE_M)} />}
      <circle className="map-robot" cx={0} cy={0} r={robot} />
    </svg>
  );
}

function Pane({ title, foot, children }) {
  return (
    <figure className="pane">
      <figcaption className="pane-head">
        <span className="label">{title}</span>
      </figcaption>
      <div className="stage">{children}</div>
      {foot && <div className="pane-foot">{foot}</div>}
    </figure>
  );
}

function Frame({ name, frame }) {
  return frame
    ? <img src={jpeg(frame)} alt={name} />
    : <div className="no-signal label">No frame</div>;
}

function KV({ rows }) {
  return (
    <dl className="kv">
      {rows.map(([key, value]) => (
        <div key={key}><dt>{key}</dt><dd>{value}</dd></div>
      ))}
    </dl>
  );
}

function SummaryPanel({ summary }) {
  if (!summary || !(summary.n > 0)) {
    return <p className="status">No paired trials yet.</p>;
  }
  const rates = summary.stage_rates ?? {};
  const confusion = summary.confusion ?? {};
  const paired = summary.paired_success ?? {};
  return (
    <>
      <p className="status">{`n = ${summary.n} paired trials`}</p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Stage</th><th className="n">Sim %</th><th className="n">Real %</th><th className="n">Gap (pp)</th></tr>
          </thead>
          <tbody>
            {STAGES.map((stage) => {
              const row = rates[stage] ?? {};
              return (
                <tr key={stage}>
                  <th scope="row">{stage}</th>
                  <td className="n">{fmtPct(row.sim)}</td>
                  <td className="n">{fmtPct(row.real)}</td>
                  <td className="n">{fmtPp(row.gap)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="label sub">Confusion: sim rows, real columns</p>
      <div className="table-wrap">
        <table className="confusion">
          <thead>
            <tr>
              <th>sim / real</th>
              {STAGE_ORDER.map((r) => <th key={r} className="n">{r}</th>)}
            </tr>
          </thead>
          <tbody>
            {STAGE_ORDER.map((s) => (
              <tr key={s}>
                <th scope="row">{s}</th>
                {STAGE_ORDER.map((r) => <td key={r} className="n">{confusion[s]?.[r] ?? 0}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="status sub">
        {`both ${paired.both ?? 0} / sim only ${paired.sim_only ?? 0} / real only ${paired.real_only ?? 0} / neither ${paired.neither ?? 0}, McNemar p = ${fmtP(paired.mcnemar?.p_value)}`}
      </p>
    </>
  );
}

function TrialsTable({ trials }) {
  if (trials.length === 0) return <p className="status">No saved trials yet.</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th className="n">ID</th><th>Start</th><th className="n">x (m)</th><th className="n">y (m)</th>
            <th className="n">Yaw (deg)</th><th>Sim</th><th>Real</th><th>Real note</th>
          </tr>
        </thead>
        <tbody>
          {trials.map((t) => (
            <tr key={t.trial_id}>
              <td className="n">{show(t.trial_id)}</td>
              <td>{show(t.start_id)}</td>
              <td className="n">{fmtM(t.x)}</td>
              <td className="n">{fmtM(t.y)}</td>
              <td className="n">{toDeg(t.yaw)}</td>
              <td>{show(t.sim?.stage)}</td>
              <td>{show(t.real?.stage)}</td>
              <td>{show(t.real?.note)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function App() {
  const [payload, connected] = useSimRealStream();
  const [fault, setFault] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [manual, setManual] = useState({ x: "0", y: "0", yaw: "0" });
  const [pick, setPick] = useState("none");
  const [note, setNote] = useState("");
  const [opacity, setOpacity] = useState(50);
  const [difference, setDifference] = useState(false);

  const info = payload?.info ?? null;
  const session = payload?.session ?? null;
  const frames = payload?.frames ?? {};
  const state = session?.state ?? "idle";
  const closed = !!payload?.closed;
  const live = connected && !!payload && !closed;
  const locked = !live || !!payload?.busy || !!payload?.pending;
  const running = state.endsWith("_running");
  const realPlaced = !!session?.real_placed;
  const starts = info?.starts ?? [];
  const index = selectedIndex < starts.length ? selectedIndex : 0;
  const trials = [...(payload?.trials ?? [])].sort((a, b) => b.trial_id - a.trial_id);

  // Enabled buttons follow the session state table in the contract.
  const canSpawn = live && !locked && ["idle", "spawned", "sim_done"].includes(state);
  const canPlaced = live && !locked && ["spawned", "sim_done"].includes(state);
  const canRunSim = live && !locked && ["spawned", "sim_done", "real_done"].includes(state);
  const canRunReal = live && !locked && realPlaced && ["sim_done", "real_done"].includes(state);
  const canDiscard = live && !locked && ["spawned", "sim_done", "real_done"].includes(state);
  const canLabel = live && !locked && state === "real_done";
  const canSave = canLabel && session?.label != null;
  const canStop = live && running; // Stop is accepted even while busy

  const send = (body) => {
    setFault(null);
    postCommand(body).catch((err) => setFault(err.message));
  };

  const spawnManual = () => {
    const x = parseField(manual.x);
    const y = parseField(manual.y);
    const yawDeg = parseField(manual.yaw);
    if (![x, y, yawDeg].every(Number.isFinite)) {
      setFault("x, y and yaw must be numbers");
      return;
    }
    send({ cmd: "spawn", x, y, yaw: (yawDeg * Math.PI) / 180 });
  };

  const pickPoint = (x, y) => setManual((m) => ({ ...m, x: String(x), y: String(y) }));

  return (
    <main>
      <header>
        <div className="title-block">
          <h1>Sim ↔ Real Eval</h1>
          <span className="label">PAIR-01</span>
        </div>
        <div className="strip-right">
          <span className="conn-text">{connected ? "Connected" : "Disconnected"}</span>
          <span className="signal" data-state={connected ? "ok" : "off"}>{connected ? "Link" : "No link"}</span>
        </div>
      </header>

      {info && (info.motion
        ? <p className="banner motion" role="alert">MOTION ENABLED — keep a hand on the power switch</p>
        : <p className="banner shadow" role="status">SHADOW MODE — the real arm is not commanded</p>)}
      {closed && <p className="banner closed closed-note" role="status">Evaluation closed</p>}

      <dl className="kv" style={{ marginTop: "var(--space-3)" }}>
        <div><dt>Policy</dt><dd>{show(info?.policy)}</dd></div>
        <div><dt>Checkpoint</dt><dd>{show(info?.checkpoint)}</dd></div>
        <div><dt>Instruction</dt><dd>{show(info?.instruction)}</dd></div>
        <div><dt>Eval set</dt><dd>{show(info?.eval_set)}</dd></div>
      </dl>

      <div className="statusbar">
        <span className="readout" title="Session state">
          <span className="label">State</span>
          <span className="readout-value">{state}</span>
        </span>
        <p className={`status ${running ? "rec" : ""}`}>
          {payload ? show(payload.status) : "Waiting for the evaluation server…"}
        </p>
      </div>
      {fault && <p className="fault" role="alert"><span className="label">Fault</span>{fault}</p>}

      <div className="columns">
        <div className="col">
          <section className="panel" aria-label="Start picker">
            <h2 className="label panel-title">Start</h2>
            <div className="row">
              <select aria-label="Eval-set start" value={index} disabled={!canSpawn || starts.length === 0}
                onChange={(e) => setSelectedIndex(Number(e.target.value))}>
                {starts.map((s, i) => (
                  <option key={`${s.start_id}-${i}`} value={i}>{`${s.start_id} (${s.split})`}</option>
                ))}
              </select>
              <button className="btn" disabled={!canSpawn || starts.length === 0}
                onClick={() => send({ cmd: "spawn", start_id: starts[index]?.start_id })}>
                Spawn selected
              </button>
            </div>
            <div className="fields">
              <label className="field">
                <span className="label">x (m)</span>
                <input type="number" step="any" value={manual.x} disabled={!canSpawn}
                  onChange={(e) => setManual((m) => ({ ...m, x: e.target.value }))} />
              </label>
              <label className="field">
                <span className="label">y (m)</span>
                <input type="number" step="any" value={manual.y} disabled={!canSpawn}
                  onChange={(e) => setManual((m) => ({ ...m, y: e.target.value }))} />
              </label>
              <label className="field">
                <span className="label">yaw (deg)</span>
                <input type="number" step="any" value={manual.yaw} disabled={!canSpawn}
                  onChange={(e) => setManual((m) => ({ ...m, yaw: e.target.value }))} />
              </label>
            </div>
            <button className="btn" disabled={!canSpawn} onClick={spawnManual}>Spawn manual</button>
          </section>

          <section className="panel" aria-label="Map">
            <h2 className="label panel-title">Map, robot base frame (m)</h2>
            <StartMap info={info} current={session?.start ?? null} disabled={!canSpawn} onPick={pickPoint} />
            <p className="hint">Click the map to fill x and y. +y is up.</p>
          </section>

          <section className="panel" aria-label="Trial">
            <h2 className="label panel-title">Trial</h2>
            <KV rows={[
              ["Start", show(session?.start?.start_id)],
              ["Real placed", session ? (realPlaced ? "yes" : "no") : "—"],
              ["Label", labelText(session?.label)],
            ]} />
            {Array.isArray(session?.start?.arm_start_state) && (
              <div className="arm-start">
                <h3 className="label">Arm start pose (raw units)</h3>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        {["pan", "lift", "elbow", "wrist flex", "wrist roll", "gripper"].map((h) => (
                          <th key={h} className="n">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        {session.start.arm_start_state.map((v, i) => (
                          <td key={i} className="n">{typeof v === "number" ? v.toFixed(1) : "—"}</td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            <button className="btn start stop-btn" disabled={!canStop} onClick={() => send({ cmd: "stop" })}>
              Stop
            </button>
            <div className="trial-actions">
              <button className="btn" disabled={!canPlaced} onClick={() => send({ cmd: "placed" })}>Placed</button>
              <button className="btn" disabled={!canRunSim} onClick={() => send({ cmd: "run_sim" })}>Run sim</button>
              <button className="btn" disabled={!canRunReal} onClick={() => send({ cmd: "run_real" })}>Run real</button>
              <button className="btn discard" disabled={!canDiscard} onClick={() => send({ cmd: "discard" })}>
                Discard
              </button>
              <button className="btn" disabled={!canSave} onClick={() => send({ cmd: "save" })}>Save</button>
            </div>
            <div className="results">
              <div>
                <h3 className="label">Sim result</h3>
                {session?.sim
                  ? <KV rows={[
                      ["Stage", show(session.sim.stage)],
                      ["Success", session.sim.success ? "yes" : "no"],
                      ["Steps", show(session.sim.steps)],
                      ["Seconds", fmtSec(session.sim.seconds)],
                      ["Failure", show(session.sim.failure_mode)],
                    ]} />
                  : <p className="status">Not run</p>}
              </div>
              <div>
                <h3 className="label">Real result</h3>
                {session?.real
                  ? <KV rows={[
                      ["Status", show(session.real.status)],
                      ["Steps", show(session.real.steps)],
                      ["Seconds", fmtSec(session.real.seconds)],
                      ["Detail", show(session.real.detail)],
                      ["Stage", show(session.real.stage)],
                      ["Note", show(session.real.note)],
                    ]} />
                  : <p className="status">Not run</p>}
              </div>
            </div>
          </section>

          <section className="panel" aria-label="Real stage labelling">
            <h2 className="label panel-title">Real stage</h2>
            <fieldset className="stage-picker" disabled={!canLabel}>
              <legend className="label">Furthest stage reached</legend>
              {STAGE_ORDER.map((s) => (
                <label key={s} className="radio">
                  <input type="radio" name="real-stage" value={s} checked={pick === s} disabled={!canLabel}
                    onChange={() => setPick(s)} />
                  {STAGE_LABELS[s]}
                </label>
              ))}
            </fieldset>
            <label className="field" style={{ marginBottom: "var(--space-3)" }}>
              <span className="label">Note</span>
              <input type="text" maxLength={500} value={note} disabled={!canLabel}
                onChange={(e) => setNote(e.target.value)} />
            </label>
            <button className="btn" disabled={!canLabel}
              onClick={() => send({ cmd: "label", stage: pick, note })}>
              Set label
            </button>
          </section>
        </div>

        <div className="col">
          <section className="panel" aria-label="Cameras">
            <h2 className="label panel-title">Cameras</h2>
            <div className="cam-grid">
              <Pane title="Sim room"><Frame name="Sim room" frame={frames.sim_room} /></Pane>
              <Pane
                title="Real room (overlay)"
                foot={
                  <>
                    <label className="ctl">
                      <span className="label">Overlay</span>
                      <input type="range" min="0" max="100" step="1" value={opacity}
                        disabled={difference || closed}
                        onChange={(e) => setOpacity(Number(e.target.value))} />
                      <span>{`${difference ? 100 : opacity}%`}</span>
                    </label>
                    <label className="ctl">
                      <input type="checkbox" checked={difference} disabled={closed}
                        onChange={(e) => setDifference(e.target.checked)} />
                      Difference
                    </label>
                  </>
                }
              >
                {frames.real_room
                  ? <img src={jpeg(frames.real_room)} alt="Real room" />
                  : <div className="no-signal label">No frame</div>}
                {frames.sim_room && (
                  <img className="overlay" src={jpeg(frames.sim_room)} alt="Sim room overlay"
                    style={{
                      opacity: difference ? 1 : opacity / 100,
                      mixBlendMode: difference ? "difference" : "normal",
                    }} />
                )}
              </Pane>
              <Pane title="Sim wrist"><Frame name="Sim wrist" frame={frames.sim_wrist} /></Pane>
              <Pane title="Real wrist"><Frame name="Real wrist" frame={frames.real_wrist} /></Pane>
            </div>
          </section>

          <section className="panel" aria-label="Summary">
            <h2 className="label panel-title">Summary</h2>
            <SummaryPanel summary={payload?.summary ?? null} />
          </section>

          <section className="panel" aria-label="Trials">
            <h2 className="label panel-title">Trials, newest first</h2>
            <TrialsTable trials={trials} />
          </section>
        </div>
      </div>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
