// s2r recorder UI (v1 GUI, kept until the new one replaces it). Served by src/sim/recording/web_gui.py:
// state + camera frames arrive on GET /events (Server-Sent Events),
// button clicks go to POST /command.

const { useEffect, useState, useRef } = React;
const CAMERA_ORDER = { "Sim: realsense_rgb": 0, "Sim: wrist_cam": 1, "Real: external": 2, "Real: gripper": 3 };

// "active" = an episode is running, including its pre-recording countdown.
const BUTTONS = [
  { cmd: "start", label: "Start recording", tone: "start", enabled: (s) => s.can_start && !s.active },
  { cmd: "stop", label: "Stop recording", tone: "stop", enabled: (s) => s.active },
  { cmd: "discard", label: "Discard recording", tone: "discard", enabled: (s) => s.active || s.rerecord_episode != null },
  // Teleporting the object mid-episode would corrupt the recording.
  { cmd: "spawn", label: "Spawn object", tone: "spawn", enabled: (s) => !s.active },
];

function useRecorderState() {
  const [state, setState] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    // EventSource reconnects on its own if the agent restarts.
    const events = new EventSource("/events");
    events.onopen = () => setConnected(true);
    events.onerror = () => setConnected(false);
    events.onmessage = (event) => {
      const next = JSON.parse(event.data);
      setState(next);
      if (next.closed) events.close();
      setConnected(true);
    };
    return () => events.close();
  }, []);

  return [state, connected];
}

async function sendCommand(cmd, options = {}) {
  const response = await fetch("/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cmd, ...options }),
  });
  if (!response.ok) {
    throw new Error(`"${cmd}" was rejected (HTTP ${response.status})`);
  }
}

function CameraPane({ name, frame, index }) {
  return (
    <figure className="pane">
      <figcaption className="pane-head">
        <span className="label">{`CAM-${String(index + 1).padStart(2, "0")}`}</span>
        <span className="pane-name">{name}</span>
      </figcaption>
      {frame ? (
        <img src={`data:image/jpeg;base64,${frame}`} alt={name} />
      ) : (
        <div className="no-signal label">No signal</div>
      )}
    </figure>
  );
}

function ExitDialog({ state, disabled, onExit, onCancel }) {
  const dialog = useRef(null);
  const cancel = useRef(null);
  // s2r encodes video while recording, so only episodes left from an older session can be pending.
  const count = state.pending_videos;
  useEffect(() => {
    const previous = document.activeElement;
    dialog.current.showModal();
    cancel.current.focus();
    return () => previous?.focus();
  }, []);
  return (
    <dialog ref={dialog} aria-labelledby="exit-title" onCancel={(event) => {
      event.preventDefault();
      onCancel();
    }} onClick={(event) => {
      const bounds = dialog.current.getBoundingClientRect();
      if (event.target === dialog.current && (event.clientX < bounds.left || event.clientX > bounds.right ||
          event.clientY < bounds.top || event.clientY > bounds.bottom)) onCancel();
    }}>
      <h2 id="exit-title">{count > 0
        ? `${count} episode(s) are not encoded yet. Do you want to encode them before exiting?`
        : "Exit the recorder?"}</h2>
      {state.recording && <p>The episode being recorded will be saved first.</p>}
      {count > 0 && <p className="status">Unencoded episodes stay as PNGs and are encoded automatically
        the next time this dataset is opened.</p>}
      <div className="dialog-actions">
        <button className="btn stop" disabled={disabled} onClick={() => onExit(true)}>
          {count > 0 ? "Encode & Exit" : "Exit"}
        </button>
        {count > 0 && <button className="btn discard" disabled={disabled} onClick={() => onExit(false)}>
          Exit without encoding
        </button>}
        <button ref={cancel} className="btn spawn" onClick={onCancel}>Cancel</button>
      </div>
    </dialog>
  );
}

function RerecordDialog({ state, disabled, onRecord, onCancel }) {
  const dialog = useRef(null);
  const input = useRef(null);
  const [episode, setEpisode] = useState("");
  const index = Number(episode);
  const valid = /^\d+$/.test(episode) && Number.isSafeInteger(index) && index < state.saved_episodes;
  useEffect(() => {
    const previous = document.activeElement;
    dialog.current.showModal();
    input.current.focus();
    return () => previous?.focus();
  }, []);
  return (
    <dialog ref={dialog} aria-labelledby="rerecord-title" onCancel={(event) => {
      event.preventDefault();
      onCancel();
    }}>
      <form onSubmit={(event) => {
        event.preventDefault();
        if (valid && !disabled) onRecord(index);
      }}>
        <h2 id="rerecord-title">Re-record a saved episode</h2>
        <p>Choose episode 0–{state.saved_episodes - 1}. Prepare the scene, then press Start recording.
          Save replaces that episode in the sim and connected real dataset; Discard keeps the original.</p>
        <p>A backup of each original dataset is kept when you save. Preparing the copy needs extra disk space.</p>
        <label htmlFor="rerecord-episode">Episode number</label>{" "}
        <input ref={input} id="rerecord-episode" type="number" min="0" max={state.saved_episodes - 1}
          step="1" required value={episode} onChange={(event) => setEpisode(event.target.value)} />
        <div className="dialog-actions">
          <button type="submit" className="btn start" disabled={disabled || !valid}>Prepare re-recording</button>
          <button type="button" className="btn spawn" onClick={onCancel}>Cancel</button>
        </div>
      </form>
    </dialog>
  );
}

function App() {
  const [state, connected] = useRecorderState();
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(false);
  const [exitOpen, setExitOpen] = useState(false);
  const [rerecordOpen, setRerecordOpen] = useState(false);
  const live = connected && state !== null && !state.closed;
  const disabled = !live || pending || state?.busy || state?.pending;
  const cameras = [...(state?.cameras ?? [])].sort(
    (a, b) => (CAMERA_ORDER[a] ?? 4) - (CAMERA_ORDER[b] ?? 4)
  );

  const onClick = (cmd, options) => {
    setError(null);
    setPending(true);
    sendCommand(cmd, options).catch((err) => setError(err.message)).finally(() => setPending(false));
  };

  return (
    <main>
      <header>
        <div className="title-block">
          <h1>S2R Recorder</h1>
          <span className="label">REC-01</span>
        </div>
        <div className="strip-right">
          <span className="conn-text">
            {state?.closed ? "Recorder closed" : connected ? "Connected" : "Disconnected: is lerobot_agent running?"}
          </span>
          <span className="signal" data-state={connected ? "ok" : "off"}>{connected ? "Link" : "No link"}</span>
        </div>
      </header>

      <section className="toolbar">
        {BUTTONS.map(({ cmd, label, tone, enabled }) => (
          <button
            key={cmd}
            className={`btn ${tone}`}
            disabled={!live || pending || state.busy || state.pending || !enabled(state)}
            onClick={() => onClick(cmd)}
          >
            {state?.mode === "eval-sim-real" && cmd === "start" ? "Start real → sim" :
              state?.mode === "eval-sim-real" && cmd === "stop" ? "Stop evaluation" : label}
          </button>
        ))}
        <button className="btn stop"
          disabled={disabled || state?.active || state?.encoding || !!state?.pending_videos ||
            !state?.saved_episodes || state?.rerecord_episode != null}
          onClick={() => setRerecordOpen(true)}>Re-record episode</button>
      </section>

      {error && <p className="error"><span className="label">Fault</span>{error}</p>}
      <div className="statusbar">
        {state?.active && <span className="signal" data-state="live">Live</span>}
        {state && <span className="readout" title="Saved episodes">
          <span className="label">Saved</span>
          <span className="readout-value">{String(state.saved_episodes).padStart(3, "0")}</span>
        </span>}
        <p className={`status ${state?.active ? "rec" : ""}`}>
          {state ? state.status : "Waiting for the recorder…"}
        </p>
      </div>
      {state?.rerecord_episode != null && <p className="status status-note" role="status">
        Replacing episode {state.rerecord_episode}. The original stays safe until Save.
      </p>}

      <section className="grid">
        {cameras.map((name, index) => (
          <CameraPane key={name} name={name} index={index} frame={state.frames[name]} />
        ))}
      </section>
      <section className="bottom-bar" aria-label="Recording actions">
        <button className="btn stop" disabled={disabled || !state?.recording} onClick={() => onClick("save")}>Save</button>
        <button className="btn stop" disabled={disabled || !state?.pending_videos || state?.encoding}
          onClick={() => onClick("encode")}>Encode videos</button>
        <button className="btn discard" disabled={disabled} onClick={() => setExitOpen(true)}>Exit</button>
      </section>
      <p className="encoding-status" role="status">{state?.closed ? "Recorder closed." : state?.encode_progress}</p>
      {exitOpen && !state?.closed && <ExitDialog state={state} disabled={disabled}
        onCancel={() => setExitOpen(false)} onExit={(encode) => {
          setExitOpen(false);
          onClick("exit", { encode });
        }} />}
      {rerecordOpen && !state?.closed && <RerecordDialog state={state}
        disabled={disabled || state?.active || state?.encoding || !!state?.pending_videos || state?.rerecord_episode != null}
        onCancel={() => setRerecordOpen(false)} onRecord={(episode) => {
          setRerecordOpen(false);
          onClick("rerecord", { episode });
        }} />}
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
