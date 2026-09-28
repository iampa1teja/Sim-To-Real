// Pick-Place recorder UI. Served by sim_to_real_so101/utils/recording_web_gui.py:
// state + camera frames arrive on GET /events (Server-Sent Events),
// button clicks go to POST /command.

const { useEffect, useState, useRef } = React;

// "active" = an episode is running, including its pre-recording countdown.
const BUTTONS = [
  { cmd: "start", label: "Start recording", tone: "start", enabled: (s) => s.can_start && !s.active },
  { cmd: "stop", label: "Stop recording", tone: "stop", enabled: (s) => s.active },
  { cmd: "discard", label: "Discard recording", tone: "discard", enabled: (s) => s.active },
  // Teleporting the cube mid-episode would corrupt the recording.
  { cmd: "spawn", label: "Spawn the cube", tone: "spawn", enabled: (s) => !s.active },
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

function CameraPane({ name, frame }) {
  return (
    <figure className="pane">
      <figcaption>{name}</figcaption>
      {frame ? (
        <img src={`data:image/jpeg;base64,${frame}`} alt={name} />
      ) : (
        <div className="no-signal">No signal</div>
      )}
    </figure>
  );
}

function ExitDialog({ state, disabled, onExit, onCancel }) {
  const dialog = useRef(null);
  const cancel = useRef(null);
  const count = state.pending_videos + (state.recording ? 1 : 0);
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

function App() {
  const [state, connected] = useRecorderState();
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(false);
  const [exitOpen, setExitOpen] = useState(false);
  const live = connected && state !== null && !state.closed;
  const disabled = !live || pending || state?.busy || state?.pending;

  const onClick = (cmd, options) => {
    setError(null);
    setPending(true);
    sendCommand(cmd, options).catch((err) => setError(err.message)).finally(() => setPending(false));
  };

  return (
    <main>
      <header>
        <h1>Pick-Place Recorder</h1>
        <span className={`conn ${connected ? "on" : "off"}`}>
          {state?.closed ? "Recorder closed" : connected ? "Connected" : "Disconnected: is pick_place_agent running?"}
        </span>
      </header>

      <section className="toolbar">
        {BUTTONS.map(({ cmd, label, tone, enabled }) => (
          <button
            key={cmd}
            className={`btn ${tone}`}
            disabled={!live || pending || state.busy || state.pending || !enabled(state)}
            onClick={() => onClick(cmd)}
          >
            {label}
          </button>
        ))}
      </section>

      {error && <p className="error">{error}</p>}
      <p className={`status ${state?.active ? "rec" : ""}`}>
        {state ? state.status : "Waiting for the recorder…"}
      </p>

      <section className="grid">
        {(state?.cameras ?? []).map((name) => (
          <CameraPane key={name} name={name} frame={state.frames[name]} />
        ))}
      </section>
      <section className="bottom-bar" aria-label="Recording actions">
        <button className="btn stop" disabled={disabled || !state?.recording} onClick={() => onClick("save")}>Save</button>
        <button className="btn start" disabled={disabled || !state?.pending_videos || state?.encoding}
          onClick={() => onClick("encode")}>Encode videos</button>
        <button className="btn discard" disabled={disabled} onClick={() => setExitOpen(true)}>Exit</button>
      </section>
      <p className="encoding-status" role="status">{state?.closed ? "Recorder closed." : state?.encode_progress}</p>
      {exitOpen && !state?.closed && <ExitDialog state={state} disabled={disabled}
        onCancel={() => setExitOpen(false)} onExit={(encode) => {
          setExitOpen(false);
          onClick("exit", { encode });
        }} />}
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
