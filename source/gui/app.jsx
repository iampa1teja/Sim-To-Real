// Pick-Place recorder UI. Served by sim_to_real_so101/utils/recording_web_gui.py:
// state + camera frames arrive on GET /events (Server-Sent Events),
// button clicks go to POST /command.

const { useEffect, useState } = React;

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
      setState(JSON.parse(event.data));
      setConnected(true);
    };
    return () => events.close();
  }, []);

  return [state, connected];
}

async function sendCommand(cmd) {
  const response = await fetch("/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cmd }),
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

function App() {
  const [state, connected] = useRecorderState();
  const [error, setError] = useState(null);
  const live = connected && state !== null;

  const onClick = (cmd) => {
    setError(null);
    sendCommand(cmd).catch((err) => setError(err.message));
  };

  return (
    <main>
      <header>
        <h1>Pick-Place Recorder</h1>
        <span className={`conn ${connected ? "on" : "off"}`}>
          {connected ? "Connected" : "Disconnected: is pick_place_agent running?"}
        </span>
      </header>

      <section className="toolbar">
        {BUTTONS.map(({ cmd, label, tone, enabled }) => (
          <button
            key={cmd}
            className={`btn ${tone}`}
            disabled={!live || !enabled(state)}
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
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
