const $ = (id) => document.getElementById(id);
const clientId = sessionStorage.getItem('s2r-real-client') || crypto.randomUUID();
sessionStorage.setItem('s2r-real-client', clientId);
let config, state, live = false, pending = '', localError = '', historyKey = '', lastFrame = -1;
let stageSide = null;
const cameraElements = new Map();
const isEvaluation = () => config?.mode === 'eval-sim-real';
const labels = {disconnected:'Disconnected', connecting_robot:'Connecting arm', ready:'Ready', queued:'Preparing',
  connecting_policy:'Connecting policy', running:'Running', real:'Real trial', sim:'Simulation trial', spawning:'Spawning',
  ending_real:'Ending real trial', ending_sim:'Ending simulation trial',
  stopping:'Stopping', finishing:'Saving', disconnecting:'Disconnecting'};

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache:'no-store'} : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({...body, client_id:clientId}),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}

function policyChanged() {
  const policy = config.policies.find(p => p.name === $('policy').value);
  $('policy-description').textContent = policy ? (policy.reason || policy.description) : 'No compatible policy is available.';
  if (policy) $('port').value = policy.name === config.defaults.policy && config.defaults.port ? config.defaults.port : policy.port;
  render();
}

function cameraLabel(name) {
  return name === 'external' ? 'External camera' : name === 'gripper' ? 'Gripper camera' : name;
}

function refreshCamera(view) {
  const connected = live && state?.connected;
  const available = connected || (isEvaluation() && view.name.startsWith('Sim: '));
  if (live && available && state?.cameras.includes(view.name)) {
    view.image.src = `/api/camera?name=${encodeURIComponent(view.name)}&v=${state.frame_seq}`;
  } else {
    view.image.hidden = true; view.image.removeAttribute('src'); view.placeholder.hidden = false;
    view.placeholder.textContent = view.name.startsWith('Sim: ') ? 'Waiting for a simulation frame…' : 'Connect the real arm to view this camera.';
  }
}

function addCamera(name, index) {
  const figure = document.createElement('figure'); figure.className = 'camera';
  const caption = document.createElement('figcaption');
  const label = document.createElement('select'); label.className = 'camera-select';
  label.setAttribute('aria-label', `Camera panel ${index + 1}`);
  for (const camera of config.cameras) {
    const option = document.createElement('option'); option.value = camera; option.textContent = cameraLabel(camera); label.append(option);
  }
  label.value = name;
  const number = document.createElement('span'); number.className = 'camera-number'; number.textContent = `CAM ${String(index + 1).padStart(2,'0')}`;
  caption.append(label, number);
  const well = document.createElement('div'); well.className = 'camera-well';
  const image = document.createElement('img'); image.alt = `${cameraLabel(name)}, live camera feed`; image.hidden = true;
  const placeholder = document.createElement('div'); placeholder.className = 'no-camera'; placeholder.textContent = 'Connect the real arm to view this camera.';
  image.addEventListener('load', () => { image.hidden = false; placeholder.hidden = true; });
  image.addEventListener('error', () => { image.hidden = true; placeholder.hidden = false; placeholder.textContent = 'Waiting for a camera frame…'; });
  well.append(image, placeholder); figure.append(caption, well); $('cameras').append(figure);
  const view = {image, placeholder, name};
  label.addEventListener('change', () => {
    view.name = label.value; image.alt = `${cameraLabel(view.name)}, live camera feed`; image.hidden = true;
    placeholder.hidden = false; placeholder.textContent = 'Waiting for a camera frame…'; refreshCamera(view);
    sessionStorage.setItem(`s2r-${isEvaluation() ? 'paired' : 'real'}-camera-${index}`, view.name);
  });
  cameraElements.set(index, view);
}

function renderHistory() {
  const history = state?.history || [], key = JSON.stringify(history);
  if (key === historyKey) return;
  historyKey = key; $('history-rows').replaceChildren();
  $('empty-history').hidden = history.length > 0; $('history-table').hidden = !history.length;
  for (const run of history) {
    const row = document.createElement('tr');
    const stageLabel = (value, seconds) => (config.stages?.find(stage => stage.value === value)?.label || 'Not labelled') +
      (typeof seconds === 'number' ? ` · ${seconds.toFixed(1)} s` : '');
    const stages = isEvaluation() ? [stageLabel(run.real_stage, run.real_seconds), stageLabel(run.sim_stage, run.sim_seconds)] : [];
    for (const value of [run.policy, run.instruction, ...stages, run.status === 'completed' ? 'Finished' : run.status === 'stopped' ? 'Stopped' : 'Error', `${run.seconds.toFixed(1)} s`]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    if (run.detail) row.title = run.detail;
    const cell = document.createElement('td'), link = document.createElement('a');
    link.href = `/api/report/${encodeURIComponent(run.id)}`; link.target = '_blank'; link.rel = 'noopener'; link.className = 'report-link'; link.textContent = 'View report';
    cell.append(link); row.append(cell); $('history-rows').append(row);
  }
}

function render() {
  const connected = live && state?.connected, busy = !!state?.busy;
  const active = ['queued','connecting_policy','running','real','sim','ending_real','ending_sim','stopping','finishing'].includes(state?.phase);
  $('connection').textContent = !live ? 'Console offline' : connected ? 'Arm connected' : state?.phase === 'connecting_robot' ? 'Connecting…' : 'Arm disconnected';
  $('connection').classList.toggle('online', !!connected);
  $('connect').textContent = connected ? 'Disconnect' : state?.phase === 'connecting_robot' ? 'Connecting…' : 'Connect real arm';
  $('connect').disabled = !live || !!pending || state?.phase === 'connecting_robot' || state?.phase === 'disconnecting';
  for (const id of ['policy','prompt','duration','host','port']) $(id).disabled = !config || busy || !!pending;
  const policy = config?.policies.find(p => p.name === $('policy').value);
  $('run').disabled = !connected || busy || !!pending || !policy?.compatible || !$('prompt').value.trim() ||
    (isEvaluation() && !state?.can_start);
  $('spawn').disabled = !connected || busy || !!pending;
  $('stop').disabled = !live || !(active || state?.positioning || pending === 'run');
  const side = state?.phase === 'real' ? 'real' : state?.phase === 'sim' ? 'sim' : null;
  if (side !== stageSide) { $('end-stage').value = ''; stageSide = side; }
  $('end-stage-label').textContent = side === 'real' ? 'Real trial ended at' : side === 'sim' ? 'Simulation trial ended at' : 'Trial ended at';
  $('end-trial').textContent = side === 'real' ? 'End real trial → run sim' : side === 'sim' ? 'End simulation trial' : 'End trial';
  $('end-stage-help').textContent = side === 'sim' ? 'Record the stage you observed and finish this comparison.' :
    'Record the stage you observed. Ending the real trial holds the arm and starts simulation.';
  const canEnd = live && isEvaluation() && connected && !!side && !pending && state.ending_side !== side;
  $('end-stage').disabled = !canEnd;
  $('end-trial').disabled = !canEnd || !$('end-stage').value;
  const error = localError || state?.error || '';
  $('error').textContent = error; $('error').hidden = !error;
  $('phase').textContent = labels[state?.phase] || 'Disconnected';
  if ($('status').textContent !== (state?.status || 'Waiting for the console…')) $('status').textContent = state?.status || 'Waiting for the console…';
  $('metrics').hidden = !active; $('progress').hidden = !active;
  $('elapsed').textContent = (state?.elapsed || 0).toFixed(1); $('steps').textContent = state?.steps || 0;
  $('progress').max = state?.duration || 30; $('progress').value = Math.min(state?.elapsed || 0, state?.duration || 30);
  if (state && state.frame_seq !== lastFrame) {
    lastFrame = state.frame_seq;
    for (const view of cameraElements.values()) refreshCamera(view);
  }
  renderHistory();
}

async function command(name, body = {}) {
  pending = name; localError = ''; render();
  try { await api(`/api/${name}`, body); await poll(); }
  catch (error) { localError = error.message; }
  finally { pending = ''; render(); }
}

let polling = false;
async function poll() {
  if (polling) return;
  polling = true;
  try { state = await api('/api/state'); live = true; }
  catch { live = false; }
  finally { polling = false; render(); }
}

$('run-form').addEventListener('submit', (event) => {
  event.preventDefault();
  if (!$('run-form').reportValidity() || $('run').disabled) return;
  command('run', {policy:$('policy').value, prompt:$('prompt').value, host:$('host').value.trim(),
    port:Number($('port').value), seconds:Number($('duration').value)});
});
$('connect').addEventListener('click', () => command(state?.connected ? 'disconnect' : 'connect'));
$('stop').addEventListener('click', () => command('stop'));
$('spawn').addEventListener('click', () => command('spawn'));
$('end-stage').addEventListener('change', render);
$('end-trial').addEventListener('click', () => {
  if (!$('end-trial').disabled) command('end-trial', {side:stageSide, stage:$('end-stage').value});
});
$('policy').addEventListener('change', policyChanged);
$('prompt').addEventListener('input', render);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('stop').disabled) { event.preventDefault(); command('stop'); }
});
window.addEventListener('pagehide', () => {
  if (state?.owner === clientId || pending === 'run') {
    navigator.sendBeacon('/api/stop', new Blob([JSON.stringify({client_id:clientId})], {type:'application/json'}));
  }
});

async function initialize() {
  try {
    config = await api('/api/config');
    if (isEvaluation()) {
      document.title = 'S2R · Sim–real evaluation'; $('console-title').textContent = 'Sim–real evaluation';
      $('run').textContent = 'Run real → sim'; $('duration-label').textContent = 'Duration per side';
      $('spawn').hidden = !config.requires_spawn; $('spawn-help').hidden = !config.requires_spawn;
      $('trial-end-controls').hidden = false;
      $('real-stage-heading').hidden = $('sim-stage-heading').hidden = false;
      for (const stage of config.stages || []) {
        const option = document.createElement('option'); option.value = stage.value; option.textContent = stage.label;
        $('end-stage').append(option);
      }
      $('monitor-note').textContent = 'Real / simulation';
      $('console-footer').textContent = 'Both trials use the same prompt, policy and saved starting pose. Camera selectors change the preview only.';
    }
    $('arm-label').textContent = `${config.arm.toUpperCase()} · ${config.follower_id || 'Follower not configured'}`;
    $('policy').replaceChildren();
    for (const policy of config.policies) {
      const option = document.createElement('option'); option.value = policy.name;
      option.textContent = policy.name + (policy.compatible ? '' : ' — incompatible'); option.disabled = !policy.compatible;
      $('policy').append(option);
    }
    const preferred = config.policies.find(p => p.name === config.defaults.policy && p.compatible) || config.policies.find(p => p.compatible);
    if (preferred) $('policy').value = preferred.name;
    else { $('policy').value = ''; }
    $('host').value = config.defaults.host; $('duration').value = config.defaults.episode_seconds;
    $('prompt').value = config.defaults.instruction;
    const defaults = config.default_views || config.cameras;
    for (let index = 0; index < Math.min(2, config.cameras.length); index++) {
      const saved = sessionStorage.getItem(`s2r-${isEvaluation() ? 'paired' : 'real'}-camera-${index}`);
      addCamera(config.cameras.includes(saved) ? saved : defaults[index], index);
    }
    for (const [label, value] of [['Arm port', config.follower_port], ...Object.entries(config.camera_devices).map(([name,path]) => [`${name} camera`, path])]) {
      const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = label; dd.textContent = value || 'Not configured'; $('hardware').append(dt, dd);
    }
    policyChanged(); await poll();
    setInterval(poll, 250);
    setInterval(() => { if (live) api('/api/heartbeat', {}).catch(() => {}); }, 700);
  } catch (error) { localError = error.message; render(); }
}
initialize();
