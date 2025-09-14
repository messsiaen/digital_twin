const elRun   = document.getElementById('runSelect');
const elReload= document.getElementById('btnReload');
const elStart = document.getElementById('btnStart');
const elStop  = document.getElementById('btnStop');
const elStamp = document.getElementById('stamp');

const rainChart = echarts.init(document.getElementById('rainChart'));
const sinrChart = echarts.init(document.getElementById('sinrChart'));
const dutyChart = echarts.init(document.getElementById('dutyChart'));
const mosChart  = echarts.init(document.getElementById('mosChart'));

let playTimer = null;
let runData   = null;
let playIdx   = 0;
let seriesBuf = { rain:[], sinr:[], duty:[], mos:[] };

function lineOpts(title, data, yfmt='{value}') {
  return {
    title: { text: title, left: 12, top: 8, textStyle:{ color:'#e9eef7', fontSize:14 } },
    grid: { left: 48, right: 16, top: 48, bottom: 36 },
    xAxis: { type:'time', axisLabel:{ color:'#9fb3c8' }, axisLine:{ lineStyle:{ color:'#9fb3c8' } } },
    yAxis: { type:'value', axisLabel:{ color:'#9fb3c8', formatter:yfmt }, splitLine:{ lineStyle:{ color:'#1a2242' } } },
    tooltip: { trigger:'axis' },
    series: [{ type:'line', showSymbol:false, smooth:true, data }]
  };
}

function drawAll() {
  rainChart.setOption(lineOpts('Rain (mm/h)', seriesBuf.rain));
  sinrChart.setOption(lineOpts('SINR (dB)',   seriesBuf.sinr));
  dutyChart.setOption(lineOpts('Duty Cycle',  seriesBuf.duty, '{value}'));
  mosChart.setOption (lineOpts('MOS',         seriesBuf.mos,  '{value}'));
}

async function fetchJSON(url) {
  const r = await fetch(url + '?t=' + Date.now(), { cache: 'no-store' });
  if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
  return await r.json();
}

// Load runs list into <select>
async function loadRuns() {
  try {
    elStamp.textContent = 'Loading runs…';
    const idx = await fetchJSON(window.RUNS_INDEX_URL);
    elRun.innerHTML = '<option value="">Select a run…</option>';
    (idx.runs || []).sort((a,b)=> (b.created_at||'').localeCompare(a.created_at||''))
      .forEach(r => {
        const o = document.createElement('option');
        o.value = r.id;
        o.text  = `${r.id} · ${r.algo || 'algo'} · ${r.created_at || ''}`;
        elRun.appendChild(o);
      });
    elStamp.textContent = `Loaded ${ (idx.runs||[]).length } run(s)`;
  } catch (e) {
    elStamp.textContent = 'Failed to load runs.';
    // console.error(e);
  }
}

async function loadRunDetail(runId) {
  const url = window.RUNS_BASE_URL + encodeURIComponent(runId) + '.json';
  runData = await fetchJSON(url);
  elStamp.textContent = `Run loaded: ${runData.id} (${runData.algo}), samples=${runData.n}`;
}

function resetPlayback() {
  playIdx = 0;
  seriesBuf = { rain:[], sinr:[], duty:[], mos:[] };
  drawAll();
}

function startPlayback() {
  if (!runData) { elStamp.textContent = 'Select a run first.'; return; }
  if (playTimer) clearInterval(playTimer);
  resetPlayback();
  const stepMs = Math.max(50, Math.round((runData.step_ms || 1000) / (window.PLAY_SPEED || 1)));
  const t0     = Date.now();

  playTimer = setInterval(() => {
    if (playIdx >= runData.n) { stopPlayback(); return; }
    const ts = t0 + playIdx * stepMs;

    // Append one point per tick
    seriesBuf.rain.push([ts, runData.rain[playIdx]]);
    seriesBuf.sinr.push([ts, runData.sinr_db[playIdx]]);
    seriesBuf.duty.push([ts, runData.duty[playIdx]]);
    seriesBuf.mos .push([ts, runData.mos[playIdx]]);

    drawAll();
    elStamp.textContent = `Playing ${runData.id} · ${playIdx+1}/${runData.n}`;
    playIdx += 1;
  }, stepMs);

  elStart.disabled = true;
  elStop.disabled  = false;
}

function stopPlayback() {
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
  elStart.disabled = false;
  elStop.disabled  = true;
  elStamp.textContent = 'Stopped.';
}

elReload.addEventListener('click', loadRuns);
elStart .addEventListener('click', startPlayback);
elStop  .addEventListener('click', stopPlayback);
elRun   .addEventListener('change', async (e) => {
  const id = e.target.value;
  if (!id) return;
  await loadRunDetail(id);
  resetPlayback();
});

window.addEventListener('resize', () => { rainChart.resize(); sinrChart.resize(); dutyChart.resize(); mosChart.resize(); });

// initial
loadRuns();
drawAll();
