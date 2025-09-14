const elReload = document.getElementById('btnReload');
const elStart  = document.getElementById('btnStart');
const elStop   = document.getElementById('btnStop');
const elStamp  = document.getElementById('stamp');

const rainChart = echarts.init(document.getElementById('rainChart'));
const sinrChart = echarts.init(document.getElementById('sinrChart'));
const dutyChart = echarts.init(document.getElementById('dutyChart'));
const mosChart  = echarts.init(document.getElementById('mosChart'));

let rawData   = null;   // the whole JSON
let playTimer = null;   // setInterval handle
let idx       = 0;      // current index in arrays
let stepMs    = 1000;   // inferred step (from timestamps)
let buf       = { rain:[], sinr:[], duty:[], mos:[] }; // incremental series

function lineOpts(title, data, yfmt='{value}') {
  return {
    title: { text: title, left:12, top:8, textStyle:{ color:'#e9eef7', fontSize:14 } },
    grid:  { left:48, right:16, top:48, bottom:36 },
    xAxis: { type:'time', axisLabel:{ color:'#9fb3c8' }, axisLine:{ lineStyle:{ color:'#9fb3c8' } } },
    yAxis: { type:'value', axisLabel:{ color:'#9fb3c8', formatter:yfmt }, splitLine:{ lineStyle:{ color:'#1a2242' } } },
    tooltip: { trigger:'axis' },
    series: [{ type:'line', showSymbol:false, smooth:true, data }]
  };
}

function drawAll(){
  rainChart.setOption(lineOpts('Rain (mm/h)', buf.rain));
  sinrChart.setOption(lineOpts('SINR (dB)',   buf.sinr));
  dutyChart.setOption(lineOpts('Duty Cycle',  buf.duty, '{value}'));
  mosChart.setOption (lineOpts('MOS',         buf.mos,  '{value}'));
}

async function fetchJSON(url) {
  const r = await fetch(url + '?t=' + Date.now(), { cache:'no-store' });
  if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
  return await r.json();
}

function resetPlayback(){
  idx = 0;
  buf = { rain:[], sinr:[], duty:[], mos:[] };
  drawAll();
}

function inferStepMs(series){
  // series like [[ts,value], ...]
  if (!series || series.length < 2) return 1000;
  const d = Math.max(1, Number(series[1][0]) - Number(series[0][0]));
  return isFinite(d) ? d : 1000;
}

async function loadData(){
  try {
    const url = window.METRICS_URL || './data/metrics.json';
    rawData = await fetchJSON(url);

    // Basic checks
    const S = rawData.series || {};
    const n = Math.min(S.rain?.length||0, S.sinr?.length||0, S.duty?.length||0, S.mos?.length||0);
    stepMs  = inferStepMs(S.rain);
    resetPlayback();

    elStamp.textContent = `Loaded: ${n} samples · step ≈ ${stepMs} ms · ${rawData.meta?.algo||''}`;
    elStart.disabled = (n === 0);
    elStop.disabled  = true;
  } catch (e) {
    elStamp.textContent = 'Failed to load metrics.json';
    // console.error(e);
  }
}

function startPlayback(){
  if (!rawData) { elStamp.textContent = 'Load data first.'; return; }
  const S = rawData.series || {};
  const n = Math.min(S.rain?.length||0, S.sinr?.length||0, S.duty?.length||0, S.mos?.length||0);
  if (n === 0) { elStamp.textContent = 'No data to play.'; return; }

  if (playTimer) clearInterval(playTimer);
  resetPlayback();

  const speed = Math.max(1, Number(window.PLAY_SPEED || 1)); // points per tick
  let tickMs  = Math.max(30, Math.round(stepMs / speed));

  playTimer = setInterval(() => {
    // push one point
    buf.rain.push(S.rain[idx]);
    buf.sinr.push(S.sinr[idx]);
    buf.duty.push(S.duty[idx]);
    buf.mos .push(S.mos[idx]);

    drawAll();
    elStamp.textContent = `Playing… ${idx+1}/${n}`;

    idx += 1;
    if (idx >= n) stopPlayback();
  }, tickMs);

  elStart.disabled = true;
  elStop.disabled  = false;
}

function stopPlayback(){
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
  elStart.disabled = false;
  elStop.disabled  = true;
  elStamp.textContent = 'Stopped.';
}

elReload.addEventListener('click', loadData);
elStart .addEventListener('click', startPlayback);
elStop  .addEventListener('click', stopPlayback);
window.addEventListener('resize', ()=>{ rainChart.resize(); sinrChart.resize(); dutyChart.resize(); mosChart.resize(); });

// init
loadData();
drawAll();
