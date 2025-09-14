const elReload = document.getElementById('btnReload');
const elStart  = document.getElementById('btnStart');
const elStop   = document.getElementById('btnStop');
const elStamp  = document.getElementById('stamp');

const rainChart = echarts.init(document.getElementById('rainChart'));
const sinrChart = echarts.init(document.getElementById('sinrChart'));
const dutyChart = echarts.init(document.getElementById('dutyChart'));
const mosChart  = echarts.init(document.getElementById('mosChart'));

let rawData   = null;   // whole JSON
let S         = null;   // series shortcut
let N         = 0;      // min length across 4 series
let playTimer = null;
let idx       = 0;

// ---------- helpers ----------
async function fetchJSON(url) {
  const r = await fetch(url + '?t=' + Date.now(), { cache:'no-store' });
  if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
  return await r.json();
}

function toXY(series, mode) {
  // series: [[ts,value], ...]
  if (!Array.isArray(series)) return [];
  if (mode === 'time') {
    // Use timestamp (ms) directly on x-axis
    return series.map(p => [Number(p[0]), Number(p[1])]);
  }
  // default: 'slot' => x = 1..N
  return series.map((p, i) => [i + 1, Number(p[1])]);
}

function baseOption(title, xType, yFmt='{value}') {
  return {
    title: { text: title, left:12, top:8, textStyle:{ color:'#e9eef7', fontSize:14 } },
    grid:  { left:48, right:16, top:48, bottom:36 },
    xAxis: { type: xType, axisLabel:{ color:'#9fb3c8' }, axisLine:{ lineStyle:{ color:'#9fb3c8' } } },
    yAxis: { type:'value', axisLabel:{ color:'#9fb3c8', formatter:yFmt }, splitLine:{ lineStyle:{ color:'#1a2242' } } },
    tooltip: { trigger:'axis' },
    series: [{ type:'line', showSymbol:false, smooth:true, data: [] }]
  };
}

function initCharts() {
  const xType = (window.X_AXIS_MODE === 'time') ? 'time' : 'value';
  rainChart.setOption(baseOption('Rain (mm/h)', xType));
  sinrChart.setOption(baseOption('SINR (dB)',   xType));
  dutyChart.setOption(baseOption('Duty Cycle',  xType, '{value}'));
  mosChart .setOption(baseOption('MOS',         xType, '{value}'));
}

// append one point to each chart efficiently
function appendOne(i) {
  const xr = S.rain_xy[i];
  const xs = S.sinr_xy[i];
  const xd = S.duty_xy[i];
  const xm = S.mos_xy[i];
  // appendData is efficient & prevents full re-render each time
  rainChart.appendData({ seriesIndex: 0, data: [xr] });
  sinrChart.appendData({ seriesIndex: 0, data: [xs] });
  dutyChart.appendData({ seriesIndex: 0, data: [xd] });
  mosChart .appendData({ seriesIndex: 0, data: [xm] });
}

// ---------- load & prepare ----------
async function loadData() {
  const url = window.METRICS_URL || './data/metrics.json';
  rawData = await fetchJSON(url);
  const series = rawData.series || {};
  const mode   = (window.X_AXIS_MODE === 'time') ? 'time' : 'slot';

  const rain = Array.isArray(series.rain) ? series.rain : [];
  const sinr = Array.isArray(series.sinr) ? series.sinr : [];
  const duty = Array.isArray(series.duty) ? series.duty : [];
  const mos  = Array.isArray(series.mos ) ? series.mos  : [];

  N = Math.min(rain.length, sinr.length, duty.length, mos.length);

  // Pre-convert to the final [x,y] pairs we want to plot (slot index or timestamp)
  S = {
    rain_xy: toXY(rain, mode),
    sinr_xy: toXY(sinr, mode),
    duty_xy: toXY(duty, mode),
    mos_xy:  toXY(mos,  mode),
  };

  idx = 0;
  initCharts();

  const algo = rawData.meta?.algo || '';
  elStamp.textContent = `Loaded: ${N} samples · x-axis=${mode} ${algo ? '· '+algo : ''}`;
  elStart.disabled = (N === 0);
  elStop.disabled  = true;
}

// ---------- playback ----------
function startPlayback() {
  if (!rawData || N === 0) { elStamp.textContent = 'No data to play.'; return; }
  if (playTimer) clearInterval(playTimer);

  const tick = Math.max(10, Number(window.PLAY_TICK_MS || 500)); // 0.5s default

  playTimer = setInterval(() => {
    appendOne(idx);
    idx += 1;
    elStamp.textContent = `Playing… ${idx}/${N}`;
    if (idx >= N) stopPlayback();
  }, tick);

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

// ---------- wire up ----------
elReload.addEventListener('click', loadData);
elStart .addEventListener('click', startPlayback);
elStop  .addEventListener('click', stopPlayback);
window.addEventListener('resize', ()=>{
  rainChart.resize(); sinrChart.resize(); dutyChart.resize(); mosChart.resize();
});

// init
initCharts();
loadData();
