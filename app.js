const elReload = document.getElementById('btnReload');
const elStart  = document.getElementById('btnStart');
const elStop   = document.getElementById('btnStop');
const elStamp  = document.getElementById('stamp');

const rainChart = echarts.init(document.getElementById('rainChart'));
const sinrChart = echarts.init(document.getElementById('sinrChart'));
const dutyChart = echarts.init(document.getElementById('dutyChart'));
const mosChart  = echarts.init(document.getElementById('mosChart'));

let rawData = null, S = null, N = 0;
let playTimer = null, idx = 0;
let buf = { rain:[], sinr:[], duty:[], mos:[] };

function baseOption(title, xType, yFmt='{value}') {
  return {
    title: { text: title, left:12, top:8, textStyle:{ color:'#e9eef7', fontSize:14 } },
    grid:  { left:48, right:16, top:48, bottom:36 },
    xAxis: { type:xType, axisLabel:{ color:'#9fb3c8' }, axisLine:{ lineStyle:{ color:'#9fb3c8' } } },
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
  // 调试：若容器高度为 0，则会看不到图
  console.log('sizes', {
    rain: rainChart.getWidth()+'x'+rainChart.getHeight(),
    sinr: sinrChart.getWidth()+'x'+sinrChart.getHeight()
  });
}

async function fetchJSON(url) {
  const r = await fetch(url + '?t=' + Date.now(), { cache:'no-store' });
  if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
  return await r.json();
}

function toXY(series, mode) {
  if (!Array.isArray(series)) return [];
  if (mode === 'time') return series.map(p => [Number(p[0]), Number(p[1])]); // ts, value
  return series.map((p,i) => [i+1, Number(p[1])]);                            // slotIndex, value
}

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

  S = {
    rain_xy: toXY(rain, mode),
    sinr_xy: toXY(sinr, mode),
    duty_xy: toXY(duty, mode),
    mos_xy:  toXY(mos,  mode)
  };

  idx = 0;
  buf = { rain:[], sinr:[], duty:[], mos:[] };
  initCharts();

  elStamp.textContent = `Loaded: ${N} samples · x-axis=${mode} · algo=${rawData.meta?.algo || ''}`;
  elStart.disabled = (N === 0);
  elStop.disabled  = true;
}

function paintAll() {
  rainChart.setOption({ series: [{ data: buf.rain }] });
  sinrChart.setOption({ series: [{ data: buf.sinr }] });
  dutyChart.setOption({ series: [{ data: buf.duty }] });
  mosChart .setOption({ series: [{ data: buf.mos  }] });
}

function startPlayback() {
  if (!rawData || N === 0) { elStamp.textContent = 'No data to play.'; return; }
  if (playTimer) clearInterval(playTimer);

  const tick = Math.max(20, Number(window.PLAY_TICK_MS || 500)); // 0.5s 默认
  idx = 0;
  buf = { rain:[], sinr:[], duty:[], mos:[] };
  paintAll();

  playTimer = setInterval(() => {
    buf.rain.push(S.rain_xy[idx]);
    buf.sinr.push(S.sinr_xy[idx]);
    buf.duty.push(S.duty_xy[idx]);
    buf.mos .push(S.mos_xy[idx]);

    paintAll();
    elStamp.textContent = `Playing… ${idx+1}/${N}`;

    idx += 1;
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

elReload.addEventListener('click', loadData);
elStart .addEventListener('click', startPlayback);
elStop  .addEventListener('click', stopPlayback);

window.addEventListener('resize', () => {
  rainChart.resize(); sinrChart.resize(); dutyChart.resize(); mosChart.resize();
});

// 初始化
initCharts();
loadData();
