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

const COLORS = ['#8AB4FF', '#7FDBCA', '#F6D06F', '#EE7A7A']; // 高对比

function baseOption(title, xType, yFmt='{value}') {
  return {
    color: [COLORS[0]], // 每个图只有一条线，各自颜色在 init 时改
    title: { text: title, left: 12, top: 8, textStyle:{ color:'#e9eef7', fontSize:14 } },
    grid:  { left:48, right:16, top:48, bottom:36 },
    xAxis: { type:xType, axisLabel:{ color:'#9fb3c8' }, axisLine:{ lineStyle:{ color:'#9fb3c8' } }, splitLine:{ lineStyle:{ color:'#1a2242' } }, scale: true },
    yAxis: { type:'value', axisLabel:{ color:'#9fb3c8', formatter:yFmt }, splitLine:{ lineStyle:{ color:'#1a2242' } }, scale: true },
    tooltip: { trigger:'axis' },
    animation: false,
    series: [{ type:'line', showSymbol:false, smooth:true, data: [] }]
  };
}

function initCharts() {
  const xType = (window.X_AXIS_MODE === 'time') ? 'time' : 'value';
  // 四个图分别指定颜色，避免主题差异
  const optsR = baseOption('Rain (mm/h)', xType);  optsR.color = [COLORS[0]];
  const optsS = baseOption('SINR (dB)',   xType);  optsS.color = [COLORS[1]];
  const optsD = baseOption('Duty Cycle',  xType);  optsD.color = [COLORS[2]];
  const optsM = baseOption('MOS',         xType);  optsM.color = [COLORS[3]];
  rainChart.setOption(optsR, true);
  sinrChart.setOption(optsS, true);
  dutyChart.setOption(optsD, true);
  mosChart .setOption(optsM, true);
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

  // 首次载入后强制一次 resize，避免某些浏览器初始宽高异常
  setTimeout(() => { rainChart.resize(); sinrChart.resize(); dutyChart.resize(); mosChart.resize(); }, 0);
}

function paintAll(forceResize=false) {
  const optionR = { series: [{ data: buf.rain }] };
  const optionS = { series: [{ data: buf.sinr }] };
  const optionD = { series: [{ data: buf.duty }] };
  const optionM = { series: [{ data: buf.mos  }] };

  // replaceMerge 确保按我们提供的 series 完整替换，避免内部状态干扰
  rainChart.setOption(optionR, false, { replaceMerge: ['series'] });
  sinrChart.setOption(optionS, false, { replaceMerge: ['series'] });
  dutyChart.setOption(optionD, false, { replaceMerge: ['series'] });
  mosChart .setOption(optionM, false, { replaceMerge: ['series'] });

  if (forceResize) {
    rainChart.resize(); sinrChart.resize(); dutyChart.resize(); mosChart.resize();
  }
}

function startPlayback() {
  if (!rawData || N === 0) { elStamp.textContent = 'No data to play.'; return; }
  if (playTimer) clearInterval(playTimer);

  const tick = Math.max(20, Number(window.PLAY_TICK_MS || 500)); // 0.5s 默认
  idx = 0;
  buf = { rain:[], sinr:[], duty:[], mos:[] };
  paintAll(true);

  let resizeCounter = 0;
  playTimer = setInterval(() => {
    buf.rain.push(S.rain_xy[idx]);
    buf.sinr.push(S.sinr_xy[idx]);
    buf.duty.push(S.duty_xy[idx]);
    buf.mos .push(S.mos_xy[idx]);

    // 每次刷新 series；每隔一段时间强制 resize 一次，保证可见性
    paintAll(++resizeCounter % 8 === 0);

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
