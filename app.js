const elReload = document.getElementById('btnReload');
const elStart  = document.getElementById('btnStart');
const elStop   = document.getElementById('btnStop');
const elStamp  = document.getElementById('stamp');
const elTicker = document.getElementById('ticker');

const rainChart = echarts.init(document.getElementById('rainChart'));
const sinrChart = echarts.init(document.getElementById('sinrChart'));
const dutyChart = echarts.init(document.getElementById('dutyChart'));
const mosChart  = echarts.init(document.getElementById('mosChart'));

let rawData = null, S = null, N = 0;
let playTimer = null, idx = 0;
let buf = { rain:[], sinr:[], duty:[], mos:[] };

// 用于“底部5条记录”的时间推导
let baseTs = null;         // 第一条数据的时间戳(ms)
let stepMsTicker = 15000;  // 每点间隔（优先用 meta.slot_seconds，否则从数据推断）

const COLORS = ['#8AB4FF', '#7FDBCA', '#F6D06F', '#EE7A7A']; // 高对比色

function baseOption(title, xType, yFmt='{value}') {
  return {
    color: [COLORS[0]],
    title: { text: title, left: 12, top: 8, textStyle:{ color:'#e9eef7', fontSize:14 } },
    grid:  { left:48, right:16, top:48, bottom:36 },
    xAxis: { 
      type:xType,
      axisLabel:{ color:'#9fb3c8' },
      axisLine:{ lineStyle:{ color:'#9fb3c8' } },
      splitLine:{ show:false },       // ← 关闭纵向参考线
      scale: true
    },
    yAxis: { 
      type:'value',
      axisLabel:{ color:'#9fb3c8', formatter:yFmt },
      splitLine:{ lineStyle:{ color:'#1a2242' } },
      scale: true
    },
    tooltip: { trigger:'axis' },
    animation: false,
    series: [{ type:'line', showSymbol:false, smooth:true, data: [] }]
  };
}

function initCharts() {
  const xType = (window.X_AXIS_MODE === 'time') ? 'time' : 'value';
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

function fmtTs(ms) {
  const d = new Date(ms);
  const pad = (n)=> String(n).padStart(2,'0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
const f2 = (x)=> (isFinite(x) ? Number(x).toFixed(3) : 'NaN');

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

  // 时间信息：优先 meta.slot_seconds；否则从前两个时间戳推断
  baseTs = Number(rain?.[0]?.[0] ?? Date.now());
  if (rawData.meta && Number(rawData.meta.slot_seconds)) {
    stepMsTicker = Number(rawData.meta.slot_seconds) * 1000;
  } else if (rain.length >= 2) {
    const diff = Number(rain[1][0]) - Number(rain[0][0]);
    stepMsTicker = (isFinite(diff) && diff > 0) ? diff : 15000;
  } else {
    stepMsTicker = 15000;
  }

  S = {
    rain_xy: toXY(rain, mode),
    sinr_xy: toXY(sinr, mode),
    duty_xy: toXY(duty, mode),
    mos_xy:  toXY(mos,  mode)
  };

  idx = 0;
  buf = { rain:[], sinr:[], duty:[], mos:[] };
  initCharts();

  elStamp.textContent = `Loaded: ${N} samples · step=${stepMsTicker/1000}s · x-axis=${mode} · algo=${rawData.meta?.algo || ''}`;
  elStart.disabled = (N === 0);
  elStop.disabled  = true;

  // 初次载入后强制一次 resize
  setTimeout(() => { rainChart.resize(); sinrChart.resize(); dutyChart.resize(); mosChart.resize(); }, 0);
  updateTicker(-1); // 清空或显示初始
}

function paintAll(forceResize=false) {
  rainChart.setOption({ series: [{ data: buf.rain }] }, false, { replaceMerge: ['series'] });
  sinrChart.setOption({ series: [{ data: buf.sinr }] }, false, { replaceMerge: ['series'] });
  dutyChart.setOption({ series: [{ data: buf.duty }] }, false, { replaceMerge: ['series'] });
  mosChart .setOption({ series: [{ data: buf.mos  }] }, false, { replaceMerge: ['series'] });
  if (forceResize) { rainChart.resize(); sinrChart.resize(); dutyChart.resize(); mosChart.resize(); }
}

function updateTicker(curIndex) {
  // 显示“当前 + 前 4 个”的时间戳与四个值
  const series = rawData?.series || {};
  if (!series.rain) { elTicker.textContent = 'No data.'; return; }
  const last5 = [];
  for (let k = 0; k < 5; k++) {
    const i = curIndex - k;
    if (i < 0) break;
    const t  = baseTs + i * stepMsTicker;
    const rv = series.rain[i]?.[1];
    const sv = series.sinr[i]?.[1];
    const dv = series.duty[i]?.[1];
    const mv = series.mos [i]?.[1];
    last5.push(`${fmtTs(t)} | rain=${f2(rv)}, sinr=${f2(sv)} dB, duty=${f2(dv)}, mos=${f2(mv)}`);
  }
  if (last5.length === 0) {
    elTicker.textContent = 'Waiting for playback…';
  } else {
    elTicker.textContent = last5.reverse().join('   •   '); // 旧→新
  }
}

function startPlayback() {
  if (!rawData || N === 0) { elStamp.textContent = 'No data to play.'; return; }
  if (playTimer) clearInterval(playTimer);

  const tick = Math.max(20, Number(window.PLAY_TICK_MS || 500)); // 0.5s 默认
  idx = 0;
  buf = { rain:[], sinr:[], duty:[], mos:[] };
  paintAll(true);
  updateTicker(-1);

  let resizeCounter = 0;
  playTimer = setInterval(() => {
    buf.rain.push(S.rain_xy[idx]);
    buf.sinr.push(S.sinr_xy[idx]);
    buf.duty.push(S.duty_xy[idx]);
    buf.mos .push(S.mos_xy[idx]);

    paintAll(++resizeCounter % 8 === 0);
    updateTicker(idx);

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
window.addEventListener('resize', () => { rainChart.resize(); sinrChart.resize(); dutyChart.resize(); mosChart.resize(); });

// 初始化
initCharts();
loadData();
