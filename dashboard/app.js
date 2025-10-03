// ===== DOM =====
const elReload = document.getElementById('btnReload');
const elStart  = document.getElementById('btnStart');
const elPause  = document.getElementById('btnPause');
const elStamp  = document.getElementById('stamp');
const elTicker = document.getElementById('ticker');

// ===== ECharts init =====
const rainChart = echarts.init(document.getElementById('rainChart'));
const sinrChart = echarts.init(document.getElementById('sinrChart'));
const pesqChart = echarts.init(document.getElementById('pesqChart')); // <-- duty 改为 pesq
const mosChart  = echarts.init(document.getElementById('mosChart'));

function axisStyle() {
  return {
    axisLabel: { color: '#9fb3c8' },
    axisLine: { lineStyle: { color: '#9fb3c8' } },
    splitLine: { lineStyle: { color: '#1a2242' } },
    scale: true
  };
}
function baseOption(title, yName) {
  return {
    title: { text: title, left: 'center', textStyle: { color: '#cfe3ff', fontSize: 14 } },
    grid: { left: 56, right: 20, top: 36, bottom: 40 },
    animation: false,
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    xAxis: { ...axisStyle(), type: 'time', name: 'time' },
    yAxis: { ...axisStyle(), type: 'value', name: yName },
    series: [{ type: 'line', smooth: true, showSymbol: false, data: [] }]
  };
}

rainChart.setOption(baseOption('Rain (mm/h)', 'mm/h'));
sinrChart.setOption(baseOption('SINR (dB)', 'dB'));
pesqChart.setOption(baseOption('PESQ (score)', 'PESQ'));   // <-- 新标题
mosChart.setOption(baseOption('MOS (score)', 'MOS'));

// ===== Data & playback =====
let raw = null;          // { ts, meta, series }
let idx = 0;
let timer = null;

async function loadMetrics() {
  const res = await fetch('./data/metrics.json', { cache: 'no-store' });
  raw = await res.json();
  idx = 0;
  elStamp.textContent = `algo=${raw?.meta?.algo || '-'} · slot=${raw?.meta?.slot_seconds || '-'}s · n=${raw?.meta?.n || '-'}`;
  elTicker.textContent = 'Loaded metrics.json';
  // 预热一次渲染（空数据）
  render(0);
}

function sliceUpTo(arr, k) {
  // arr: [[ts_ms, val], ...]
  return arr.slice(0, Math.max(0, k)).map(([t, v]) => [t, v]);
}

function render(k) {
  if (!raw?.series) return;
  const S = raw.series;

  // 关键：使用 pesq 替换 duty
  const rain = S.rain || [];
  const sinr = S.sinr || [];
  const pesq = S.pesq || [];   // <-- 从 metrics.json 读取 pesq（已确认存在）
  const mos  = S.mos  || [];

  rainChart.setOption({ series: [{ data: sliceUpTo(rain, k) }] });
  sinrChart.setOption({ series: [{ data: sliceUpTo(sinr, k) }] });
  pesqChart.setOption({ series: [{ data: sliceUpTo(pesq, k) }] }); // <-- 第三幅图展示 pesq
  mosChart.setOption({  series: [{ data: sliceUpTo(mos,  k) }] });

  // 显示当前时间戳
  if (k > 0) {
    const t = rain[k-1]?.[0] ?? sinr[k-1]?.[0] ?? pesq[k-1]?.[0] ?? mos[k-1]?.[0];
    if (t) elTicker.textContent = (new Date(t)).toISOString();
  }
}

function start() {
  if (!raw?.series) return;
  if (timer) return;
  elStart.disabled = true;
  elPause.disabled = false;

  const n = raw.meta?.n || Math.max(
    raw.series.rain?.length || 0,
    raw.series.sinr?.length || 0,
    raw.series.pesq?.length || 0,
    raw.series.mos?.length  || 0
  );
  const stepMs = 50; // 播放速度（可调）
  timer = setInterval(() => {
    idx = Math.min(idx + 1, n);
    render(idx);
    if (idx >= n) pause();
  }, stepMs);
}

function pause() {
  if (timer) { clearInterval(timer); timer = null; }
  elStart.disabled = false;
  elPause.disabled = true;
}

elReload.addEventListener('click', loadMetrics);
elStart .addEventListener('click', start);
elPause .addEventListener('click', pause);

window.addEventListener('load', loadMetrics);
window.addEventListener('resize', () => {
  rainChart.resize(); sinrChart.resize(); pesqChart.resize(); mosChart.resize();
});
