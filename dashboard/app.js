// ===== DOM =====
const elReload = document.getElementById('btnReload');
const elStart  = document.getElementById('btnStart');
const elPause  = document.getElementById('btnPause');
const elStamp  = document.getElementById('stamp');
const elTicker = document.getElementById('ticker');

// ===== ECharts init =====
const rainChart   = echarts.init(document.getElementById('rainChart'));
const energyChart = echarts.init(document.getElementById('energyChart'));
const socChart    = echarts.init(document.getElementById('socChart'));
const wptChart    = echarts.init(document.getElementById('wptChart'));

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

rainChart.setOption(baseOption('Rain Attenuation (dB)', 'dB'));
energyChart.setOption(baseOption('Energy Harvest (mW)', 'mW'));
socChart.setOption(baseOption('State of Charge (%)', '%'));
wptChart.setOption(baseOption('WPT Power (mW)', 'mW'));

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

  const rain   = S.rain_attenuation || [];
  const energy = S.energy_harvest || [];
  const soc    = S.soc || [];
  const wpt    = S.wpt_power || [];

  rainChart.setOption({ series: [{ data: sliceUpTo(rain, k) }] });
  energyChart.setOption({ series: [{ data: sliceUpTo(energy, k) }] });
  socChart.setOption({ series: [{ data: sliceUpTo(soc, k) }] });
  wptChart.setOption({ series: [{ data: sliceUpTo(wpt, k) }] });

  // 显示当前时间戳
  if (k > 0) {
    const t = rain[k-1]?.[0] ?? energy[k-1]?.[0] ?? soc[k-1]?.[0] ?? wpt[k-1]?.[0];
    if (t) elTicker.textContent = (new Date(t)).toISOString();
  }
}

function start() {
  if (!raw?.series) return;
  if (timer) return;
  elStart.disabled = true;
  elPause.disabled = false;

  const n = raw.meta?.n || Math.max(
    raw.series.rain_attenuation?.length || 0,
    raw.series.energy_harvest?.length || 0,
    raw.series.soc?.length || 0,
    raw.series.wpt_power?.length  || 0
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
  rainChart.resize(); energyChart.resize(); socChart.resize(); wptChart.resize();
});
