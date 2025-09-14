const elStamp   = document.getElementById('stamp');
const rainChart = echarts.init(document.getElementById('rainChart'));
const sinrChart = echarts.init(document.getElementById('sinrChart'));
const dutyChart = echarts.init(document.getElementById('dutyChart'));
const mosChart  = echarts.init(document.getElementById('mosChart'));

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

async function refresh() {
  try {
    const url = (window.METRICS_URL || './data/metrics.json') + '?t=' + Date.now();
    const r   = await fetch(url, { cache:'no-store' });
    if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
    const m   = await r.json();

    elStamp.textContent = `Updated: ${new Date(m.ts || Date.now()).toLocaleString()}`;

    const s = (m.series || {});
    rainChart.setOption(lineOpts('Rain (mm/h)', s.rain || []));
    sinrChart.setOption(lineOpts('SINR (dB)',   s.sinr || []));
    dutyChart.setOption(lineOpts('Duty Cycle',  s.duty || [], '{value}'));
    mosChart.setOption (lineOpts('MOS',         s.mos  || [], '{value}'));
  } catch (e) {
    elStamp.textContent = 'Failed to fetch metrics.json';
    // console.error(e);
  }
}

refresh();
setInterval(refresh, window.REFRESH_MS || 5000);
window.addEventListener('resize', () => {
  rainChart.resize(); sinrChart.resize(); dutyChart.resize(); mosChart.resize();
});
