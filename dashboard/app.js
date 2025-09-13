const elStamp = document.getElementById('stamp');
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
    const r = await fetch(window.METRICS_URL + '?t=' + Date.now(), { cache: 'no-store' });
    if (!r.ok) throw new Error(r.statusText);
    const m = await r.json();

    elStamp.textContent = `更新：${new Date(m.ts).toLocaleString()}`;

    const series = m.series || {};
    rainChart.setOption(lineOpts('Rain (mm/h)', series.rain || []));
    sinrChart.setOption(lineOpts('SINR (dB)', series.sinr || []));
    dutyChart.setOption(lineOpts('Duty', series.duty || [], '{value}'));
    mosChart.setOption (lineOpts('MOS',  series.mos  || [], '{value}'));
  } catch (e) {
    elStamp.textContent = '数据拉取失败，稍后重试…';
  }
}

refresh();
setInterval(refresh, window.REFRESH_MS || 5000);
window.addEventListener('resize', () => { rainChart.resize(); sinrChart.resize(); dutyChart.resize(); mosChart.resize(); });
