/* website/javascript/sr_vs_k_dual.js
 * 新增：逐点延时动画
 */

const COLOR = {
  'CIFAR-10 No Frozen'      : '#002b5b',
  'CIFAR-10 Frozen'         : '#7aa2d2',
  'Tiny-ImageNet No Frozen' : '#d90429',
  'Tiny-ImageNet Frozen'    : '#ffb5b5'
};

const SUBSETS = {
  left : ['CIFAR-10 No Frozen', 'Tiny-ImageNet No Frozen'],
  right: [
    'CIFAR-10 No Frozen', 'CIFAR-10 Frozen',
    'Tiny-ImageNet No Frozen', 'Tiny-ImageNet Frozen'
  ]
};

async function main () {
  const res = await fetch('website/data/sr_vs_k_series_2.json');
  const all = await res.json();
  const map = new Map(all.map(s => [s.label, s]));

  createChart('chart-sr-no-frozen', SUBSETS.left .map(k => map.get(k)));
  createChart('chart-sr-all',       SUBSETS.right.map(k => map.get(k)));
}

function createChart (canvasId, seriesArr) {
  /* ---------- 配色 & 基本样式 ---------- */
  seriesArr.forEach(s => {
    s.borderColor     = COLOR[s.label];
    s.backgroundColor = hexToRGBA(COLOR[s.label], 0.35);
    s.tension         = 0.3;
    s.pointRadius     = 3;
    s.borderWidth     = 2;
  });

  /* ---------- 逐点延时动画 ---------- */
  const maxPts          = Math.max(...seriesArr.map(s => s.data.length));
  const totalDuration   = 8000;                       // 整条线拉完的总时长 (ms)
  const delayPerPoint   = totalDuration / maxPts;     // 每个点的延时

  const animConf = axis => ({
    type     : 'number',
    easing   : 'easeInOutQuart',
    duration : delayPerPoint,
    from     : Number.NaN,            // 前一帧无穷小，避免折线瞬间出现
    delay(ctx) {
      if (ctx.type !== 'data' || ctx[`${axis}Started`]) return 0;
      ctx[`${axis}Started`] = true;
      return ctx.index * delayPerPoint;
    }
  });

  new Chart(document.getElementById(canvasId), {
    type : 'line',
    data : {
      labels   : seriesArr[0].x,
      datasets : seriesArr
    },
    options : {
      responsive          : true,
      maintainAspectRatio : false,
      animation : {                // 👈 加入动画配置
        x: animConf('x'),
        y: animConf('y')
      },
      scales : {
        x : { title: {display:true, text:'Time Step'} },
        y : { title: {display:true, text:'Acc. (%)'},
              suggestedMin:0, suggestedMax:100 }
      },
      plugins : {
        legend : { position:'bottom', labels:{usePointStyle:true, font:{size:10}} },
        tooltip: {
          mode:'index', intersect:false,
          callbacks:{
            label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`
          }
        }
      }
    }
  });
}

/* 工具：#rrggbb 转 rgba */
function hexToRGBA (hex, alpha = 1) {
  hex = hex.replace('#','');
  if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
  const num = parseInt(hex, 16);
  const r = (num >> 16) & 255,
        g = (num >>  8) & 255,
        b =  num        & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

main();