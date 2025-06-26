let sr_data;

const task_types = [
  'avg_micro',
  'reasoning',
];

const task_type_to_name = {
  'avg_micro': 'the AitW General subset',
  'reasoning': 'the AitW Web Shopping subset',
}

const all_models = [
  'Set-of-Marks GPT-4V',
  'Set-of-Marks Gemini-1.5-Pro',
  'AppAgent GPT-4V',
  'AppAgent Gemini-1.5-Pro',
  'AutoUI Pre-trained',
  'CogAgent Pre-trained',
  'AutoUI Filtered BC Offline',
  'AutoUI DigiRL Offline',
  'AutoUI Filtered BC Online',
  'AutoUI DigiRL Online',
]

function calculateDifferences(sr_data, task_type) {
  const data = sr_data[task_type].sort((a, b) => a[0].localeCompare(b[0]));

  return data.map(item => {
    const name = item[0];
    const values = item.slice(1);
    return [name, ...values];
  });
}

function formatWithSign(num) {
  const fixedNum = parseFloat(num).toFixed(2);
  return (Math.sign(num) === 1 ? "+" : "") + fixedNum;
}

let last_task_type = 'avg_micro';
let last_names_to_keep = all_models;
let chart = null;

function createChart(task_type, namesToKeep) {
  const sr_with_diff = calculateDifferences(sr_data, task_type);
  const sr_with_diff_filtered = sr_with_diff.filter(item => namesToKeep.includes(item[0]));

  // Calculate the total length of each bar
  const sr_with_totals = sr_with_diff_filtered.map(item => {
    const totalLength = item.slice(1).reduce((acc, val) => acc + val, 0);
    return { name: item[0], values: item.slice(1), totalLength };
  });

  // Sort the data by the total length of the bars in descending order
  sr_with_totals.sort((a, b) => b.totalLength - a.totalLength);

  const labels = sr_with_totals.map(item => item.name);
  const failure_modes_data = sr_with_totals.map(item => item.values);

  const colors = [
    'rgba(255, 99, 132, 0.5)',
    'rgba(54, 162, 235, 0.5)',
    'rgba(255, 206, 86, 0.5)',
    'rgba(75, 192, 192, 0.5)',
    'rgba(153, 102, 255, 0.5)',
    'rgba(255, 159, 64, 0.5)'
  ];

  const datasets = [];
  const failure_mode_names = [
    "Fail to recover from mistakes",
    "Fail to click on the right link or fail to type",
    "Fail to take reasonable attempts at all",
    "Quit or press HOME early",
    "Stops at wrong but relevant page",
    "Technical issues",
    "Task success"
  ];

  for (let i = 0; i < 6; i++) { // Assuming 6 failure modes
    datasets.push({
      label: failure_mode_names[i],
      data: failure_modes_data.map(data => data[i]),
      backgroundColor: colors[i % colors.length],
      stack: 'Stack 0'
    });
  }

  const title_text = 'Failure weight of ' + task_type_to_name[task_type];

  if (chart) {
    chart.destroy();
  }

  const ctx = document.getElementById('chart-sr-w-feedback');
  chart = new Chart(ctx, {
    plugins: [ChartDataLabels],
    type: 'bar',
    data: {
      labels: labels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
      },
      indexAxis: 'y',
      scales: {
        y: {
          beginAtZero: true,
          stacked: true,
          title: {
            display: true,
            text: 'Evaluated VLM',
            font: {
              size: 14,
            }
          }
        },
        x: {
          grace: 20,
          stacked: true,
          title: {
            display: true,
            text: 'Failure Percent in All Trajectories (%)',
            font: {
              size: 14,
            }
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: title_text,
          font: function (context) {
            var width = context.chart.width;
            var size = Math.round(width / 32);
            size = Math.min(size, 16);

            return {
              weight: 'bold',
              size: size
            };
          }
        },
        legend: {
          display: true,
          labels: {
            usePointStyle: true,
            font: {
              size: 10,
            }
          },
          align: 'center',
          position: 'bottom'
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              label += context.formattedValue + '%';
              return label;
            },
            footer: function (tooltipItems) {
              let diff = 0;
              tooltipItems.forEach(function (tooltipItem) {
                diff += tooltipItem.raw;
              });
              return 'Total: ' + diff.toFixed(2) + '%';
            }
          }
        },
        datalabels: {
          display: false // Disable data labels
        }
      }
    }
  });
}



task_types.forEach(task_type => {
  const btn = document.getElementById(task_type);

  btn.addEventListener('click', () => {
    document.querySelectorAll('.btn-group.task-selector .btn.active')
      .forEach(active => {
        active.classList.remove('active');
      });

    btn.classList.add('active');
    last_task_type = task_type;
    createChart(task_type, last_names_to_keep);
  });
});

document.addEventListener('DOMContentLoaded', function () {
  fetch('website/data/sr_data.json')
    .then(response => response.json())
    .then(data => {
      sr_data = data;
      createChart(last_task_type, last_names_to_keep);

      document.getElementById("visualize-feedback-sr-no-diff-open-close").addEventListener("click", function () {
        createChart(last_task_type, all_models);
      });

      document.getElementById("visualize-feedback-sr-sift-rlhf").addEventListener("click", function () {
        createChart(last_task_type, [
          'Set-of-Marks GPT-4V',
          'Set-of-Marks Gemini-1.5-Pro',
          'AppAgent GPT-4V',
          'AppAgent Gemini-1.5-Pro',
          'AutoUI Pre-trained',
          'CogAgent Pre-trained',
          'AutoUI Filtered BC Offline',
          'AutoUI DigiRL Offline',
          'AutoUI Filtered BC Online',
          'AutoUI DigiRL Online',
        ]);
      });

      document.getElementById("visualize-feedback-sr-gpt-4-self").addEventListener("click", function () {
        createChart(last_task_type, [
          'Set-of-Marks GPT-4V',
          'Set-of-Marks Gemini-1.5-Pro',
          'AppAgent GPT-4V',
          'AppAgent Gemini-1.5-Pro',
          'AutoUI Pre-trained',
          'CogAgent Pre-trained',
          'AutoUI Filtered BC Offline',
          'AutoUI DigiRL Offline',
          'AutoUI Filtered BC Online',
          'AutoUI DigiRL Online',
        ]);
      });
    });
});
