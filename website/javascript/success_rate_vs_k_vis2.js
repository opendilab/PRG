let sr_vs_k_series;
let chart = null;
let isUpdatingChart = false; // Flag to check if chart is being updated interactively

const color_Tableau20 = ['#F28E2B', '#A0CBE8', '#FFBE7D', '#59A14F', '#8CD17D', '#B6992D', '#F1CE63', '#499894', '#86BCB6', '#E15759', '#FF9D9A', '#79706E', '#BAB0AC', '#D37295', '#FABFD2', '#B07AA1', '#D4A6C8', '#9D7660', '#D7B5A6'];

function hexToRGBA(hex, alpha = 1) {
    hex = hex.replace('#', '');
    if (hex.length === 3) {
        hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    }
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

const default_order = [
    "DigiRL Online Learning",
    "DigiRL Online Frozen"
];

const all_models = [
    "DigiRL Online Learning",
    "DigiRL Online Frozen",
    "DigiRL Offline Frozen"
];

function createChart(subsetNames, annotate_model_name = true) {
    if (chart) {
        chart.destroy();
    }
    isUpdatingChart = true; // Set the flag to indicate the chart is being updated

    const ctx = document.getElementById('chart-sr-vs-k2');
    const sr_vs_k_series_subset = sr_vs_k_series.filter(series => subsetNames.includes(series.label));

    sr_vs_k_series_subset.forEach((series, index) => {
        series.borderColor = color_Tableau20[index];
        series.backgroundColor = hexToRGBA(color_Tableau20[index], 0.5);
        series.tension = 0.4;
    });

    const totalDuration = 12000; // Keep total duration fixed for consistency
    const maxDataPoints = Math.max(...sr_vs_k_series_subset.map(series => series.data.length));
    const delayBetweenPoints = totalDuration / maxDataPoints; // Now based on maximum data points in any subset

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: 12 }, (_, i) => i * 8),
            datasets: sr_vs_k_series_subset
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time elapsed (hours)',
                        font: {
                            size: 14,
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Success Rate every 16 trajectories',
                        font: {
                            size: 14,
                        }
                    }
                }
            },
            animation: {
                x: {
                    type: 'number',
                    easing: 'easeInOutQuart',
                    duration: delayBetweenPoints,
                    from: NaN,
                    delay(ctx) {
                        if (ctx.type !== 'data' || ctx.xStarted) {
                            return 0;
                        }
                        ctx.xStarted = true;
                        return ctx.index * delayBetweenPoints;
                    }
                },
                y: {
                    type: 'number',
                    easing: 'easeInOutQuart',
                    duration: delayBetweenPoints,
                    from: NaN,
                    delay(ctx) {
                        if (ctx.type !== 'data' || ctx.yStarted) {
                            return 0;
                        }
                        ctx.yStarted = true;
                        return ctx.index * delayBetweenPoints;
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        usePointStyle: true,
                        font: {
                            size: 10,
                        }
                    },
                    align: 'center',
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    position: 'nearest',
                    intersect: false,
                    itemSort: function (a, b) {
                        return b.raw - a.raw;
                    },
                    callbacks: {
                        title: function (context) {
                            return 'Number of trajectories trained on: ' + context[0].label;
                        },
                        label: function (context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });

    setTimeout(() => isUpdatingChart = false, 500); // Reset flag after a short delay
}


document.addEventListener('DOMContentLoaded', function () {
    fetch('website/data/sr_vs_k_series_2.json')
        .then(response => response.json())
        .then(data => {
            sr_vs_k_series = data;
            createChart(default_order);
        });

    const taskButtons = [
        'exc-offline',
        'inc-offline'
    ];

    taskButtons.forEach(buttonId => {
        const btn = document.getElementById(buttonId);
        btn.addEventListener('click', () => {
            taskButtons.forEach(id => {
                document.getElementById(id).classList.remove('active');
            });
            btn.classList.add('active');

            if (buttonId === 'exc-offline') {
                createChart(default_order);
            } else if (buttonId === 'inc-offline') {
                createChart(all_models);
            }
        });
    });
});
