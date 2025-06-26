let sr_vs_k_series;
let chart = null;
let animationTimer = null;
const color_Tableau20 = ['#4E79A7', '#A0CBE8', '#F28E2B', '#FFBE7D', '#59A14F', '#8CD17D', '#B6992D', '#F1CE63', '#499894', '#86BCB6', '#E15759', '#FF9D9A', '#79706E', '#BAB0AC', '#D37295', '#FABFD2', '#B07AA1', '#D4A6C8', '#9D7660', '#D7B5A6'];

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

document.addEventListener('DOMContentLoaded', function () {
    const taskButtons = [
        'visualize-sr-vs-k-scale-with-model-size-llama2-base',
        'visualize-sr-vs-k-scale-with-model-size-llama2-rlhf'
    ];

    taskButtons.forEach(buttonId => {
        const btn = document.getElementById(buttonId);

        btn.addEventListener('click', () => {
            taskButtons.forEach(id => {
                const otherBtn = document.getElementById(id);
                otherBtn.classList.remove('active');
            });

            btn.classList.add('active');
            clearTimeout(animationTimer);
            if (buttonId === 'visualize-sr-vs-k-scale-with-model-size-llama2-base') {
                createChart([
                    'AutoUI Filtered BC, Run 1 (General)',
                    'AutoUI Filtered BC, Run 2 (General)',
                    'AutoUI DigiRL, Run 1 (General)',
                    'AutoUI DigiRL, Run 2 (General)',
                    'AutoUI Pretrained (General)',
                    'GPT-4V Set-of-Marks (General)'
                ]);
            } else if (buttonId === 'visualize-sr-vs-k-scale-with-model-size-llama2-rlhf') {
                createChart([
                    'AutoUI Filtered BC, Run 1 (Webshop)',
                    'AutoUI Filtered BC, Run 2 (Webshop)',
                    'AutoUI DigiRL, Run 1 (Webshop)',
                    'AutoUI DigiRL, Run 2 (Webshop)',
                    'AutoUI Pretrained (Webshop)',
                    'GPT-4V Set-of-Marks (Webshop)'
                ]);
            }
        });
    });

    fetch('website/data/sr_vs_k_series.json')
        .then(response => response.json())
        .then(data => {
            sr_vs_k_series = data;
            createChart(default_order);
        });

        function createChart(subsetNames) {
            if (chart) {
                chart.destroy();
            }
        
            const ctx = document.getElementById('chart-sr-vs-k');
            const sr_vs_k_series_subset = sr_vs_k_series.filter(series => subsetNames.includes(series.label));
        
            sr_vs_k_series_subset.forEach((series, index) => {
                series.borderColor = color_Tableau20[index];
                series.backgroundColor = hexToRGBA(color_Tableau20[index], 0.5);
                series.tension = 0.4; // Smooth curves
            });
        
            const fixedTotalDuration = 20000; // Total duration of the animation (in milliseconds)
            const totalDataPoints = sr_vs_k_series_subset.reduce((acc, series) => acc + series.data.length, 0);
            const delayBetweenPoints = fixedTotalDuration / totalDataPoints;
            const durationPerPoint = fixedTotalDuration / totalDataPoints;
        
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({ length: 10 }, (_, i) => i * 6 * 16),
                    datasets: sr_vs_k_series_subset
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Number of Trajectories Trained',
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
                            duration: durationPerPoint,
                            from: NaN,
                            delay: (ctx) => ctx.index * delayBetweenPoints
                        },
                        y: {
                            type: 'number',
                            easing: 'easeInOutQuart',
                            duration: durationPerPoint,
                            from: (ctx) => ctx.chart.scales.y.getPixelForValue(0),
                            delay: (ctx) => ctx.index * delayBetweenPoints
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
                            itemSort: (a, b) => b.raw - a.raw,
                            callbacks: {
                                title: (context) => 'Number of trajectories trained on: ' + context[0].label,
                                label: (context) => context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%'
                            }
                        }
                    }
                }
            });
        
            animationTimer = setTimeout(function() {
                chart.destroy();
                createChart(subsetNames); // Restart the animation after a short delay
            }, fixedTotalDuration + 3000);
        }
        
});

const default_order = [
    'AutoUI Filtered BC, Run 1 (General)',
    'AutoUI Filtered BC, Run 2 (General)',
    'AutoUI DigiRL, Run 1 (General)',
    'AutoUI DigiRL, Run 2 (General)',
    'AutoUI Pretrained (General)',
    'GPT-4V Set-of-Marks (General)'
];
