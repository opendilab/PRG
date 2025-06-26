
document.addEventListener('DOMContentLoaded', function () {

    const taskButtons = [
        'all-methods',
        'each-method'
    ];

    taskButtons.forEach(buttonId => {
        const btn = document.getElementById(buttonId);

        btn.addEventListener('click', () => {
            taskButtons.forEach(id => {
                const otherBtn = document.getElementById(id);
                otherBtn.classList.remove('active');
            });

            btn.classList.add('active');
        });
    });

    document.getElementById('all-methods').addEventListener('click', function() {
        document.getElementById('all-video-container').style.display = 'block';
        document.getElementById('each-video-container').style.display = 'none';
    });

    document.getElementById('each-method').addEventListener('click', function() {
        document.getElementById('all-video-container').style.display = 'none';
        document.getElementById('each-video-container').style.display = 'flex'; // Use flex to display videos side by side
    });
});

document.addEventListener('DOMContentLoaded', function () {

    const taskButtons = [
        'webshop-all-methods',
        'webshop-each-method'
    ];

    taskButtons.forEach(buttonId => {
        const btn = document.getElementById(buttonId);

        btn.addEventListener('click', () => {
            taskButtons.forEach(id => {
                const otherBtn = document.getElementById(id);
                otherBtn.classList.remove('active');
            });

            btn.classList.add('active');
        });
    });

    document.getElementById('webshop-all-methods').addEventListener('click', function() {
        document.getElementById('all-webshop-container').style.display = 'block';
        document.getElementById('each-webshop-container').style.display = 'none';
    });

    document.getElementById('webshop-each-method').addEventListener('click', function() {
        document.getElementById('all-webshop-container').style.display = 'none';
        document.getElementById('each-webshop-container').style.display = 'flex'; // Use flex to display videos side by side
    });
});

document.addEventListener('DOMContentLoaded', function () {

    const taskButtons = [
        'fast-button',
        'slow-button'
    ];

    taskButtons.forEach(buttonId => {
        const btn = document.getElementById(buttonId);

        btn.addEventListener('click', () => {
            taskButtons.forEach(id => {
                const otherBtn = document.getElementById(id);
                otherBtn.classList.remove('active');
            });

            btn.classList.add('active');
        });
    });

    document.getElementById('fast-button').addEventListener('click', function() {
        document.getElementById('fast-container').style.display = 'block';
        document.getElementById('slow-container').style.display = 'none';
    });

    document.getElementById('slow-button').addEventListener('click', function() {
        document.getElementById('fast-container').style.display = 'none';
        document.getElementById('slow-container').style.display = 'block';
    });
});
