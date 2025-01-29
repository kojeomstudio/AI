function reloadConfig() {
    fetch('/api/config/reload')
        .then(response => response.json())
        .then(data => {
            document.getElementById("status").innerText = data.message;
        })
        .catch(error => console.error('Error:', error));
}

function startProcessing() {
    fetch('/api/process', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("status").innerText = data.message;
    })
    .catch(error => console.error('Error:', error));
}

function updateStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById("processing-count").innerText = data.processing;
            document.getElementById("completed-count").innerText = data.completed;
        })
        .catch(error => console.error('Error:', error));
}

// 3초마다 상태 업데이트
setInterval(updateStatus, 3000);
