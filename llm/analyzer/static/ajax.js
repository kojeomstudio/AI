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

function updateFiles() {
    fetch('/api/files')
        .then(response => response.json())
        .then(data => {
            let inputList = document.getElementById("input-files");
            let outputList = document.getElementById("output-files");

            inputList.innerHTML = "";
            outputList.innerHTML = "";

            data.input_files.forEach(file => {
                let li = document.createElement("li");
                li.textContent = file;
                inputList.appendChild(li);
            });

            data.output_files.forEach(file => {
                let li = document.createElement("li");
                li.textContent = file;
                outputList.appendChild(li);
            });
        })
        .catch(error => console.error('Error fetching file list:', error));
}


// 상태 및 파일 목록 주기적 업데이트
setInterval(updateStatus, 3000); // 3sec
setInterval(updateFiles, 5000); // 5sec
