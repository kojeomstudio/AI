document.getElementById("submit-btn").addEventListener("click", sendMessage);
document.getElementById("input").addEventListener("keypress", function (e) {
    if (e.key === 'Enter') sendMessage();
});

let default_typingDelay = 5; // 기본값
let typingDelay = 0;

// config.json에서 typing_delay 값을 가져오는 함수
async function fetchTypingDelay() {
    try {
        const response = await fetch("../config.json");
        const config = await response.json();
        typingDelay = config.typing_delay || default_typingDelay; // 설정값이 없으면 기본값 사용
    } catch (error) {
        console.error("Config 파일을 로드하는 중 오류 발생:", error);
    }
    finally
    {
        typingDelay = default_typingDelay;
    }
}

// 페이지 로드 시 config.json에서 설정값 로드
document.addEventListener("DOMContentLoaded", fetchTypingDelay);

async function typeMessage(container, text) {
    for (const char of text) {
        container.innerHTML += char;
        await new Promise((resolve) => setTimeout(resolve, typingDelay)); // 글자당 지연 속도.
    }
}

async function sendMessage() {
    const input = document.getElementById("input");
    const question = input.value.trim();
    const messages = document.getElementById("messages");

    if (question === "") return;

    // 유저 메시지를 대화창에 추가
    messages.innerHTML += `<div class="message user-message">${question}</div>`;
    input.value = "";
    input.disabled = true;
    document.getElementById("submit-btn").disabled = true;  // 버튼 비활성화

    // 로더 표시
    const loader = document.createElement("div");
    loader.className = "loader";
    messages.appendChild(loader);

    // 응답 시간 측정 시작
    const startTime = Date.now();

    // 서버로 요청 보내기
    try {
        const response = await fetch("/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });
        const result = await response.json();
        const responseText = result.responses[0].message_content;

        // 로더 제거
        loader.remove();

        // 봇 메시지 추가 및 타이핑 효과 적용
        const botMessage = document.createElement("div");
        botMessage.className = "message bot-message";
        messages.appendChild(botMessage);

        // 응답 시간 추가
        const endTime = Date.now();
        const elapsedSeconds = ((endTime - startTime) / 1000).toFixed(2);

        // 지연 출력 효과 시작 (비동기)
        await typeMessage(botMessage, responseText);

        botMessage.innerHTML += `<div class="response-time">응답 시간: ${elapsedSeconds}초</div>`;
    } catch (error) {
        loader.remove();
        messages.innerHTML += `<div class="message bot-message">오류 발생: ${error}</div>`;
    } finally {
        input.disabled = false;
        document.getElementById("submit-btn").disabled = false;  // 버튼 활성화
        input.focus();
    }

    // 스크롤을 최신 메시지로 이동
    loader.scrollIntoView({ behavior: "smooth", block: "end" });
}
