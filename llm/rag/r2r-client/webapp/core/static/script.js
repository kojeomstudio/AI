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

// 메시지 전송 함수
async function sendMessage() {
    const inputField = document.getElementById("input");
    const modeSelect = document.getElementById("mode-select");
    const messageContainer = document.getElementById("messages");
    const submitButton = document.getElementById("submit-btn");

    const userMessage = inputField.value.trim();
    const selectedMode = modeSelect.value; // query 또는 agent

    if (!userMessage) return;

    // 사용자 메시지 추가
    const userBubble = document.createElement("div");
    userBubble.className = "user-message";
    userBubble.textContent = userMessage;
    messageContainer.appendChild(userBubble);

    inputField.value = ""; // 입력 필드 초기화

    // 입력과 버튼 비활성화
    inputField.disabled = true;
    submitButton.disabled = true;

    // 로딩 표시 추가 (스로버 형태)
    const loadingBubble = document.createElement("div");
    loadingBubble.className = "loading-message";
    const spinner = document.createElement("div");
    spinner.className = "spinner"; // CSS로 스로버 스타일 적용 필요
    loadingBubble.appendChild(spinner);
    messageContainer.appendChild(loadingBubble);

    // 서버에 요청 보내기
    try {
        const response = await fetch(`/${selectedMode}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question: userMessage }),
        });

        if (!response.ok) {
            throw new Error(`서버 오류: ${response.statusText}`);
        }

        // 서버 응답 처리 부분 수정
        const data = await response.json();
        const botMessages = data.responses.map(
            (resp) => resp.message_content || "응답이 없습니다."
        );

        // 로딩 메시지 제거
        messageContainer.removeChild(loadingBubble);

        // 봇 응답 추가
        botMessages.forEach((botMessage) => {
            const botBubble = document.createElement("div");
            botBubble.className = "bot-message";
            typeMessage(botBubble, botMessage).then(() => {
                messageContainer.appendChild(botBubble);
            });
        });
    } catch (error) {
        console.error("메시지 전송 중 오류:", error);
        const errorBubble = document.createElement("div");
        errorBubble.className = "error-message";
        errorBubble.textContent = "메시지 전송 중 오류가 발생했습니다.";
        messageContainer.appendChild(errorBubble);
    } finally {
        // 입력과 버튼 다시 활성화
        inputField.disabled = false;
        submitButton.disabled = false;
        inputField.focus(); // 입력 필드에 포커스 복원
    }
}
