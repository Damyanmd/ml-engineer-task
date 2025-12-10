const chatContainer = document.getElementById("chatContainer");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");

questionInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !askBtn.disabled) {
    askQuestion();
  }
});

askBtn.addEventListener("click", askQuestion);

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  questionInput.value = "";
  addMessage(question, "question");
  askBtn.disabled = true;

  const answerDiv = createStreamingMessage();

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    });

    if (!response.ok) throw new Error(`Server returned ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.chunk) appendToStreamingMessage(answerDiv, data.chunk);
          } catch (e) {
            console.error("Error parsing JSON:", e);
          }
        }
      }
    }
  } catch (error) {
    console.error("Error:", error);
    if (answerDiv) answerDiv.remove();
    addMessage("❌ Error: " + error.message, "answer error");
  }

  askBtn.disabled = false;
  questionInput.focus();
}

function createStreamingMessage() {
  const messageDiv = document.createElement("div");
  messageDiv.className = "message";
  const contentDiv = document.createElement("div");
  contentDiv.className = "answer";
  contentDiv.textContent = "";
  messageDiv.appendChild(contentDiv);
  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return contentDiv;
}

function appendToStreamingMessage(element, text) {
  element.textContent += text;
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addMessage(text, className) {
  const messageDiv = document.createElement("div");
  messageDiv.className = "message";
  const contentDiv = document.createElement("div");
  contentDiv.className = "answer " + className;
  contentDiv.textContent = text;
  messageDiv.appendChild(contentDiv);
  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Focus input on page load
questionInput.focus();
