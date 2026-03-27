import { useState, useEffect, useRef } from 'react';
import { sendChatQuery } from '../../api/client';

function generateSessionId() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function ChatPanel() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [sessionId] = useState(() => generateSessionId());
  const [followUps, setFollowUps] = useState([]);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleSend(text) {
    const query = (text || input).trim();
    if (!query) return;

    const userMessage = { role: 'user', content: query };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setFollowUps([]);
    setSending(true);

    try {
      const response = await sendChatQuery({
        query,
        session_id: sessionId,
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.response || response.answer || JSON.stringify(response),
      };
      setMessages((prev) => [...prev, assistantMessage]);

      if (response.follow_up_questions && response.follow_up_questions.length > 0) {
        setFollowUps(response.follow_up_questions);
      } else if (response.suggested_questions && response.suggested_questions.length > 0) {
        setFollowUps(response.suggested_questions);
      } else {
        setFollowUps([]);
      }
    } catch (err) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${err.response?.data?.detail || err.message}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setSending(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="chat-placeholder">
            Start a conversation by typing a message below.
          </p>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`chat-bubble ${msg.role === 'user' ? 'chat-user' : 'chat-assistant'}`}
          >
            {msg.content}
          </div>
        ))}
        {sending && (
          <div className="chat-bubble chat-assistant chat-loading">
            Thinking...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {followUps.length > 0 && (
        <div className="chat-followups">
          {followUps.map((q, i) => (
            <button
              key={i}
              className="chip"
              onClick={() => handleSend(q)}
              disabled={sending}
            >
              {q}
            </button>
          ))}
        </div>
      )}

      <div className="chat-input-area">
        <input
          type="text"
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={sending}
        />
        <button
          className="btn btn-primary"
          onClick={() => handleSend()}
          disabled={sending || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatPanel;
