// Configuration
const CONFIG = {
    BACKEND_URL: window.location.origin,
    CHAT_ENDPOINT: '/api/v1/chat',
    CHAT_HISTORY_ENDPOINT: '/api/v1/history',
    CLEAR_HISTORY_ENDPOINT: '/api/v1/history',
};

// State management
const state = {
    userId: '',
    userName: 'Guest',
    chatHistory: [],
    isWaiting: false,
    storageKey: 'skysafe-chat-user',
};

// DOM elements
const elements = {
    chatHistory: document.getElementById('chat-history'),
    loadingIndicator: document.getElementById('loading-indicator'),
    chatForm: document.getElementById('chat-form'),
    userInput: document.getElementById('user-input'),
    sendBtn: document.getElementById('send-btn'),
    sourcesContainer: document.getElementById('sources-container'),
    userIdInput: document.getElementById('user-id-input'),
    userNameInput: document.getElementById('user-name-input'),
    loadHistoryBtn: document.getElementById('load-history-btn'),
    clearChatBtn: document.getElementById('clear-chat-btn'),
    menuToggle: document.getElementById('menu-toggle'),
    menuPanel: document.getElementById('menu-panel'),
    menuHome: document.getElementById('menu-home'),
    menuChat: document.getElementById('menu-chat'),
    menuClassification: document.getElementById('menu-classification'),
};

/**
 * Persist user identity in localStorage
 */
function setCookie(name, value, days = 30) {
    const expires = new Date(Date.now() + days * 864e5).toUTCString();
    document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/`;
}

function getCookie(name) {
    return document.cookie
        .split('; ')
        .map((v) => v.split('='))
        .reduce((acc, [k, v]) => (k === name ? decodeURIComponent(v || '') : acc), null);
}

function clearUserCookies() {
    document.cookie = `skysafe_user_id=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
    document.cookie = `skysafe_user_name=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
}

function persistUserInfo() {
    const payload = {
        userId: state.userId,
        userName: state.userName,
    };
    setCookie('skysafe_user_id', state.userId);
    setCookie('skysafe_user_name', state.userName);
    try {
        localStorage.setItem(state.storageKey, JSON.stringify(payload));
    } catch (e) {
        console.debug('LocalStorage unavailable, skipping persist');
    }
}

/**
 * Load user identity from cookies/localStorage or defaults
 */
function hydrateUserInfo() {
    let stored = null;
    try {
        stored = JSON.parse(localStorage.getItem(state.storageKey) || 'null');
    } catch (e) {
        stored = null;
    }

    const cookieId = getCookie('skysafe_user_id');
    const cookieName = getCookie('skysafe_user_name');

    state.userId = cookieId || stored?.userId || `user_${Date.now()}`;
    state.userName = cookieName || stored?.userName || 'Guest';

    elements.userIdInput.value = state.userId;
    elements.userNameInput.value = state.userName;
}

/**
 * Sync state from inputs and persist
 */
function syncUserStateFromInputs() {
    state.userId = (elements.userIdInput.value || '').trim() || state.userId || `user_${Date.now()}`;
    state.userName = (elements.userNameInput.value || '').trim() || 'Guest';
    elements.userIdInput.value = state.userId;
    elements.userNameInput.value = state.userName;
    persistUserInfo();
}
/**
 * Display user message in chat
 */
function displayUserMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(message)}</div>`;
    elements.chatHistory.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Display assistant message in chat (with markdown rendering)
 */
function displayAssistantMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    try {
        // Render markdown safely
        const htmlContent = marked.parse(message);
        const cleanContent = DOMPurify.sanitize(htmlContent);
        messageDiv.innerHTML = `<div class="message-content">${cleanContent}</div>`;
    } catch (error) {
        console.error('Error rendering markdown:', error);
        messageDiv.innerHTML = `<div class="message-content">${escapeHtml(message)}</div>`;
    }
    
    elements.chatHistory.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Display loading indicator
 */
function showLoading() {
    elements.loadingIndicator.classList.remove('hidden');
    scrollToBottom();
}

function hideLoading() {
    elements.loadingIndicator.classList.add('hidden');
}

/**
 * Display sources
 */
function displaySources(sources) {
    elements.sourcesContainer.innerHTML = '';
    
    if (!sources || sources.length === 0) {
        elements.sourcesContainer.innerHTML = '<p class="placeholder-text">No sources available</p>';
        return;
    }

    sources.forEach((source, index) => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';
        sourceItem.innerHTML = `
            <div class="source-item-title">Source ${index + 1}</div>
            <div class="source-item-meta">${escapeHtml(source)}</div>
        `;
        elements.sourcesContainer.appendChild(sourceItem);
    });
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    setTimeout(() => {
        elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
    }, 0);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

/**
 * Extract user-friendly error message from server response or error object
 * Server returns: { error: { code, message, details? } }
 */
async function extractErrorMessage(response, fallbackStatus) {
    try {
        const data = await response.json();
        if (data && data.error) {
            const { code, message, details } = data.error;
            let msg = message || code || 'Unknown error';
            if (details) {
                msg += ` (${typeof details === 'string' ? details : JSON.stringify(details)})`;
            }
            return msg;
        }
        return `Server error (${fallbackStatus})`;
    } catch {
        return `Server error (${fallbackStatus})`;
    }
}

/**
 * Display error message in chat with styling
 */
function displayErrorMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant error';
    messageDiv.innerHTML = `<div class="message-content error-content">⚠️ ${escapeHtml(message)}</div>`;
    elements.chatHistory.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Remove welcome message if it exists
 */
function removeWelcomeMessage() {
    const welcome = elements.chatHistory.querySelector('.welcome-message');
    if (welcome) {
        welcome.remove();
    }
}

function clearChatUI() {
    elements.chatHistory.innerHTML = '';
}

function renderHistoryFromState() {
    clearChatUI();
    if (!state.chatHistory.length) {
        const empty = document.createElement('div');
        empty.className = 'placeholder-text';
        empty.textContent = 'No messages yet for this user';
        elements.chatHistory.appendChild(empty);
        return;
    }

    state.chatHistory.forEach((msg) => {
        if (msg.role === 'user') {
            displayUserMessage(msg.content || '');
        } else if (msg.role === 'assistant') {
            displayAssistantMessage(msg.content || '');
        }
    });
}
    function normalizeHistoryRecords(records) {
        if (!Array.isArray(records)) return [];
        return records
            .filter((r) => r && typeof r === 'object')
            .map((r) => ({
                role: r.role,
                content: typeof r.content === 'string' ? r.content : '',
            }))
            .filter((r) => r.role === 'user' || r.role === 'assistant');
    }

    async function loadChatHistory() {
        syncUserStateFromInputs();
        const userId = state.userId;
        if (!userId) {
            alert('Please provide a User ID to load history.');
            return;
        }

        showLoading();
        try {
            const url = `${CONFIG.BACKEND_URL}${CONFIG.CHAT_HISTORY_ENDPOINT}/${encodeURIComponent(userId)}`;
            const resp = await fetch(url, { method: 'GET' });
            hideLoading();

            if (!resp.ok) {
                const errMsg = await extractErrorMessage(resp, resp.status);
                throw new Error(errMsg);
            }

            const data = await resp.json();
            const records = data.messages || [];
            state.chatHistory = normalizeHistoryRecords(records);
            renderHistoryFromState();
        } catch (err) {
            hideLoading();
            console.error('History error:', err);
        
            let userMessage;
            if (err.name === 'TypeError' && err.message.includes('fetch')) {
                userMessage = 'Unable to connect to the server. Please check your connection.';
            } else {
                userMessage = `Could not load history: ${err.message}`;
            }
            displayErrorMessage(userMessage);
        }
    }

/**
 * Clear chat history from server and UI
 */
async function clearChatHistory() {
    syncUserStateFromInputs();
    const userId = state.userId;
    if (!userId) {
        alert('Please provide a User ID to clear history.');
        return;
    }

    // Ask for confirmation
    if (!confirm('Are you sure you want to delete all chat history for this user? This action cannot be undone.')) {
        return;
    }

    showLoading();
    try {
        const url = `${CONFIG.BACKEND_URL}${CONFIG.CLEAR_HISTORY_ENDPOINT}/${encodeURIComponent(userId)}`;
        const resp = await fetch(url, { method: 'DELETE' });
        hideLoading();

        if (!resp.ok) {
            const errMsg = await extractErrorMessage(resp, resp.status);
            throw new Error(errMsg);
        }

        const data = await resp.json();
        console.log(`Cleared ${data.deleted_count} messages for user ${userId}`);

        // Clear local state and UI
        state.chatHistory = [];
        clearChatUI();

        // Show success message
        const successDiv = document.createElement('div');
        successDiv.className = 'welcome-message';
        successDiv.innerHTML = `
            <h2>Chat Cleared</h2>
            <p>Your chat history has been deleted. Start a new conversation!</p>
        `;
        elements.chatHistory.appendChild(successDiv);
    } catch (err) {
        hideLoading();
        console.error('Clear history error:', err);
        
        let userMessage;
        if (err.name === 'TypeError' && err.message.includes('fetch')) {
            userMessage = 'Unable to connect to the server. Please check your connection.';
        } else {
            userMessage = `Could not clear history: ${err.message}`;
        }
        displayErrorMessage(userMessage);
    }
}


/**
 * Send message to backend
 */
async function sendMessage(question) {
    syncUserStateFromInputs();
    if (!state.userId) {
        alert('Please provide a User ID before chatting.');
        return;
    }

    // Remove welcome message on first interaction
    removeWelcomeMessage();

    // Add user message to chat
    displayUserMessage(question);

    // Add to chat history
    state.chatHistory.push({
        role: 'user',
        content: question
    });

    // Show loading
    showLoading();
    state.isWaiting = true;
    elements.sendBtn.disabled = true;
    elements.userInput.disabled = true;

    try {
        const payload = {
            user_id: state.userId,
            user_name: state.userName,
            preprocess_query: false,
            chat_history: state.chatHistory
        };

        console.log('Sending chat request with payload:', JSON.stringify(payload, null, 2));

        const response = await fetch(`${CONFIG.BACKEND_URL}${CONFIG.CHAT_ENDPOINT}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        console.log('Response received:', response.status, response.statusText);

        hideLoading();

        if (!response.ok) {
            const errMsg = await extractErrorMessage(response, response.status);
            throw new Error(errMsg);
        }

        const data = await response.json();

        // Add assistant message to chat
        displayAssistantMessage(data.answer);

        // Add to chat history
        state.chatHistory.push({
            role: 'assistant',
            content: data.answer
        });

        // Display sources
        if (data.sources && data.sources.length > 0) {
            displaySources(data.sources);
        }

    } catch (error) {
        hideLoading();
        console.error('Chat error:', error);
        
        let userMessage;
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            userMessage = 'Unable to connect to the server. Please check your connection and try again.';
        } else {
            userMessage = `Error: ${error.message}`;
        }
        
        displayErrorMessage(userMessage);
        
        // Don't add error messages to chat history to avoid sending them back to server
    } finally {
        state.isWaiting = false;
        elements.sendBtn.disabled = false;
        elements.userInput.disabled = false;
        elements.userInput.focus();
    }
}

/**
 * Handle form submission
 */
function handleFormSubmit(e) {
    e.preventDefault();

    const question = elements.userInput.value.trim();
    if (!question || state.isWaiting) {
        return;
    }

    elements.userInput.value = '';
    sendMessage(question);
}

/**
 * Initialize app
 */
function init() {
    hydrateUserInfo();
    syncUserStateFromInputs();

    // Add event listeners
    elements.chatForm.addEventListener('submit', handleFormSubmit);
    elements.userInput.focus();

    elements.userIdInput.addEventListener('change', syncUserStateFromInputs);
    elements.userNameInput.addEventListener('change', syncUserStateFromInputs);
    elements.loadHistoryBtn.addEventListener('click', loadChatHistory);
    elements.clearChatBtn.addEventListener('click', clearChatHistory);

    // Menu interactions
    elements.menuToggle.addEventListener('click', () => {
        elements.menuPanel.classList.toggle('open');
    });
    elements.menuHome.addEventListener('click', () => window.location.href = '/home');
    elements.menuChat.addEventListener('click', () => window.location.href = '/chat');
    elements.menuClassification.addEventListener('click', () => window.location.href = '/classification');
    document.addEventListener('click', (e) => {
        if (!elements.menuPanel.contains(e.target) && !elements.menuToggle.contains(e.target)) {
            elements.menuPanel.classList.remove('open');
        }
    });

    // Remove welcome message on first input
    elements.userInput.addEventListener('focus', () => {
        if (elements.chatHistory.querySelector('.welcome-message')) {
            // Just focus, don't remove yet
        }
    });

    // Attempt to load existing history for stored user
    loadChatHistory();

    console.log('Chatbot initialized');
}

// Start app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
