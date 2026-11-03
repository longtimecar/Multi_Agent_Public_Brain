const state = { facts: [], factTotal: null, factsTruncated: false, selected: null, episodes: [], episodeTotal: 0, episodeAgent: '', filters: { q: '', status: '', type: '' }, tree: { type: '', subject: '' }, currentView: 'chat', dialog: null, sourceLabels: {},
  memoryLimit: 8, chat: { sessionId: null, messages: [], memory: [], sending: false }, model: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const labels = {
  type: { preference: '偏好', fact: '事实', goal: '目标', project: '项目', constraint: '约束', entity: '实体', event: '事件', case: '案例', tool: '工具', belief: '信念' },
  status: { active: '有效', contested: '冲突', superseded: '已替代', retracted: '已撤回' },
};

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function icon(name) {
  const paths = {
    edit: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 17.5V20h2.5L19 7.5 16.5 5 4 17.5Z"/><path d="m14.8 6.7 2.5 2.5"/></svg>',
    flag: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 21V4m0 1h11l-1.5 3L16 11H5"/></svg>',
    undo: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 8 4 12l5 4"/><path d="M5 12h8a6 6 0 0 1 6 6"/></svg>',
    plus: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
  };
  return paths[name] || '';
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, { headers: { 'Accept': 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}) }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || (response.status === 409 ? '已有导入任务正在运行，请等待它完成。' : `请求失败（${response.status}）`));
  return payload;
}

function notify(message, error = false) {
  const node = document.createElement('div');
  node.className = `toast${error ? ' is-error' : ''}`;
  node.textContent = message;
  $('#toast-stack').append(node);
  window.setTimeout(() => node.remove(), 3600);
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value).slice(0, 10) : date.toLocaleDateString('zh-CN', { year: 'numeric', month: 'short', day: 'numeric' });
}

function formatTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value).slice(0, 16) : date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}

function shortToken(value, size = 10) {
  const text = String(value || '');
  return text.length > size * 2 ? `${text.slice(0, size)}…${text.slice(-4)}` : text;
}

function widthBucket(value) {
  const n = Math.max(0, Math.min(100, Number(value) || 0));
  return `w-${n === 0 ? 0 : Math.ceil(n / 10) * 10}`;
}

function statusBadge(status) {
  return `<span class="status-badge status-${escapeHtml(status)}">${escapeHtml(labels.status[status] || status)}</span>`;
}

function typeTag(type) {
  return `<span class="type-tag ${escapeHtml(type)}">${escapeHtml(labels.type[type] || type)}</span>`;
}

function displaySource(source) {
  return state.lib?.sourceLabels?.[source] || state.sourceLabels?.[source] || source || '未知来源';
}

function renderRecent(recent) {
  const root = $('#recent-list');
  if (!root) return;
  const items = Array.isArray(recent) ? recent.slice(0, 6) : [];
  if (!items.length) {
    root.innerHTML = '<div class="side-empty">还没有形成长期记忆。</div>';
    return;
  }
  root.innerHTML = items.map((fact) => `
    <button class="recent-item" type="button" data-recent-fact="${escapeHtml(fact.id)}">
      <span class="recent-type ${escapeHtml(fact.type)}" aria-hidden="true"></span>
      <span class="recent-main">
        <strong>${escapeHtml(fact.abstract || fact.claim)}</strong>
        <small>${escapeHtml(labels.type[fact.type] || fact.type)} · ${escapeHtml(displaySource(fact.source?.agent || 'unknown'))} · ${formatDate(fact.valid_from || fact.created_at)}</small>
      </span>
      ${statusBadge(fact.status)}
    </button>`).join('');
  root.querySelectorAll('[data-recent-fact]').forEach((button) => button.addEventListener('click', async () => {
    state.tree = { type: '', subject: '' };
    switchView('memories');
    await selectFact(button.dataset.recentFact);
  }));
}

function renderSourceSummary(agents) {
  const root = $('#source-summary');
  if (!root) return;
  const entries = Object.entries(agents || {}).filter(([, count]) => Number(count) > 0).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    root.innerHTML = '<div class="side-empty">还没有来源数据。</div>';
    return;
  }
  const total = entries.reduce((sum, [, count]) => sum + Number(count), 0);
  const max = Math.max(...entries.map(([, count]) => Number(count)), 1);
  const tones = ['teal', 'coral', 'gold', 'slate', 'moss', 'blue'];
  root.innerHTML = `<p class="source-total"><strong>${total}</strong><span>条事实来自 ${entries.length} 个来源</span></p>
    <div class="source-rows">${entries.map(([source, count], index) => `
      <div class="source-row" title="${escapeHtml(source)}">
        <div class="source-label"><span>${escapeHtml(displaySource(source))}</span><strong>${count}</strong></div>
        <div class="source-track"><span class="source-fill ${tones[index % tones.length]} ${widthBucket(Math.max(8, Number(count) / max * 100))}"></span></div>
      </div>`).join('')}</div>`;
}

function renderOverview(data) {
  const stats = data.stats || {};
  state.sourceLabels = data.source_labels || state.sourceLabels;
  $('#stat-total').textContent = stats.total ?? 0;
  $('#stat-active').textContent = stats.active ?? 0;
  $('#stat-contested').textContent = stats.contested ?? 0;
  $('#stat-episodes').textContent = stats.userEpisodes ?? 0;
  $('#memory-home').textContent = data.home || '—';
  $('#profile-body').textContent = data.profile?.content || '尚未创建 profile.md。';
  const online = $('#system-dot');
  online.classList.add('is-online');
  $('#system-label').textContent = '本地服务在线';
  $('#system-detail').textContent = (stats.total ?? 0) > 0 ? `${stats.active ?? 0} 条有效记忆` : '等待第一条记忆';
  const isEmpty = !(stats.total ?? 0);
  $('#overview-empty')?.classList.toggle('is-hidden', !isEmpty);
  $('#overview-lower')?.classList.toggle('is-hidden', isEmpty);
  renderRecent(data.recent);
  renderSourceSummary(data.agents);
}

const CHAT_EMPTY_HTML = `<div class="chat-empty">
  <div class="empty-mark">AI</div>
  <p class="chat-kicker">MEMORY-ASSISTED CHAT</p>
  <h3>从一个真实问题开始</h3>
  <p>回答会带着你已经确认过的长期记忆，不再是每次重新自我介绍。</p>
  <div class="chat-starters">
    <button type="button" data-chat-prompt="我最近在做什么项目？">我最近在做什么项目？</button>
    <button type="button" data-chat-prompt="回顾一下我的工作偏好。">回顾我的工作偏好</button>
    <button type="button" data-chat-prompt="有哪些信息需要我确认或纠正？">检查待确认的信息</button>
  </div>
</div>`;

function renderChat() {
  const list = $('#chat-messages');
  if (!state.chat.messages.length) {
    list.innerHTML = CHAT_EMPTY_HTML;
  } else {
    list.innerHTML = state.chat.messages.map((item) => `<div class="chat-message chat-message-${escapeHtml(item.role)}"><div class="chat-message-label">${item.role === 'user' ? '你' : 'Agent'}</div><div class="chat-bubble">${escapeHtml(item.content).replace(/\n/g, '<br>')}</div></div>`).join('');
    list.scrollTop = list.scrollHeight;
  }
  $('#chat-session-label').textContent = state.chat.sessionId ? `会话 ${state.chat.sessionId.slice(0, 8)}` : '新会话';
  $('#chat-send').disabled = state.chat.sending || (state.model && !state.model.configured);
  $('#chat-send span').textContent = state.chat.sending ? '思考中…' : '发送';
}

function renderMemorySources() {
  const list = $('#chat-memory-list');
  $('#chat-memory-count').textContent = state.chat.memory.length;
  list.innerHTML = state.chat.memory.length ? state.chat.memory.map((item) => `
    <div class="memory-source-item"><div class="memory-source-top"><span>${escapeHtml(item.subject)}</span>${statusBadge(item.status)}</div><p>${escapeHtml(item.claim)}</p><span class="memory-source-meta">${Math.round((item.confidence || 0) * 100)}% confidence · ${escapeHtml(item.id)}</span></div>`).join('') : '<div class="side-empty">发送消息后，这里会列出召回的记忆。</div>';
}

function renderModel(data) {
  state.model = data;
  const configured = Boolean(data.configured);
  $('#model-dot').classList.remove('is-online');
  $('#model-status-badge').className = `status-badge ${configured ? 'status-contested' : 'status-retracted'}`;
  $('#model-status-badge').textContent = configured ? '待测试' : '未配置';
  $('#model-name').textContent = data.model || '未配置模型';
  $('#model-format').textContent = data.base_url || '—';
  $('#model-api-format').textContent = data.format || '—';
  $('#model-key-status').textContent = data.api_key_set ? '已设置' : '未设置';
  $('#model-timeout').textContent = data.timeout ? `${data.timeout}s` : '—';
  $('#model-embedding').textContent = data.embedding || '—';
  $('#model-test-button').disabled = !configured;
  $('#model-result').textContent = configured ? '已读取配置，请测试连接。' : '请先在 .env 中配置 LLM 地址和模型名。';
  if (!$('#model-config').open) {
    $('#model-config-format').value = data.format === 'anthropic' ? 'anthropic' : 'openai';
    $('#model-config-base-url').value = data.base_url === '—' ? '' : data.base_url;
    $('#model-config-model').value = data.model === '—' ? '' : data.model;
    $('#model-config-timeout').value = data.timeout || 30;
  }
  renderEmbedding(data);
}

function renderEmbedding(data) {
  state.model = data;
  const configured = Boolean(data.embedding_configured);
  $('#embedding-dot').classList.remove('is-online');
  $('#embedding-status-badge').className = `status-badge ${configured ? 'status-contested' : 'status-retracted'}`;
  $('#embedding-status-badge').textContent = configured ? '待测试' : '未启用';
  $('#embedding-name').textContent = configured ? (data.embedding_model || '未命名模型') : '未配置 Embedding';
  $('#embedding-format').textContent = data.embedding_base_url || '—';
  $('#embedding-model').textContent = data.embedding_model || '—';
  $('#embedding-key-status').textContent = data.embedding_api_key_set ? '已设置' : '未设置';
  $('#embedding-timeout').textContent = data.embedding_timeout ? `${data.embedding_timeout}s` : '—';
  $('#embedding-mode').textContent = configured ? '语义 + 全文' : '仅全文';
  $('#embedding-test-button').disabled = !configured;
  $('#embedding-result').textContent = configured
    ? '已读取配置，请测试连接。'
    : '未配置 Embedding 时检索退回全文匹配，功能仍可用。';
  if (!$('#embedding-config').open) {
    $('#embedding-config-base-url').value = data.embedding_base_url === '—' ? '' : data.embedding_base_url;
    $('#embedding-config-model').value = data.embedding_model === '—' ? '' : data.embedding_model;
    $('#embedding-config-timeout').value = data.embedding_timeout || 20;
  }
}

function memoryGroups() {
  const groups = new Map();
  for (const fact of state.facts) {
    if (!groups.has(fact.type)) groups.set(fact.type, new Map());
    const subs = groups.get(fact.type);
    subs.set(fact.subject, (subs.get(fact.subject) || 0) + 1);
  }
  return groups;
}

function memoryMode() {
  if (state.filters.q) return 'search';
  if (state.tree.subject) return 'entries';
  if (state.tree.type) return 'subjects';
  return 'types';
}

function memoryStatusLabel() {
  return { active: '当前有效', contested: '待处理冲突', superseded: '已替代', retracted: '已撤回' }[state.filters.status] || '状态筛选';
}

function memoryPath(mode) {
  const root = $('#memory-breadcrumb');
  const back = $('#memory-back');
  if (!root) return;
  const typeLabel = labels.type[state.tree.type] || state.tree.type;
  let html = '<button class="memory-crumb" type="button" data-memory-root>全部记忆</button>';
  if (mode === 'search') {
    html += `<span class="memory-crumb-sep">/</span><span class="memory-crumb is-current">搜索结果</span>`;
  } else {
    if (state.filters.status) {
      html += `<span class="memory-crumb-sep">/</span>`;
      html += state.tree.type
        ? `<button class="memory-crumb" type="button" data-memory-status>${escapeHtml(memoryStatusLabel())}</button>`
        : `<span class="memory-crumb is-current">${escapeHtml(memoryStatusLabel())}</span>`;
    }
    if (mode === 'subjects' || mode === 'entries') {
      html += `<span class="memory-crumb-sep">/</span>`;
      html += mode === 'entries'
        ? `<button class="memory-crumb" type="button" data-memory-type="${escapeHtml(state.tree.type)}">${escapeHtml(typeLabel)}</button><span class="memory-crumb-sep">/</span><span class="memory-crumb is-current">${escapeHtml(state.tree.subject)}</span>`
        : `<span class="memory-crumb is-current">${escapeHtml(typeLabel)}</span>`;
    }
  }
  root.innerHTML = html;
  if (back) {
    const atTop = mode === 'types' && !state.filters.status;
    back.disabled = atTop;
    back.title = atTop ? '已经在最上层' : '返回上一级';
  }
}

function folderCard(attrs, level, title, count, meta, preview) {
  return `<button class="memory-folder" type="button" ${attrs}>
    <span class="memory-folder-top"><span class="memory-folder-level">${level}</span><span class="memory-folder-count">${count} 条</span></span>
    <strong>${escapeHtml(title)}</strong>
    <span class="memory-folder-meta">${escapeHtml(meta)}</span>
    <span class="memory-folder-preview">${escapeHtml(preview)}</span>
    <span class="memory-folder-open">打开文件夹 <span aria-hidden="true">→</span></span>
  </button>`;
}

function renderFolderView(mode) {
  const root = $('#memory-folder-view');
  if (!root) return;
  const groups = memoryGroups();
  if (!state.facts.length) {
    root.innerHTML = '<div class="empty-state"><div class="empty-mark">M</div><h3>还没有记忆</h3><p>接入 Agent 并开始对话后，记忆会按分类出现在这里。</p></div>';
    return;
  }
  if (mode === 'types') {
    const types = [...groups.keys()].sort((a, b) => (labels.type[a] || a).localeCompare(labels.type[b] || b, 'zh'));
    root.innerHTML = `<div class="memory-folder-head"><div><span class="memory-folder-level">L1 分类</span><h3>选择一个分类文件夹</h3><p>里面再按主题分层；进入主题后查看 L0 摘要。</p></div><span class="memory-folder-total">${state.factTotal ?? state.facts.length} 条记忆</span></div>
      <div class="memory-folder-grid">${types.map((type) => {
        const subs = groups.get(type);
        const count = [...subs.values()].reduce((sum, n) => sum + n, 0);
        const subjects = [...subs.keys()].sort((a, b) => a.localeCompare(b, 'zh'));
        return folderCard(`data-folder-type="${escapeHtml(type)}"`, 'L1 分类', labels.type[type] || type, count,
          `${subs.size} 个主题`, subjects.slice(0, 3).join(' · ') || '暂无主题');
      }).join('')}</div>`;
    return;
  }
  const subs = groups.get(state.tree.type) || new Map();
  const subjects = [...subs.keys()].sort((a, b) => a.localeCompare(b, 'zh'));
  const total = [...subs.values()].reduce((sum, n) => sum + n, 0);
  root.innerHTML = `<div class="memory-folder-head"><div><span class="memory-folder-level">L1 主题</span><h3>${escapeHtml(labels.type[state.tree.type] || state.tree.type)}</h3><p>再选一个主题，查看它下面的 L0 摘要。</p></div><span class="memory-folder-total">${subs.size} 个主题 · ${total} 条</span></div>
    <div class="memory-subject-grid">${subjects.map((subject) => `
      <button class="memory-subject-folder" type="button" data-folder-subject="${escapeHtml(subject)}">
        <span class="memory-subject-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M3.5 7.5A2 2 0 0 1 5.5 5.5h4l2 2h7a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2v-9Z"/></svg></span>
        <span class="memory-subject-copy"><strong>${escapeHtml(subject)}</strong><small>进入查看 ${subs.get(subject)} 条 L0 摘要</small></span>
        <span class="memory-subject-count">${subs.get(subject)}</span>
        <span class="memory-subject-arrow" aria-hidden="true">→</span>
      </button>`).join('')}</div>`;
}

function renderFacts() {
  const list = $('#memory-list');
  const folderView = $('#memory-folder-view');
  const tableHead = $('#memory-table-head');
  const mode = memoryMode();
  memoryPath(mode);
  if (mode === 'types' || mode === 'subjects') {
    list.classList.add('is-hidden');
    tableHead?.classList.add('is-hidden');
    folderView?.classList.remove('is-hidden');
    $('#memory-count').textContent = mode === 'types'
      ? (state.factTotal ?? state.facts.length)
      : [...(memoryGroups().get(state.tree.type) || new Map()).values()].reduce((sum, n) => sum + n, 0);
    renderFolderView(mode);
    return;
  }
  folderView?.classList.add('is-hidden');
  list.classList.remove('is-hidden');
  tableHead?.classList.remove('is-hidden');
  const rows = state.facts.filter((fact) => (!state.tree.type || fact.type === state.tree.type) && (!state.tree.subject || fact.subject === state.tree.subject));
  $('#memory-count').textContent = state.filters.q || state.filters.status ? rows.length : (state.tree.subject ? rows.length : (state.factTotal ?? rows.length));
  const selectedIndex = state.selected ? rows.findIndex((fact) => fact.id === state.selected.id) : -1;
  if (selectedIndex >= state.memoryLimit) state.memoryLimit = Math.ceil((selectedIndex + 1) / 8) * 8;
  const visible = rows.slice(0, state.memoryLimit);
  list.innerHTML = rows.length ? visible.map((fact) => `
    <button class="memory-row${state.selected?.id === fact.id ? ' is-selected' : ''}" type="button" data-fact-id="${escapeHtml(fact.id)}">
      <div class="memory-main"><div class="memory-line">${typeTag(fact.type)}<span class="memory-claim" title="${escapeHtml(fact.claim)}">${escapeHtml(fact.abstract || fact.claim)}</span></div><div class="memory-subject">${escapeHtml(fact.subject)} · ${formatDate(fact.valid_from)}</div></div>
      <div>${statusBadge(fact.status)}</div>
      <div class="evidence-count">${fact.evidence?.length ?? 0}</div>
    </button>`).join('') : '<div class="empty-state"><div class="empty-mark">M</div><h3>还没有匹配的记忆</h3><p>换个文件夹或关键词，或等待 Agent 捕获新的用户信息。</p></div>';
  if (rows.length > visible.length) {
    list.insertAdjacentHTML('beforeend', `
      <div class="memory-more">
        <span>已显示 ${visible.length} / ${rows.length}</span>
        <button class="button button-outline" type="button" id="memory-more-button">再显示 ${Math.min(8, rows.length - visible.length)} 条</button>
      </div>`);
    $('#memory-more-button')?.addEventListener('click', () => {
      state.memoryLimit += 8;
      renderFacts();
    });
  }
  $$('.memory-row').forEach((row) => row.addEventListener('click', () => selectFact(row.dataset.factId)));
}

function resetMemoryLimit() {
  state.memoryLimit = 8;
}

function clearMemoryFilters() {
  state.filters.q = '';
  state.filters.status = '';
  state.filters.type = '';
  $('#search-input').value = '';
  $('#status-filter').value = '';
}

function memoryGoBack() {
  if (state.filters.q) {
    state.filters.q = '';
    $('#search-input').value = '';
    resetMemoryLimit();
    loadData();
    return;
  }
  if (state.tree.subject) {
    state.tree.subject = '';
  } else if (state.tree.type) {
    state.tree = { type: '', subject: '' };
  } else if (state.filters.status) {
    state.filters.status = '';
    $('#status-filter').value = '';
    resetMemoryLimit();
    loadData();
    return;
  }
  resetMemoryLimit();
  state.selected = null;
  renderFacts();
  renderDetail(null);
}

function isTechnicalEpisode(episode) {
  return /^\s*<(?:task-notification|message_quote|user_info|project_context)\b/i.test(String(episode.text || ''));
}

function renderEpisodes() {
  const grid = $('#episode-agent-grid');
  const list = $('#episode-list');
  const breadcrumb = $('#episode-breadcrumb');
  const back = $('#episode-back');
  if (!grid || !list) return;
  const loaded = state.episodes || [];
  const episodes = loaded.filter((episode) => !isTechnicalEpisode(episode));
  const hiddenCount = loaded.length - episodes.length;
  const selectedAgent = state.episodeAgent;
  $('#episode-count').textContent = `${episodes.length} 条`;
  $('#episode-count').title = hiddenCount ? `已隐藏 ${hiddenCount} 条系统通知` : '';

  breadcrumb.innerHTML = '<button class="memory-crumb" type="button" data-episode-root>最近捕获</button>'
    + (selectedAgent ? `<span class="memory-crumb-sep">/</span><span class="memory-crumb is-current">${escapeHtml(displaySource(selectedAgent))}</span>` : '');
  back.disabled = !selectedAgent;

  if (!episodes.length) {
    grid.innerHTML = '<div class="empty-state"><h3>还没有捕获记录</h3><p>Agent 产生新对话后，原始用户回合会出现在这里。</p></div>';
    grid.classList.remove('is-hidden');
    list.classList.add('is-hidden');
    list.innerHTML = '';
    return;
  }

  const groups = new Map();
  for (const episode of episodes) {
    const agent = episode.agent || 'unknown';
    const group = groups.get(agent) || { count: 0, latest: '', preview: '' };
    group.count += 1;
    if (!group.latest || String(episode.ts) > group.latest) {
      group.latest = String(episode.ts || '');
      group.preview = String(episode.text || '');
    }
    groups.set(agent, group);
  }

  if (!selectedAgent) {
    grid.classList.remove('is-hidden');
    list.classList.add('is-hidden');
    list.innerHTML = '';
    const rows = [...groups.entries()].sort((a, b) => b[1].count - a[1].count || displaySource(a[0]).localeCompare(displaySource(b[0]), 'zh'));
    grid.innerHTML = `<div class="memory-folder-head"><div><span class="memory-folder-level">Agent</span><h3>选择一个来源</h3><p>只展示最近 ${episodes.length} 条捕获；进入后查看该 Agent 的原始回合。</p></div><span class="memory-folder-total">${rows.length} 个 Agent · ${episodes.length} 条</span></div>
      <div class="episode-agent-list">${rows.map(([agent, group]) => `
        <button class="memory-subject-folder episode-agent-folder" type="button" data-episode-agent="${escapeHtml(agent)}">
          <span class="memory-subject-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><rect x="5" y="7" width="14" height="11" rx="3"/><path d="M9 12h.01M15 12h.01M9 15h6M12 4v3"/></svg></span>
          <span class="memory-subject-copy"><strong>${escapeHtml(displaySource(agent))}</strong><small>${escapeHtml(formatTime(group.latest))} · ${escapeHtml(shortToken(group.preview, 42))}</small></span>
          <span class="memory-subject-count">${group.count}</span>
          <span class="memory-subject-arrow" aria-hidden="true">→</span>
        </button>`).join('')}</div>`;
    return;
  }

  grid.classList.add('is-hidden');
  list.classList.remove('is-hidden');
  const rows = episodes.filter((episode) => (episode.agent || 'unknown') === selectedAgent);
  $('#episode-count').textContent = `${rows.length} 条`;
  list.innerHTML = rows.length ? rows.map((episode) => `
    <div class="episode-row"><span class="episode-dot"></span><time class="episode-time">${formatTime(episode.ts)}</time><span class="episode-text" title="${escapeHtml(episode.text)}">${escapeHtml(episode.text)}</span></div>`).join('')
    : '<div class="empty-state"><h3>这个 Agent 没有捕获记录</h3><p>换一个来源或等待新的对话。</p></div>';
}

function renderDetail(fact) {
  const content = $('#detail-content');
  const panel = $('#detail-panel');
  const grid = $('#memories');
  if (!fact) {
    panel?.classList.add('is-hidden');
    grid?.classList.remove('has-detail');
    $('#detail-empty').classList.remove('is-hidden');
    content.classList.add('is-hidden');
    return;
  }
  panel?.classList.remove('is-hidden');
  grid?.classList.add('has-detail');
  $('#detail-empty').classList.add('is-hidden');
  content.classList.remove('is-hidden');
  const confidence = Math.round((fact.confidence || 0) * 100);
  const evidence = (fact.evidence || []).map((item) => `<div class="evidence-item"><p class="evidence-quote">“${escapeHtml(item.quote)}”</p><span class="evidence-meta">${escapeHtml(item.episode)} · ${formatDate(item.date)} · ${escapeHtml(item.origin)}</span></div>`).join('');
  const canAct = fact.status !== 'retracted' && fact.status !== 'superseded';
  content.innerHTML = `
    <div class="detail-top"><div><div class="memory-line"><span class="memory-folder-level">L2 详情</span>${typeTag(fact.type)}${statusBadge(fact.status)}</div><h2 class="detail-title">${escapeHtml(fact.claim)}</h2><p class="detail-id">${escapeHtml(fact.id)}</p></div><button class="icon-button detail-close" id="detail-close" type="button" title="关闭详情" aria-label="关闭详情"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg></button></div>
    <div class="detail-actions">${canAct ? `<button class="button button-outline" data-action="correct">${icon('edit')}纠正</button><button class="button button-outline" data-action="contest">${icon('flag')}标记冲突</button><button class="button button-danger" data-action="retract">${icon('undo')}撤回</button>${fact.status === 'contested' ? `<button class="button button-outline" data-action="resolve">解决冲突</button>` : ''}` : '<span class="panel-meta">此记忆已结束当前生命周期</span>'}</div>
    <div class="detail-section"><div class="detail-section-label">Confidence</div><div class="confidence-row"><div class="confidence-track"><div class="confidence-fill ${widthBucket(confidence)}"></div></div><span class="confidence-value">${confidence}%</span></div></div>
    <div class="detail-section"><div class="detail-section-label">原文证据 · ${fact.evidence?.length ?? 0}</div>${evidence || '<p class="panel-subtitle">没有证据。</p>'}</div>
    <div class="detail-section"><div class="detail-meta"><div><span class="meta-label">来源 Agent</span><span class="meta-value">${escapeHtml(displaySource(fact.source?.agent || 'unknown'))}</span></div><div><span class="meta-label">会话</span><span class="meta-value">${escapeHtml(fact.source?.session || '—')}</span></div><div><span class="meta-label">首次记录</span><span class="meta-value">${formatDate(fact.created_at)}</span></div><div><span class="meta-label">有效起点</span><span class="meta-value">${formatDate(fact.valid_from)}</span></div></div></div>`;
  content.querySelectorAll('[data-action]').forEach((button) => button.addEventListener('click', () => openAction(button.dataset.action, fact)));
  content.querySelector('#detail-close')?.addEventListener('click', () => {
    state.selected = null;
    renderFacts();
    renderDetail(null);
  });
}

async function loadData() {
  const params = new URLSearchParams();
  if (state.filters.q) params.set('q', state.filters.q);
  if (state.filters.status) params.set('status', state.filters.status);
  if (state.filters.type) params.set('type', state.filters.type);
  const settled = await Promise.allSettled([
    fetchJson('/api/overview'), fetchJson(`/api/facts?${params.toString()}`), fetchJson('/api/episodes?limit=100'), fetchJson('/api/model'),
  ]);
  const [overviewResult, factsResult, episodesResult, modelResult] = settled;
  const failures = settled.filter((result) => result.status === 'rejected');
  const overview = overviewResult.status === 'fulfilled' ? overviewResult.value : null;
  const facts = factsResult.status === 'fulfilled' ? factsResult.value : null;
  const episodes = episodesResult.status === 'fulfilled' ? episodesResult.value : null;
  const model = modelResult.status === 'fulfilled' ? modelResult.value : null;
  if (overview) renderOverview(overview);
  if (model) renderModel(model);
  if (facts) {
    state.facts = facts.items || [];
    state.factTotal = Number.isFinite(facts.count) ? facts.count : state.facts.length;
    state.factsTruncated = Boolean(facts.truncated);
  }
  if (episodes) {
    state.episodes = episodes.items || [];
    state.episodeTotal = Number.isFinite(episodes.count) ? episodes.count : state.episodes.length;
  }
  if (facts || episodes) { renderFacts(); renderEpisodes(); }
  if (state.selected) {
    const updated = state.facts.find((fact) => fact.id === state.selected.id);
    if (updated) { state.selected = updated; renderDetail(updated); }
    else { state.selected = null; renderDetail(null); }
  }
  if (failures.length) {
    const first = failures[0].reason;
    $('#system-label').textContent = overview ? '部分数据加载失败' : '服务连接失败';
    $('#system-dot').classList.remove('is-online');
    $('#system-detail').textContent = '请确认后台服务正在运行';
    notify(first?.message || '部分数据加载失败，请稍后重试。', true);
  }
}

function newChat() {
  localStorage.removeItem('membrain.activeChat');
  localStorage.removeItem('membrain.lastMemory');
  state.chat = { sessionId: null, messages: [], memory: [], sending: false };
  renderChat();
  renderMemorySources();
  $('#chat-input').focus();
}

async function sendChat(event) {
  event.preventDefault();
  const input = $('#chat-input');
  const message = input.value.trim();
  if (!message || state.chat.sending) return;
  if (state.model && !state.model.configured) {
    notify('请先配置并测试一个 LLM 模型。', true);
    return;
  }
  if (!state.chat.sessionId) {
    state.chat.sessionId = globalThis.crypto?.randomUUID?.() || `chat-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    localStorage.setItem('membrain.activeChat', state.chat.sessionId);
  }
  state.chat.sending = true;
  state.chat.messages.push({ role: 'user', content: message });
  input.value = '';
  renderChat();
  try {
    const result = await fetchJson('/api/chat', { method: 'POST', body: JSON.stringify({ session_id: state.chat.sessionId, message }) });
    state.chat.sessionId = result.session_id;
    state.chat.messages.push({ role: 'assistant', content: result.answer || '模型没有返回文字。' });
    state.chat.memory = result.memory || [];
    try { localStorage.setItem('membrain.lastMemory', JSON.stringify(state.chat.memory)); } catch { /* storage may be disabled */ }
    renderChat();
    renderMemorySources();
    if (result.memory_action?.ok) {
      notify(result.memory_action.action === 'correct' ? '记忆已更正并生效' : '记忆已撤回');
      await loadData();
    } else {
      notify(`已回答 · 召回 ${state.chat.memory.length} 条记忆`);
    }
  } catch (error) {
    state.chat.messages.pop();
    input.value = message;
    renderChat();
    notify(error.message, true);
  } finally {
    state.chat.sending = false;
    renderChat();
  }
}

async function restoreChat() {
  const sessionId = localStorage.getItem('membrain.activeChat');
  if (!sessionId) return;
  try {
    const saved = await fetchJson(`/api/chat/${encodeURIComponent(sessionId)}`);
    state.chat.sessionId = saved.session_id;
    state.chat.messages = saved.messages || [];
    try {
      const memory = JSON.parse(localStorage.getItem('membrain.lastMemory') || '[]');
      state.chat.memory = Array.isArray(memory) ? memory : [];
    } catch { state.chat.memory = []; }
    renderChat();
    renderMemorySources();
  } catch (error) {
    localStorage.removeItem('membrain.activeChat');
    localStorage.removeItem('membrain.lastMemory');
  }
}

async function testModel() {
  state.model = state.model || {};
  const button = $('#model-test-button');
  button.disabled = true;
  $('#model-result').textContent = '正在请求模型…';
  $('#model-status-badge').className = 'status-badge status-contested';
  $('#model-status-badge').textContent = '测试中';
  try {
    const result = await fetchJson('/api/model/test', { method: 'POST', body: '{}' });
    state.model.connected = true;
    $('#model-dot').classList.add('is-online');
    $('#model-status-badge').className = 'status-badge status-active';
    $('#model-status-badge').textContent = '连接正常';
    $('#model-result').textContent = `连接正常 · ${result.elapsed_ms} ms · ${result.answer}`;
    notify('模型连接正常');
  } catch (error) {
    state.model.connected = false;
    $('#model-dot').classList.remove('is-online');
    $('#model-status-badge').className = 'status-badge status-retracted';
    $('#model-status-badge').textContent = '连接失败';
    $('#model-result').textContent = error.message;
    notify(error.message, true);
  } finally {
    button.disabled = !state.model?.configured;
  }
}

async function saveModelConfig(event) {
  event.preventDefault();
  const form = $('#model-config-form');
  const button = form.querySelector('button[type="submit"]');
  const data = Object.fromEntries(new FormData(form).entries());
  data.clear_api_key = $('#model-config-clear-key').checked;
  button.disabled = true;
  button.textContent = '正在保存…';
  try {
    const result = await fetchJson('/api/model/config', { method: 'POST', body: JSON.stringify(data) });
    $('#model-config-key').value = '';
    $('#model-config-clear-key').checked = false;
    renderModel(result);
    $('#model-result').textContent = '配置已保存。请测试连接后开始对话。';
    notify('模型配置已保存');
  } catch (error) {
    $('#model-result').textContent = error.message;
    notify(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = '保存模型配置';
  }
}

async function testEmbedding() {
  state.model = state.model || {};
  const button = $('#embedding-test-button');
  button.disabled = true;
  $('#embedding-result').textContent = '正在请求 Embedding 接口…';
  $('#embedding-status-badge').className = 'status-badge status-contested';
  $('#embedding-status-badge').textContent = '测试中';
  try {
    const result = await fetchJson('/api/model/embedding/test', { method: 'POST', body: '{}' });
    $('#embedding-dot').classList.add('is-online');
    $('#embedding-status-badge').className = 'status-badge status-active';
    $('#embedding-status-badge').textContent = '连接正常';
    $('#embedding-result').textContent = `连接正常 · ${result.elapsed_ms} ms · 向量维度 ${result.dimensions}`;
    notify('Embedding 连接正常');
  } catch (error) {
    $('#embedding-dot').classList.remove('is-online');
    $('#embedding-status-badge').className = 'status-badge status-retracted';
    $('#embedding-status-badge').textContent = '连接失败';
    $('#embedding-result').textContent = error.message;
    notify(error.message, true);
  } finally {
    button.disabled = !state.model?.embedding_configured;
  }
}

async function saveEmbeddingConfig(event) {
  event.preventDefault();
  const form = $('#embedding-config-form');
  const button = form.querySelector('button[type="submit"]');
  const data = Object.fromEntries(new FormData(form).entries());
  data.clear_api_key = $('#embedding-config-clear-key').checked;
  button.disabled = true;
  button.textContent = '正在保存…';
  try {
    const result = await fetchJson('/api/model/embedding/config', { method: 'POST', body: JSON.stringify(data) });
    $('#embedding-config-key').value = '';
    $('#embedding-config-clear-key').checked = false;
    renderEmbedding(result);
    $('#embedding-result').textContent = '配置已保存。请测试连接。更换模型后需重新索引。';
    notify('Embedding 配置已保存');
  } catch (error) {
    $('#embedding-result').textContent = error.message;
    notify(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = '保存 Embedding 配置';
  }
}

async function selectFact(id) {
  try {
    state.selected = await fetchJson(`/api/facts/${encodeURIComponent(id)}`);
    renderFacts(); renderDetail(state.selected);
    if (window.matchMedia('(max-width: 1120px)').matches) {
      $('#detail-panel')?.scrollIntoView({ block: 'start' });
    }
  } catch (error) { notify(error.message, true); }
}

function field(name, label, value = '', type = 'text', placeholder = '') {
  const input = type === 'textarea' ? `<textarea id="field-${name}" name="${name}" placeholder="${escapeHtml(placeholder)}">${escapeHtml(value)}</textarea>` : `<input id="field-${name}" name="${name}" type="${type}" value="${escapeHtml(value)}" placeholder="${escapeHtml(placeholder)}">`;
  return `<div class="field"><label for="field-${name}">${label}</label>${input}</div>`;
}

function openAction(mode, fact = null) {
  state.dialog = { mode, fact };
  const dialog = $('#action-dialog');
  const fields = $('#dialog-fields');
  const copy = $('#dialog-copy');
  const submit = $('#dialog-submit');
  if (mode === 'add') {
    $('#dialog-eyebrow').textContent = 'NEW MEMORY'; $('#dialog-title').textContent = '添加记忆'; copy.textContent = '填写你亲自说过的内容。它会生成一个新的事实和可追溯证据。'; submit.textContent = '保存记忆';
    fields.innerHTML = '<div class="field"><label for="field-type">类型</label><select id="field-type" name="type"><option value="preference">偏好</option><option value="fact">事实</option><option value="goal">目标</option><option value="project">项目</option></select></div>' + field('subject', '主题', '', 'text', '例如 communication') + field('claim', '记忆内容', '', 'textarea', '用一句话描述这条记忆') + field('quote', '你的原话', '', 'textarea', '这条记忆来自你说过的哪句话？');
  } else if (mode === 'correct') {
    $('#dialog-eyebrow').textContent = 'REPLACE MEMORY'; $('#dialog-title').textContent = '纠正这条记忆'; copy.textContent = '原事实保持不变，系统会新增一条事实并将旧事实标记为已替代。'; submit.textContent = '保存替代事实';
    fields.innerHTML = field('claim', '新的记忆内容', fact.claim, 'textarea', '当前正确的说法') + field('quote', '你的更正原话', '', 'textarea', '你亲自说出的更正内容');
  } else if (mode === 'contest') {
    $('#dialog-eyebrow').textContent = 'CONFLICT REVIEW'; $('#dialog-title').textContent = '标记为冲突'; copy.textContent = '这条记忆会被保留，但召回时会明确提示存在冲突。'; submit.textContent = '标记冲突';
    fields.innerHTML = field('note', '备注', '', 'textarea', '例如：与最近一次陈述不一致');
  } else if (mode === 'retract') {
    $('#dialog-eyebrow').textContent = 'RETRACT MEMORY'; $('#dialog-title').textContent = '撤回这条记忆'; copy.textContent = '撤回只改变生命周期状态，原始事实文件和历史证据仍然保留。'; submit.textContent = '确认撤回';
    fields.innerHTML = field('reason', '撤回原因', '', 'textarea', '例如：这条信息已经不再适用');
  } else if (mode === 'resolve') {
    $('#dialog-eyebrow').textContent = 'RESOLVE CONFLICT'; $('#dialog-title').textContent = '解决冲突'; copy.textContent = '选择保留当前记忆，另一条将被撤回。'; submit.textContent = '保留当前记忆';
    const choices = state.facts.filter((item) => item.status === 'contested' && item.id !== fact.id).map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.claim)}</option>`).join('');
    fields.innerHTML = `<div class="field"><label for="field-drop">撤回另一条</label><select id="field-drop" name="drop">${choices || '<option value="">没有其他冲突记忆</option>'}</select></div>`;
  }
  submit.disabled = false;
  dialog.showModal();
}

let actionSubmitting = false;
async function submitAction(event) {
  event.preventDefault();
  if (event.submitter?.value === 'cancel') { $('#action-dialog').close(); return; }
  if (actionSubmitting) return;
  actionSubmitting = true;
  const submit = $('#dialog-submit');
  submit.disabled = true;
  const form = new FormData($('#action-form'));
  const data = Object.fromEntries(form.entries());
  const { mode, fact } = state.dialog || {};
  try {
    let endpoint = '/api/facts';
    if (mode === 'correct') endpoint = `/api/facts/${encodeURIComponent(fact.id)}/correct`;
    if (mode === 'contest') endpoint = `/api/facts/${encodeURIComponent(fact.id)}/contest`;
    if (mode === 'retract') endpoint = `/api/facts/${encodeURIComponent(fact.id)}/retract`;
    if (mode === 'resolve') { endpoint = '/api/facts/resolve'; data.keep = fact.id; }
    await fetchJson(endpoint, { method: 'POST', body: JSON.stringify(data) });
    $('#action-dialog').close();
    notify(mode === 'retract' ? '记忆已撤回' : mode === 'contest' ? '已标记为冲突' : mode === 'correct' ? '已保存替代事实' : '记忆已保存');
    state.selected = null;
    renderDetail(null);
    await loadData();
  } catch (error) { notify(error.message, true); }
  finally {
    actionSubmitting = false;
    submit.disabled = false;
  }
}

async function startExtract() {
  try {
    const job = await fetchJson('/api/jobs/extract', { method: 'POST', body: '{}' });
    notify('正在处理已捕获的对话…');
    pollExtract(job.id);
  } catch (error) { notify(error.message, true); }
}

async function pollExtract(jobId) {
  const started = Date.now();
  while (Date.now() - started < 180000) {
    await new Promise((resolve) => window.setTimeout(resolve, 1500));
    try {
      const job = await fetchJson('/api/jobs/extract');
      if (job.id !== jobId || job.status === 'idle' || job.status === 'running') continue;
      if (job.status === 'error') notify(job.error || '提取失败，请检查模型连接。', true);
      else notify(`提取完成：新增 ${job.report?.facts_added ?? 0}，更新 ${job.report?.corroborated ?? 0} 条记忆`);
      await loadData();
      return;
    } catch (error) { notify(error.message, true); return; }
  }
  notify('提取仍在后台运行；完成后可刷新页面查看。');
}

async function reindex() {
  try { const result = await fetchJson('/api/reindex', { method: 'POST', body: '{}' }); notify(`索引已重建：${result.facts ?? 0} 条事实`); await loadData(); } catch (error) { notify(error.message, true); }
}

const agentStatus = {
  connected: ['已接入', 'is-ok'], partial: ['部分接入', 'is-warn'], not_connected: ['未接入', ''],
  unreadable: ['配置特殊', 'is-warn'], manual: ['需手动接入', ''], not_installed: ['未安装', 'is-off'],
};

function renderAgents(items) {
  state.agents = items;
  const grid = $('#agent-grid');
  if (!items.length) { grid.innerHTML = '<div class="side-empty">没有扫描到 Agent。</div>'; return; }
  grid.innerHTML = items.map((agent) => {
    const [label, tone] = agentStatus[agent.status] || [agent.status, ''];
    const capture = agent.capture === 'hooks' ? '自动捕获并注入' : '按需召回；写入靠模型调用记录工具';
    const connect = agent.can_connect && agent.status !== 'connected' && agent.status !== 'unreadable'
      ? `<button class="button button-dark" type="button" data-agent-connect="${escapeHtml(agent.key)}">一键接入</button>` : '';
    return `<article class="agent-card${agent.installed ? '' : ' is-off'}">
      <div class="agent-card-head"><strong>${escapeHtml(agent.name)}</strong><span class="agent-chip ${tone}">${label}</span></div>
      <p class="agent-note">${capture}${agent.protected ? '；配置受保护' : ''}</p>
      <div class="agent-actions">${connect}<button class="button button-outline" type="button" data-agent-copy="${escapeHtml(agent.key)}">复制提示词</button></div>
    </article>`;
  }).join('');
  $$('[data-agent-connect]').forEach((button) => button.addEventListener('click', () => connectAgent(button.dataset.agentConnect, button)));
  $$('[data-agent-copy]').forEach((button) => button.addEventListener('click', () => copyPrompt(button.dataset.agentCopy)));
}

async function loadAgents() {
  try { renderAgents((await fetchJson('/api/agents')).agents || []); }
  catch (error) {
    $('#agent-grid').innerHTML = `<div class="empty-state"><div class="empty-mark">A</div><h3>扫描失败</h3><p>${escapeHtml(error.message)}</p></div>`;
    notify(error.message, true);
  }
}

async function copyAllPrompts() {
  const items = (state.agents || []).filter((a) => a.prompt && a.status !== 'connected' && a.status !== 'not_installed');
  if (!items.length) { notify('没有需要导出的接入提示词'); return; }
  const text = items.map((a) => `===== ${a.name} =====\n${a.prompt}`).join('\n\n');
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = Object.assign(document.createElement('textarea'), { value: text });
    document.body.append(area); area.select(); document.execCommand('copy'); area.remove();
  }
  notify(`已复制 ${items.length} 个 Agent 的接入提示词，逐个粘贴即可`);
}

async function connectAgent(key, button) {
  button.disabled = true;
  try {
    const result = await fetchJson(`/api/agents/${encodeURIComponent(key)}/connect`, { method: 'POST', body: '{}' });
    notify(result.changed ? `已接入，原配置已备份为 ${result.backup || '（新建文件）'}。重启该软件后生效` : '已经接入过了');
    await loadAgents();
  } catch (error) { notify(error.message, true); button.disabled = false; }
}

async function copyPrompt(key) {
  const agent = (state.agents || []).find((item) => item.key === key);
  if (!agent) return;
  try {
    await navigator.clipboard.writeText(agent.prompt);
  } catch {
    const area = Object.assign(document.createElement('textarea'), { value: agent.prompt });
    document.body.append(area); area.select(); document.execCommand('copy'); area.remove();
  }
  notify('提示词已复制，粘贴给该 Agent 即可');
}

const importStages = { preparing: '正在准备', cleaning: 'LLM 识别分类清洗中', indexing: '正在建立索引', distilling: '正在提炼短记忆' };
state.lib = { labels: {}, sourceLabels: {}, treeCounts: {}, lastReport: null, poll: null, request: 0, importLoaded: false, importLoading: false, items: [], notes: [], selected: null, tree: { source: '', category: '' }, treeOpen: {} };

function librarySourceLabel(source) {
  return state.lib.sourceLabels[source] || source || '未知来源';
}

function libraryScopeCount() {
  const counts = state.lib.treeCounts || {};
  const source = state.lib.tree.source;
  const category = state.lib.tree.category;
  if (!Object.keys(counts).length) return null;
  if (!source) return Object.values(counts).reduce((sum, cats) => sum + Object.values(cats).reduce((s, n) => s + n, 0), 0);
  if (!category) return Object.values(counts[source] || {}).reduce((sum, n) => sum + n, 0);
  return counts[source]?.[category] ?? 0;
}

function categoryChips(counts) {
  return Object.entries(counts || {}).map(([key, n]) => `<span class="agent-chip">${escapeHtml(state.lib.labels[key] || key)} ${n}</span>`).join('');
}

function renderSessionImport(prev) {
  const card = $('#session-import-card');
  if (!card) return;
  const rows = (prev.sources || []).map((s) => `<div class="batch-row"><span>${escapeHtml(s.label || s.source)}　发现 ${s.found}　已导入 ${s.already}　噪声跳过 ${s.noise || 0}</span></div>`).join('');
  const actionable = (prev.new || 0) > 0;
  const actionLabel = !(prev.total > 0) ? '未发现可导入会话' : actionable ? '一键导入全部会话' : '已全部导入';
  card.innerHTML = `<h3>Agent 会话导入 · LLM 清洗提炼</h3>
    <p class="import-note">自动发现本机 Agent 的历史会话（源只读），一键导入后由 LLM 识别分类、生成标题与 L0 摘要；纯问候/空会话/系统回执等噪声直接丢弃。落库后按 Agent→类别出现在资料库。</p>
    <div class="import-actions"><button class="button button-dark" id="session-import-start" type="button" ${actionable ? '' : 'disabled'}>${actionLabel}</button>
    <label class="check-inline"><input type="checkbox" id="session-import-distill"> 同时提炼为短记忆（默认关，防闲聊污染）</label>
    <span class="import-note" id="session-import-status"></span></div>
    ${rows}`;
  $('#session-import-start')?.addEventListener('click', startSessionImport);
}

async function startSessionImport() {
  const button = $('#session-import-start');
  if (button) button.disabled = true;
  try {
    await fetchJson('/api/import/sessions/start', {
      method: 'POST',
      body: JSON.stringify({ distill_facts: Boolean($('#session-import-distill')?.checked) }),
    });
    pollImport('sessions');
  } catch (error) {
    notify(error.message, true);
    if (button) button.disabled = false;
  }
}

function renderImportCard(preview, batches) {
  const card = $('#import-card');
  const report = state.lib.lastReport;
  const done = report ? `<p class="import-note">刚完成：新放入资料库 ${report.archived_new} 篇，提炼出 ${report.facts_added} 条个人偏好${
    report.errors?.length ? `，另有 ${report.errors.length} 篇因模型限流没能提炼，再点一次导入即可补上（已导入的不会重复）` : ''}。</p>` : '';
  if (!(preview.total > 0) && !batches?.length) {
    let note;
    if (!preview.configured) {
      note = '未配置 OpenViking 数据源。它是可选的旧记忆来源；没有它一切功能照常，历史会话请用上方的「Agent 会话导入」。';
    } else if (!preview.available) {
      note = `已配置的 OpenViking 路径不可用：${preview.root || '（空路径）'}。请检查 MEMBRAIN_OPENVIKING_ROOT 是否指向包含 memories/ 的用户目录。`;
    } else {
      note = `已连接 ${preview.root || 'OpenViking 数据源'}，但暂时没有可导入的笔记。`;
    }
    card.innerHTML = `<h3>从 OpenViking 导入旧记忆</h3><p class="import-note">${escapeHtml(note)}</p>`;
    return;
  }
  const actionable = preview.new > 0 || preview.to_distill > 0;
  const rows = (batches || []).map((b) => `<div class="batch-row"><span>${escapeHtml(formatTime(b.created))}　资料 ${b.archived}　偏好 ${b.facts}${b.undone ? '　已撤销' : ''}</span>${
    b.undone ? '' : `<button class="button button-outline" type="button" data-undo="${escapeHtml(b.batch)}">撤销这一批</button>`}</div>`).join('');
  card.innerHTML = `<h3>从 OpenViking 导入旧记忆</h3>
    <p class="import-note">来源：${escapeHtml(preview.root)}（只读，不会改动它）。共找到 ${preview.total} 篇，其中 ${preview.already_imported} 篇已导入。
    点击后会按类别自动归档到资料库，其中的个人偏好和规则会再提炼成短记忆。</p>
    <div class="import-stats">${categoryChips(preview.by_category)}</div>
    <div class="import-actions"><button class="button button-dark" id="import-start" type="button" ${actionable ? '' : 'disabled'}>${
      actionable ? '一键导入' : '没有待导入内容'}</button><span class="import-note" id="import-status"></span></div>
    ${done}${rows}`;
  $('#import-start').addEventListener('click', startImport);
  $$('[data-undo]').forEach((button) => button.addEventListener('click', () => undoImport(button.dataset.undo, button)));
}

async function loadImport() {
  if (state.lib.importLoading) return;
  state.lib.importLoading = true;
  const foldCount = $('#import-fold-count');
  if (foldCount && !state.lib.importLoaded) foldCount.textContent = '正在扫描…';
  const settled = await Promise.allSettled([
    fetchJson('/api/import/sessions/preview'), fetchJson('/api/import/preview'), fetchJson('/api/import/batches')]);
  const [sessionResult, previewResult, batchResult] = settled;
  const sessions = sessionResult.status === 'fulfilled' ? sessionResult.value : null;
  const preview = previewResult.status === 'fulfilled' ? previewResult.value : null;
  const batches = batchResult.status === 'fulfilled' ? batchResult.value : null;
  if (sessions) {
    state.lib.labels = preview?.labels || sessions.labels || state.lib.labels;
    renderSessionImport(sessions);
  } else {
    $('#session-import-card').innerHTML = `<h3>Agent 会话导入</h3><p class="import-note">${escapeHtml(sessionResult.reason?.message || '扫描失败')}</p>`;
  }
  if (preview && batches) renderImportCard(preview, batches.batches);
  else $('#import-card').innerHTML = `<h3>从 OpenViking 导入旧记忆</h3><p class="import-note">${escapeHtml(previewResult.reason?.message || batchResult.reason?.message || '读取失败')}</p>`;
  if (foldCount && sessions) {
    const ovPart = preview?.total > 0 ? ` · OV ${preview.already_imported}/${preview.total}` : '';
    foldCount.textContent = `会话 ${sessions.already_imported}/${sessions.total}（噪声 ${sessions.noise || 0}）${ovPart}`;
    const hint = $('#import-fold')?.querySelector('.import-fold-hint');
    if (hint) hint.textContent = `自动发现 ${sessions.sources?.length || 0} 个会话源${preview?.total > 0 ? ' + OpenViking' : ''} · 展开管理`;
  } else if (foldCount) {
    foldCount.textContent = '扫描失败';
  }
  state.lib.importLoaded = Boolean(sessions);
  state.lib.importLoading = false;
  const failed = settled.find((result) => result.status === 'rejected');
  if (failed) notify(failed.reason?.message || '部分导入信息读取失败。', true);
}

async function startImport() {
  const button = $('#import-start');
  button.disabled = true;
  try {
    await fetchJson('/api/import/start', { method: 'POST', body: '{}' });
    pollImport('openviking');
  } catch (error) { notify(error.message, true); button.disabled = false; }
}

function pollImport(kind = '') {
  window.clearTimeout(state.lib.poll);
  state.lib.poll = window.setTimeout(async () => {
    try {
      const job = await fetchJson('/api/import/job');
      const actualKind = job.kind || kind;
      const statusNode = $(actualKind === 'sessions' ? '#session-import-status' : '#import-status');
      if (job.status === 'running') {
        const p = job.progress || {};
        const msg = `${importStages[p.stage] || '处理中'}${p.total ? `（${p.done}/${p.total}）` : ''}，请稍候…`;
        if (statusNode) statusNode.textContent = msg;
        pollImport(actualKind);
        return;
      }
      if (job.status === 'done') { state.lib.lastReport = job.report; notify('导入完成'); }
      else if (job.status === 'error') notify(`导入没有完成：${job.error}`, true);
      await Promise.all([loadImport(), loadLibrary(), loadData()]);
    } catch (error) { notify(error.message, true); }
  }, 1500);
}


async function undoImport(batch, button) {
  if (!window.confirm('撤销这一批导入？资料库里这一批的内容会被隐藏，提炼出的偏好会被撤回，文件本身不会删除。')) return;
  button.disabled = true;
  try {
    const result = await fetchJson('/api/import/undo', { method: 'POST', body: JSON.stringify({ batch }) });
    notify(`已撤销：隐藏资料 ${result.archive_hidden} 篇，撤回偏好 ${result.facts_retracted} 条`);
    state.lib.lastReport = null;
    await Promise.all([loadImport(), loadLibrary(), loadData()]);
  } catch (error) { notify(error.message, true); button.disabled = false; }
}

function renderLibraryTree() {
  const root = $('#library-tree');
  if (!root) return;
  const groups = new Map();
  if (Object.keys(state.lib.treeCounts || {}).length) {
    for (const [source, cats] of Object.entries(state.lib.treeCounts)) groups.set(source, new Map(Object.entries(cats)));
  } else {
    for (const item of state.lib.items) {
      if (!groups.has(item.source)) groups.set(item.source, new Map());
      const cats = groups.get(item.source);
      cats.set(item.category, (cats.get(item.category) || 0) + 1);
    }
  }
  const catLabel = (c) => state.lib.labels[c] || c;
  const sources = [...groups.keys()].sort((a, b) => librarySourceLabel(a).localeCompare(librarySourceLabel(b), 'zh'));
  const allCount = [...groups.values()].reduce((sum, cats) => sum + [...cats.values()].reduce((s, n) => s + n, 0), 0);
  const searchActive = Boolean($('#library-search')?.value.trim());
  let html = `<button class="tree-folder${state.lib.tree.source ? '' : ' is-active'}" type="button" data-lib-source=""><span class="tree-label">${searchActive ? '搜索结果' : '全部资料'}</span><span class="tree-count">${allCount}</span></button>`;
  for (const source of sources) {
    const cats = groups.get(source);
    const total = [...cats.values()].reduce((s, n) => s + n, 0);
    const open = Boolean(state.lib.treeOpen[source]);
    const active = state.lib.tree.source === source && !state.lib.tree.category;
    html += `<button class="tree-folder${active ? ' is-active' : ''}" type="button" data-lib-source="${escapeHtml(source)}"><span class="tree-chevron" data-lib-toggle="${escapeHtml(source)}">${open ? '▾' : '▸'}</span><span class="tree-label">${escapeHtml(librarySourceLabel(source))}</span><span class="tree-count">${total}</span></button>`;
    if (open) {
      html += '<div class="tree-children">';
      for (const category of [...cats.keys()].sort((a, b) => catLabel(a).localeCompare(catLabel(b), 'zh'))) {
        const leaf = state.lib.tree.source === source && state.lib.tree.category === category;
        html += `<button class="tree-folder tree-leaf${leaf ? ' is-active' : ''}" type="button" data-lib-source="${escapeHtml(source)}" data-lib-category="${escapeHtml(category)}"><span class="tree-label">${escapeHtml(catLabel(category))}</span><span class="tree-count">${cats.get(category)}</span></button>`;
      }
      html += '</div>';
    }
  }
  root.innerHTML = html || '<div class="side-empty">暂无资料。</div>';
}

function renderLibraryList() {
  const list = $('#library-list');
  const rows = state.lib.items.filter((item) => (!state.lib.tree.source || item.source === state.lib.tree.source) && (!state.lib.tree.category || item.category === state.lib.tree.category));
  const searching = Boolean($('#library-search')?.value.trim());
  $('#library-count').textContent = searching ? rows.length : (libraryScopeCount() ?? rows.length);
  if (!rows.length) {
    list.innerHTML = state.lib.notes.length
      ? `<div class="side-empty">${escapeHtml(state.lib.notes.join('；'))}</div>`
      : '<div class="empty-state library-empty"><div class="empty-mark">L0</div><h3>资料库还是空的</h3><p>展开上方的“数据导入”，可以导入本机 Agent 历史会话，或配置可选的历史来源。</p></div>';
    return;
  }
  list.innerHTML = rows.map((item) => `<button type="button" class="library-item${state.lib.selected === item.id ? ' is-selected' : ''}" data-entry="${escapeHtml(item.id)}" aria-label="打开资料：${escapeHtml(item.title)}">
    <strong>${escapeHtml(item.title)}</strong>
    <div class="library-meta"><span class="agent-chip">${escapeHtml(state.lib.labels[item.category] || item.category)}</span><span>${escapeHtml(item.date)}</span><span>${item.chars} 字</span>${
      item.similarity ? `<span>相关度 ${item.similarity}</span>` : ''}</div>
    <p class="library-abstract">${escapeHtml(item.abstract)}</p></button>`).join('')
    + (state.lib.notes.length ? `<div class="side-empty">${escapeHtml(state.lib.notes.join('；'))}</div>` : '');
  $$('.library-item').forEach((node) => node.addEventListener('click', () => selectLibraryEntry(node.dataset.entry)));
}

async function selectLibraryEntry(id, offset = 0) {
  state.lib.selected = id;
  const list = $('#library-list');
  const scrollTop = list.scrollTop;
  renderLibraryList();
  list.scrollTop = scrollTop;
  const empty = $('#library-detail-empty');
  const content = $('#library-detail-content');
  try {
    const entry = await fetchJson(`/api/archive/${encodeURIComponent(id)}?offset=${Math.max(0, offset)}&limit=8000`);
    if (state.lib.selected !== id) return;
    empty.classList.add('is-hidden');
    content.classList.remove('is-hidden');
    const hasMore = entry.total_chars > entry.offset + entry.text.length;
    const pager = (entry.offset > 0 || hasMore)
      ? `<div class="l2-pager">
          <button class="button button-outline" type="button" data-lib-page="${Math.max(0, entry.offset - 8000)}" ${entry.offset > 0 ? '' : 'disabled'}>上一段</button>
          <span class="panel-meta">${entry.offset + 1}–${entry.offset + entry.text.length} / ${entry.total_chars} 字</span>
          <button class="button button-outline" type="button" data-lib-page="${entry.offset + entry.text.length}" ${hasMore ? '' : 'disabled'}>下一段</button>
        </div>` : '';
    content.innerHTML = `
      <div class="detail-top"><div><div class="memory-line"><span class="agent-chip">${escapeHtml(state.lib.labels[entry.category] || entry.category)}</span>${entry.date ? `<span class="panel-meta">${escapeHtml(entry.date)}</span>` : ''}</div>
      <h2 class="detail-title">${escapeHtml(entry.title)}</h2>
      <p class="detail-id">${escapeHtml(librarySourceLabel(entry.source))} / ${escapeHtml(state.lib.labels[entry.category] || entry.category)} / ${escapeHtml(entry.id)}</p></div></div>
      <div class="detail-section"><div class="detail-section-label">L2 正文 · ${entry.total_chars} 字</div>
        <pre class="library-body">${escapeHtml(entry.text)}</pre>
        ${pager}
        ${hasMore ? '<p class="panel-subtitle">长文已分段；也可让 Agent 用 read_archive 分段读取。</p>' : ''}</div>
      <div class="detail-section"><div class="detail-meta"><div><span class="meta-label">来源</span><span class="meta-value">${escapeHtml(entry.origin || '—')}</span></div><div><span class="meta-label">范围</span><span class="meta-value">${escapeHtml(entry.scope || '—')}</span></div><div><span class="meta-label">日期</span><span class="meta-value">${escapeHtml(entry.date || '—')}</span></div></div></div>`;
    content.querySelectorAll('[data-lib-page]').forEach((btn) => btn.addEventListener('click', () => selectLibraryEntry(id, Number(btn.dataset.libPage))));
  } catch (error) { notify(error.message, true); }
}

async function loadLibrary() {
  const request = ++state.lib.request;
  const q = $('#library-search')?.value.trim() || '';
  try {
    if (q) {
      const result = await fetchJson(`/api/archive?q=${encodeURIComponent(q)}`);
      if (request !== state.lib.request) return;
      state.lib.labels = result.labels || state.lib.labels;
      state.lib.sourceLabels = result.source_labels || state.lib.sourceLabels;
      state.lib.items = result.items || [];
      state.lib.treeCounts = {};
      state.lib.notes = [...(result.notes || []), '搜索最多返回最相关的 20 条；清空搜索可按来源和分类浏览全部'];
    } else {
      const treeRes = await fetchJson('/api/archive');
      state.lib.labels = treeRes.labels || state.lib.labels;
      state.lib.sourceLabels = treeRes.source_labels || state.lib.sourceLabels;
      state.lib.treeCounts = treeRes.tree || {};
      const sources = Object.keys(treeRes.tree || {});
      const settled = await Promise.all(sources.map((s) =>
        fetchJson(`/api/archive?source=${encodeURIComponent(s)}&limit=500`).catch(() => ({ items: [], failed: true }))));
      if (request !== state.lib.request) return;
      state.lib.items = settled.flatMap((r) => r.items || []);
      state.lib.notes = settled.some((result) => result.failed) ? ['部分来源加载失败，可点刷新重试'] : [];
    }
    renderLibraryTree();
    renderLibraryList();
  } catch (error) {
    if (request !== state.lib.request) return;
    $('#library-tree').innerHTML = '<div class="side-empty">分类加载失败。</div>';
    $('#library-list').innerHTML = `<div class="side-empty">${escapeHtml(error.message)}</div>`;
    notify(error.message, true);
  }
}

const VIEW_LABELS = { chat: 'Agent 对话', overview: '记忆总览', memories: '全部记忆', conflicts: '待处理冲突', episodes: '捕获记录', diffs: '会话 Diff', library: '资料库', agents: 'Agent 接入', model: '模型连接' };

function switchView(name, opts = {}) {
  state.currentView = name;
  if (location.hash !== `#/${name}`) history.replaceState(null, '', `#/${name}`);
  const panelName = name === 'conflicts' ? 'memories' : name;
  $$('.view').forEach((panel) => panel.classList.toggle('is-active', panel.dataset.viewPanel === panelName));
  $$('.nav-item').forEach((button) => button.classList.remove('is-active'));
  const navButton = $(`.nav-item[data-view="${name}"]`) || (name === 'conflicts' ? $('.nav-item[data-view="overview"]') : null);
  if (navButton) navButton.classList.add('is-active');
  $('#current-view').textContent = (navButton && (navButton.getAttribute('aria-label') || navButton.textContent.trim())) || VIEW_LABELS[name] || name;
  const hubViews = new Set(['memories', 'conflicts', 'episodes']);
  const showHub = hubViews.has(name);
  $('#crumb-hub')?.classList.toggle('is-hidden', !showHub);
  $('#crumb-hub-sep')?.classList.toggle('is-hidden', !showHub);
  if (name === 'conflicts') {
    state.filters.q = '';
    state.filters.type = '';
    state.filters.status = 'contested';
    state.tree = { type: '', subject: '' };
    state.selected = null;
    $('#search-input').value = '';
    resetMemoryLimit();
    renderDetail(null);
  } else if (name === 'memories') {
    state.filters.q = '';
    state.filters.type = '';
    state.filters.status = opts.status || '';
    state.tree = { type: '', subject: '' };
    state.selected = null;
    $('#search-input').value = '';
    resetMemoryLimit();
    renderDetail(null);
  } else if (name === 'episodes') {
    state.episodeAgent = '';
    state.filters.status = '';
  } else if (opts.status) {
    state.filters.status = opts.status;
  } else {
    state.filters.status = '';
  }
  $('#status-filter').value = state.filters.status;
  window.scrollTo(0, 0);
  loadData();
  if (panelName === 'diffs') loadDiffs();
  if (panelName === 'library') loadLibrary();
  if (panelName === 'agents') loadAgents();
}

function refreshCurrentView() {
  loadData();
  if (state.currentView === 'diffs') loadDiffs();
  if (state.currentView === 'library') loadLibrary();
  if (state.currentView === 'agents') loadAgents();
}

let searchTimer;
function bind() {
  $('#agents-copy-all')?.addEventListener('click', copyAllPrompts);
  $('#agents-scan-button').addEventListener('click', loadAgents);
  let libraryTimer;
  $('#library-search').addEventListener('input', () => { window.clearTimeout(libraryTimer); libraryTimer = window.setTimeout(loadLibrary, 300); });
  $('#refresh-button').addEventListener('click', refreshCurrentView);
  $('#add-button').addEventListener('click', () => openAction('add'));
  $('#extract-button').addEventListener('click', startExtract);
  $('#reindex-button').addEventListener('click', reindex);
  $('#new-chat-button').addEventListener('click', newChat);
  $('#chat-messages').addEventListener('click', (event) => {
    const starter = event.target.closest('[data-chat-prompt]');
    if (!starter) return;
    $('#chat-input').value = starter.dataset.chatPrompt || '';
    $('#chat-input').focus();
  });
  $('#chat-form').addEventListener('submit', sendChat);
  $('#chat-input').addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      $('#chat-form').requestSubmit();
    }
  });
  $('#model-test-button').addEventListener('click', testModel);
  $('#model-config-form').addEventListener('submit', saveModelConfig);
  $('#model-config-clear-key').addEventListener('change', (event) => {
    $('#model-config-key').disabled = event.target.checked;
  });
  $('#embedding-test-button').addEventListener('click', testEmbedding);
  $('#embedding-config-form').addEventListener('submit', saveEmbeddingConfig);
  $('#embedding-config-clear-key').addEventListener('change', (event) => {
    $('#embedding-config-key').disabled = event.target.checked;
  });
  $('#action-form').addEventListener('submit', submitAction);
  $('#dialog-close').addEventListener('click', () => $('#action-dialog').close());
  $('#dialog-cancel').addEventListener('click', () => $('#action-dialog').close());
  $('#search-input').addEventListener('input', (event) => { window.clearTimeout(searchTimer); searchTimer = window.setTimeout(() => { state.filters.q = event.target.value.trim(); resetMemoryLimit(); loadData(); }, 240); });
  $('#status-filter').addEventListener('change', (event) => { state.filters.status = event.target.value; resetMemoryLimit(); loadData(); });
  $('#memory-folder-view')?.addEventListener('click', (event) => {
    const typeButton = event.target.closest('[data-folder-type]');
    const subjectButton = event.target.closest('[data-folder-subject]');
    if (!typeButton && !subjectButton) return;
    state.filters.q = '';
    $('#search-input').value = '';
    state.tree = subjectButton
      ? { type: state.tree.type, subject: subjectButton.dataset.folderSubject || '' }
      : { type: typeButton.dataset.folderType || '', subject: '' };
    state.selected = null;
    resetMemoryLimit();
    renderDetail(null);
    renderFacts();
  });
  $('#memory-breadcrumb')?.addEventListener('click', (event) => {
    const rootButton = event.target.closest('[data-memory-root]');
    const typeButton = event.target.closest('[data-memory-type]');
    const statusButton = event.target.closest('[data-memory-status]');
    if (rootButton) {
      clearMemoryFilters();
      state.tree = { type: '', subject: '' };
      resetMemoryLimit();
      state.selected = null;
      renderDetail(null);
      loadData();
    } else if (statusButton) {
      state.tree = { type: '', subject: '' };
      state.selected = null;
      resetMemoryLimit();
      renderDetail(null);
      renderFacts();
    } else if (typeButton) {
      state.tree.subject = '';
      state.selected = null;
      resetMemoryLimit();
      renderDetail(null);
      renderFacts();
    }
  });
  $('#memory-back')?.addEventListener('click', memoryGoBack);
  $('#episode-agent-grid')?.addEventListener('click', (event) => {
    const agentButton = event.target.closest('[data-episode-agent]');
    if (!agentButton) return;
    state.episodeAgent = agentButton.dataset.episodeAgent || '';
    renderEpisodes();
  });
  $('#episode-breadcrumb')?.addEventListener('click', (event) => {
    if (!event.target.closest('[data-episode-root]')) return;
    state.episodeAgent = '';
    renderEpisodes();
  });
  $('#episode-back')?.addEventListener('click', () => {
    state.episodeAgent = '';
    renderEpisodes();
  });
  $('#diffs-refresh')?.addEventListener('click', () => loadDiffs());
  $('#import-fold')?.addEventListener('toggle', (event) => {
    if (event.target.open && !state.lib.importLoaded) loadImport();
  });
  $('#diffs-search')?.addEventListener('input', () => { window.clearTimeout(window.__diffsTimer); window.__diffsTimer = window.setTimeout(loadDiffs, 240); });
  $('#diffs-skipped-only')?.addEventListener('change', () => loadDiffs());
  $('#library-tree')?.addEventListener('click', (event) => {
    const toggle = event.target.closest('[data-lib-toggle]');
    if (toggle) {
      const t = toggle.dataset.libToggle;
      state.lib.treeOpen[t] = !state.lib.treeOpen[t];
      renderLibraryTree();
      return;
    }
    const btn = event.target.closest('[data-lib-source]');
    if (!btn) return;
    const source = btn.dataset.libSource || '';
    const category = btn.dataset.libCategory || '';
    if (!source) {
      state.lib.tree = { source: '', category: '' };
    } else {
      state.lib.treeOpen[source] = true;
      state.lib.tree = { source, category };
    }
    renderLibraryTree();
    renderLibraryList();
  });
  $('#crumb-home')?.addEventListener('click', () => switchView('chat'));
  $('#crumb-hub')?.addEventListener('click', () => switchView('overview'));
  $('#current-view')?.addEventListener('click', () => switchView(state.currentView || 'chat'));
  $$('[data-goto]').forEach((card) => {
    const open = () => switchView(card.dataset.goto, { status: card.dataset.status || '' });
    card.addEventListener('click', open);
    card.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
    });
  });
  $$('.nav-item[data-view]').forEach((item) => item.addEventListener('click', () => switchView(item.dataset.view)));
}


const DIFF_TYPE_LABELS = { constraint: '约束', fact: '事实', preference: '偏好', goal: '目标', project: '项目', entity: '实体', event: '事件', case: '案例', tool: '工具', belief: '信念' };

function diffOpCard(op, kind) {
  const badge = kind === 'add' ? '新增' : kind === 'update' ? '更新' : '删除';
  const text = op.after || op.deleted_content || op.claim || '';
  const before = (kind === 'update' && op.before && op.before !== op.after)
    ? `<div class="op-before">改前：${escapeHtml(op.before)}</div>` : '';
  const meta = [];
  if (op.memory_type) meta.push(DIFF_TYPE_LABELS[op.memory_type] || op.memory_type);
  if (op.ledger_op) meta.push(({ supersede: '被新版本取代', retract: '撤回' })[op.ledger_op] || op.ledger_op);
  if (op.note) meta.push(op.note === 'corroborated' ? '佐证已有记忆' : op.note);
  if (op.reason) meta.push('原因：' + op.reason);
  return `<div class="op-card op-${kind}"><span class="op-badge">${badge}</span>
    <div class="op-body"><div class="op-text">${escapeHtml(text)}</div>${before}
    ${meta.length ? `<div class="op-meta">${escapeHtml(meta.join(' · '))}</div>` : ''}</div></div>`;
}

async function loadDiffs() {
  const list = $('#diffs-list');
  const detail = $('#diffs-detail');
  if (!list) return;
  const request = (state.diffsRequest || 0) + 1;
  state.diffsRequest = request;
  try {
    const params = new URLSearchParams();
    const q = ($('#diffs-search')?.value || '').trim();
    if (q) params.set('q', q);
    if ($('#diffs-skipped-only')?.checked) params.set('skipped_only', '1');
    const data = await fetchJson(`/api/session-diffs?${params.toString()}`);
    if (request !== state.diffsRequest) return;
    const items = data.items || [];
    $('#diffs-count').textContent = String(data.total ?? items.length);
    if (!items.length) {
      list.innerHTML = '<div class="side-empty">还没有匹配的会话归档记录。</div>';
      state.diffDetailRequest = (state.diffDetailRequest || 0) + 1;
      detail.innerHTML = '<div class="side-empty">从列表选择一条归档，查看它写入和跳过了什么。</div>';
      return;
    }
    list.innerHTML = items.map((item) => {
      const skipped = item.skipped_count || 0;
      const summary = item.summary || {};
      const changed = (summary.total_adds ?? 0) + (summary.total_updates ?? 0) + (summary.total_deletes ?? 0);
      const summaryText = (changed + skipped) === 0
        ? '没有写入、更新或跳过任何记忆'
        : `新增 ${summary.total_adds ?? 0} · 更新 ${summary.total_updates ?? 0} · 删除 ${summary.total_deletes ?? 0} · 跳过 ${skipped}`;
      return `<button type="button" class="diff-row" data-diff-path="${escapeHtml(item.path)}" data-diff-key="${escapeHtml(item.session + '/' + item.archive)}">
        <strong title="${escapeHtml(item.session)}">${escapeHtml(shortToken(item.session))} / ${escapeHtml(item.archive)}</strong>
        <span>${summaryText}</span>
      </button>`;
    }).join('');
    list.querySelectorAll('[data-diff-key]').forEach((btn) => btn.addEventListener('click', async () => {
      list.querySelectorAll('.diff-row').forEach((n) => n.classList.remove('is-active'));
      btn.classList.add('is-active');
      const detailRequest = (state.diffDetailRequest || 0) + 1;
      state.diffDetailRequest = detailRequest;
      try {
        const key = btn.dataset.diffKey;
        const payload = await fetchJson(`/api/session-diffs/${encodeURIComponent(key)}`);
        if (detailRequest !== state.diffDetailRequest) return;
        const diff = payload.diff || payload;
        const skipped = (diff.skipped_operations || (diff.diff && diff.diff.skipped_operations) || []);
        const body = payload.diff || diff;
        const summary = body.summary || {};
        const ops = body.operations || {};
        const adds = ops.adds || [];
        const updates = ops.updates || [];
        const deletes = ops.deletes || [];
        const nAdds = summary.total_adds ?? adds.length;
        const nUpdates = summary.total_updates ?? updates.length;
        const nDeletes = summary.total_deletes ?? deletes.length;
        const errors = summary.errors || [];
        const errText = errors.length ? `抽取报错 ${errors.length} 条` : '抽取无报错';
        const headline = (nAdds + nUpdates + nDeletes + skipped.length) === 0
          ? `这次会话没有写入、更新或跳过任何记忆；${errText}。`
          : `本次归档：新增 ${nAdds} 条 · 更新 ${nUpdates} 条 · 删除 ${nDeletes} 条 · 跳过 ${skipped.length} 条；${errText}。`;
        const changeCards = [...adds.map((op) => diffOpCard(op, 'add')),
          ...updates.map((op) => diffOpCard(op, 'update')),
          ...deletes.map((op) => diffOpCard(op, 'delete'))].join('');
        const skippedCards = skipped.map((op) => `
          <div class="op-card op-skip"><span class="op-badge">跳过</span>
            <div class="op-body"><div class="op-text">${escapeHtml(op.reason || op.reason_code || '（无说明）')}</div>
            ${op.reason_code ? `<div class="op-meta">${escapeHtml(op.reason_code)}</div>` : ''}</div></div>`).join('');
        detail.innerHTML = `
          <div class="panel-header diff-detail-head"><div><div class="panel-title-row"><h3>${escapeHtml(shortToken(key, 14))}</h3><span class="panel-meta">${escapeHtml(body.extracted_at ? formatTime(body.extracted_at) : '时间未知')}</span></div>
          <p class="panel-subtitle">归档路径：${escapeHtml(payload.path || '')}</p></div></div>
          <p class="diff-headline">${escapeHtml(headline)}</p>
          <div class="diff-section-title">记忆变更（${nAdds + nUpdates + nDeletes}）</div>
          ${changeCards || '<div class="side-empty">没有记忆变更。</div>'}
          <div class="diff-section-title">跳过的内容（${skipped.length}）</div>
          ${skippedCards || '<div class="side-empty">没有跳过的内容。</div>'}
          <details class="diff-raw"><summary>查看原始数据（排查备用）</summary>
            <pre>${escapeHtml(JSON.stringify(body, null, 2))}</pre></details>`;
      } catch (error) {
        if (detailRequest === state.diffDetailRequest) detail.innerHTML = `<div class="side-empty">${escapeHtml(error.message)}</div>`;
      }
    }));
  } catch (error) {
    if (request === state.diffsRequest) list.innerHTML = `<div class="side-empty">${escapeHtml(error.message)}</div>`;
  }
}


bind();
renderChat();
renderMemorySources();
restoreChat();
loadData();
loadAgents();
loadLibrary();
const initialView = location.hash.replace(/^#\/?/, '');
if (initialView && initialView !== 'chat' && VIEW_LABELS[initialView]) switchView(initialView);
