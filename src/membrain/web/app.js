const state = { facts: [], factTotal: null, factsTruncated: false, selected: null, episodes: [], episodeTotal: 0, episodeAgent: '', filters: { q: '', status: '', type: '' }, tree: { type: '', subject: '' }, currentView: 'chat', dialog: null, sourceLabels: {},
  memoryLimit: 8, chat: { sessionId: null, messages: [], memory: [], sending: false }, model: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const labels = {
  type: { preference: '偏好', fact: '事实', goal: '目标', project: '项目', constraint: '约束', entity: '实体', event: '事件', case: '案例', tool: '工具', belief: '信念', task: '任务' },
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

function memoryStatusLabel() {
  return { active: '当前有效', contested: '待处理冲突', superseded: '已替代', retracted: '已撤回' }[state.filters.status] || '状态筛选';
}

function treeOpenSet() {
  if (!(state.treeOpen instanceof Set)) state.treeOpen = new Set();
  return state.treeOpen;
}

function renderFacts() {
  const tree = $('#obsidian-tree');
  if (!tree) return;
  const groups = memoryGroups();
  $('#memory-count').textContent = state.factTotal ?? state.facts.length;
  if (!state.facts.length) {
    tree.innerHTML = '<div class="side-empty">还没有记忆。接入 Agent 并开始对话后，记忆会出现在这里。</div>';
    return;
  }
  const open = treeOpenSet();
  const expandAll = Boolean(state.filters.q) || Boolean(state.filters.status);
  const types = [...groups.keys()].sort((a, b) => (labels.type[a] || a).localeCompare(labels.type[b] || b, 'zh'));
  tree.innerHTML = types.map((type) => {
    const subs = groups.get(type);
    const total = [...subs.values()].reduce((sum, n) => sum + n, 0);
    const typeKey = `t:${type}`;
    const typeOpen = expandAll || open.has(typeKey);
    const subjectHtml = typeOpen ? [...subs.keys()].sort((a, b) => a.localeCompare(b, 'zh')).map((subject) => {
      const subjectKey = `s:${type}:${subject}`;
      const subjectOpen = expandAll || open.has(subjectKey);
      const facts = state.facts.filter((fact) => fact.type === type && fact.subject === subject);
      const leaves = subjectOpen ? facts.map((fact) => `
        <button class="obsidian-leaf${state.selected?.id === fact.id ? ' is-selected' : ''}" type="button" data-fact-id="${escapeHtml(fact.id)}">
          <span class="obsidian-leaf-title">${escapeHtml(fact.abstract || fact.claim)}</span>
          <span class="obsidian-leaf-dot obsidian-dot-${escapeHtml(fact.status)}" title="${escapeHtml(fact.status)}"></span>
        </button>`).join('') : '';
      return `<div class="obsidian-branch">
        <button class="obsidian-folder obsidian-folder-sub" type="button" data-toggle="${escapeHtml(subjectKey)}">
          <span class="obsidian-caret" aria-hidden="true">${subjectOpen ? '▾' : '▸'}</span>
          <span class="obsidian-folder-name">${escapeHtml(subject)}</span>
          <span class="obsidian-count">${facts.length}</span>
        </button>${leaves}</div>`;
    }).join('') : '';
    return `<div class="obsidian-branch">
      <button class="obsidian-folder obsidian-folder-type" type="button" data-toggle="${escapeHtml(typeKey)}">
        <span class="obsidian-caret" aria-hidden="true">${typeOpen ? '▾' : '▸'}</span>
        <span class="obsidian-folder-name">${escapeHtml(labels.type[type] || type)}</span>
        <span class="obsidian-count">${total}</span>
      </button>${subjectHtml}</div>`;
  }).join('');
  tree.querySelectorAll('[data-toggle]').forEach((button) => button.addEventListener('click', () => {
    const key = button.dataset.toggle;
    const bag = treeOpenSet();
    if (bag.has(key)) bag.delete(key); else bag.add(key);
    renderFacts();
  }));
  tree.querySelectorAll('[data-fact-id]').forEach((row) => row.addEventListener('click', () => selectFact(row.dataset.factId)));
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
  // Tree navigation replaces folder drilling; kept as a no-op for legacy callers.
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
  const main = $('#obsidian-main');
  const meta = $('#obsidian-meta');
  if (!main || !meta) return;
  if (!fact) {
    main.innerHTML = '<div class="detail-empty"><div class="detail-empty-mark">文</div><h2>选择一条记忆</h2><p>左边点一条记忆，这里显示内容原文和依据的原话。</p></div>';
    meta.innerHTML = '';
    return;
  }
  const confidence = Math.round((fact.confidence || 0) * 100);
  const evidence = (fact.evidence || []).map((item) => `<div class="evidence-item"><p class="evidence-quote">“${escapeHtml(item.quote)}”</p><span class="evidence-meta">${escapeHtml(item.episode)} · ${formatDate(item.date)} · ${escapeHtml(item.origin)}</span></div>`).join('');
  const canAct = fact.status !== 'retracted' && fact.status !== 'superseded';
  main.innerHTML = `
    <div class="detail-top"><div><div class="memory-line"><span class="memory-folder-level">内容原文</span>${typeTag(fact.type)}${statusBadge(fact.status)}</div><h2 class="detail-title">${escapeHtml(fact.claim)}</h2></div></div>
    <div class="detail-section"><div class="detail-section-label">依据的原话 · ${fact.evidence?.length ?? 0}</div>${evidence || '<p class="panel-subtitle">没有证据。</p>'}</div>`;
  meta.innerHTML = `
    <div class="detail-section"><div class="detail-section-label">基础信息</div>
      <div class="detail-meta">
        <div><span class="meta-label">类型</span><span class="meta-value">${escapeHtml(labels.type[fact.type] || fact.type)}</span></div>
        <div><span class="meta-label">主题</span><span class="meta-value">${escapeHtml(fact.subject || '—')}</span></div>
        <div><span class="meta-label">状态</span><span class="meta-value">${statusBadge(fact.status)}</span></div>
        <div><span class="meta-label">来源 Agent</span><span class="meta-value">${escapeHtml(displaySource(fact.source?.agent || 'unknown'))}</span></div>
        <div><span class="meta-label">会话</span><span class="meta-value">${escapeHtml(fact.source?.session || '—')}</span></div>
        <div><span class="meta-label">首次记录</span><span class="meta-value">${formatDate(fact.created_at)}</span></div>
        <div><span class="meta-label">有效起点</span><span class="meta-value">${formatDate(fact.valid_from)}</span></div>
        <div><span class="meta-label">记忆 ID</span><span class="meta-value">${escapeHtml(fact.id)}</span></div>
      </div>
    </div>
    <div class="detail-section"><div class="detail-section-label">把握程度</div><div class="confidence-row"><div class="confidence-track"><div class="confidence-fill ${widthBucket(confidence)}"></div></div><span class="confidence-value">${confidence}%</span></div></div>
    <div class="detail-actions">${canAct ? `<button class="button button-outline" data-action="correct">${icon('edit')}纠正</button><button class="button button-outline" data-action="contest">${icon('flag')}标记冲突</button><button class="button button-danger" data-action="retract">${icon('undo')}撤回</button>${fact.status === 'contested' ? `<button class="button button-outline" data-action="resolve">解决冲突</button>` : ''}` : '<span class="panel-meta">此记忆已结束当前生命周期</span>'}</div>`;
  meta.querySelectorAll('[data-action]').forEach((button) => button.addEventListener('click', () => openAction(button.dataset.action, fact)));
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
  if (model) { renderModel(model); renderRecallBaseline(model); }
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

function toggleChatHistory() {
  const panel = $('#chat-history-panel');
  if (!panel) return;
  const opening = panel.classList.contains('is-hidden');
  panel.classList.toggle('is-hidden', !opening);
  if (opening) loadChatHistory();
}

async function loadChatHistory() {
  const list = $('#chat-history-list');
  if (!list) return;
  list.innerHTML = '<div class="side-empty">正在加载…</div>';
  try {
    const data = await fetchJson('/api/chat-sessions');
    const sessions = data.sessions || [];
    $('#chat-history-count').textContent = `${sessions.length} 个会话`;
    if (!sessions.length) {
      list.innerHTML = '<div class="side-empty">还没有历史会话。</div>';
      return;
    }
    list.innerHTML = sessions.map((row) => `
      <button class="chat-history-item${row.session_id === state.chat.sessionId ? ' is-current' : ''}" type="button" data-session-id="${escapeHtml(row.session_id)}">
        <span class="chat-history-title">${escapeHtml(row.title || '（无标题会话）')}</span>
        <span class="chat-history-meta">${escapeHtml(formatTime(row.last_ts))} · ${row.turns} 条</span>
      </button>`).join('');
    list.querySelectorAll('[data-session-id]').forEach((el) => {
      el.addEventListener('click', () => openChatSession(el.dataset.sessionId));
    });
  } catch (error) {
    list.innerHTML = `<div class="side-empty">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

async function openChatSession(sessionId) {
  try {
    const saved = await fetchJson(`/api/chat/${encodeURIComponent(sessionId)}`);
    state.chat.sessionId = saved.session_id;
    state.chat.messages = saved.messages || [];
    state.chat.memory = [];
    localStorage.setItem('membrain.activeChat', sessionId);
    renderChat();
    renderMemorySources();
    $('#chat-history-panel')?.classList.add('is-hidden');
  } catch (error) {
    notify(error.message, true);
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

async function reloadConfig() {
  const button = $('#config-reload-button');
  button.disabled = true;
  try {
    const result = await fetchJson('/api/config/reload', { method: 'POST', body: '{}' });
    const names = { llm_base_url: '接口地址', llm_model: '模型', llm_api_key: '模型 Key',
      llm_api_format: '接口格式', llm_timeout: '模型超时', embed_base_url: 'Embedding 地址',
      embed_api_key: 'Embedding Key', embed_model: 'Embedding 模型', embed_timeout: 'Embedding 超时',
      semantic_floor: '语义阈值', use_intent_default: '意图识别', use_rerank_default: '重排' };
    const items = Object.entries(result.changed || {})
      .map(([field, change]) => `${names[field] || field}: ${change.from || '（空）'} → ${change.to || '（空）'}`);
    notify(items.length ? `已热加载 ${items.length} 项变更：${items.join('；')}` : '配置无变化。', !items.length);
    await loadModel();
  } catch (error) { notify(error.message, true); }
  button.disabled = false;
}

function renderRecallBaseline(model) {
  const info = $('#recall-baseline-info');
  const note = $('#recall-baseline-note');
  if (!info || !note) return;
  const baseline = model.recall_baseline;
  if (!baseline) { info.textContent = '未建立基线'; return; }
  const rate = Math.round((baseline.pass_rate || 0) * 100);
  info.textContent = `${baseline.embed_model} · 通过率 ${rate}% · ${String(baseline.measured_at || '').slice(0, 10)}`;
  if (model.recall_baseline_stale) {
    note.textContent = '⚠️ Embedding 模型与基线不一致，建议重新验证并更新基线。';
    note.classList.add('is-warn');
  } else {
    note.textContent = '与当前配置一致。换模型或调阈值后请重新验证。';
    note.classList.remove('is-warn');
  }
}

let recallVerifyTimer;
async function startRecallVerify() {
  const button = $('#recall-verify-button');
  button.disabled = true;
  $('#recall-verify-result').textContent = '正在后台验证（82 组黄金样例）…';
  try {
    await fetchJson('/api/recall/verify', { method: 'POST', body: '{}' });
    clearInterval(recallVerifyTimer);
    let fails = 0;
    recallVerifyTimer = setInterval(async () => {
      let status;
      try {
        status = await fetchJson('/api/recall/verify/status');
        fails = 0;
      } catch (error) {
        fails += 1;
        if (fails >= 5) {
          clearInterval(recallVerifyTimer);
          button.disabled = false;
          $('#recall-verify-result').textContent = `查不到验证状态（${error.message}），请重新点击。`;
        }
        return;
      }
      if (status.status === 'running') {
        const p = status.progress || {};
        const line = p.line ? ` ${p.line}` : '';
        const elapsed = p.elapsed ? `（已用 ${p.elapsed} 秒）` : '…';
        $('#recall-verify-result').textContent = `正在后台验证…${line}${elapsed}`;
        return;
      }
      clearInterval(recallVerifyTimer);
      button.disabled = false;
      if (status.status === 'idle') {
        $('#recall-verify-result').textContent = '验证任务已丢失（服务可能重启过），请重新点击。';
        return;
      }
      const report = status.report || {};
      if (status.error) { $('#recall-verify-result').textContent = `验证出错：${status.error}`; return; }
      const lastLines = String(report.output || '').trim().split('\n').slice(-3).join(' | ');
      $('#recall-verify-result').textContent = report.ok
        ? `验证通过，基线已更新。${lastLines}`
        : `发现退化：${lastLines}`;
      if (report.ok) notify('召回质量验证通过，基线已更新。');
      else notify('召回质量有退化，详见模型连接页结果。', true);
      await loadModel();
    }, 3000);
  } catch (error) { notify(error.message, true); button.disabled = false; }
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
      $('#obsidian-main')?.scrollIntoView({ block: 'start' });
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
  let showAll = false;
  try { showAll = localStorage.getItem('membrain.agentShowAll') === '1'; } catch { /* storage may be disabled */ }
  const visible = showAll ? items : items.filter((agent) => agent.installed || agent.key.startsWith('other:'));
  const hidden = items.length - visible.length;
  const headNote = hidden > 0
    ? `<div class="agent-grid-note">已隐藏 ${hidden} 个未安装的 Agent。</div>` : '';
  grid.innerHTML = headNote + visible.map((agent) => {
    const [label, tone] = agentStatus[agent.status] || [agent.status, ''];
    const capture = agent.capture === 'hooks' ? '对话内容自动记入记忆，并把相关记忆带给模型' : '对话自动记入记忆；需要时自动想起相关内容';
    const connect = agent.can_connect && agent.installed && agent.status !== 'connected' && agent.status !== 'unreadable'
      ? `<button class="button button-dark" type="button" data-agent-connect="${escapeHtml(agent.key)}">一键接入</button>` : '';
    const connectedLike = agent.status === 'connected' || agent.isolated;
    const disconnect = connectedLike
      ? `<button class="button button-outline" type="button" data-agent-disconnect="${escapeHtml(agent.key)}">断开接入</button>` : '';
    const note = !agent.installed
      ? '这台机器上还没装它。装好后回来点“一键接入”就行。'
      : agent.isolated
        ? '已断开：它的记忆被暂时隔离（没有删除），重新接入后恢复可见。'
        : `${capture}${agent.protected ? '；它的设置不让程序自动改，点“复制提示词”照着做' : ''}`;
    const chip = agent.isolated ? '<span class="agent-chip is-warn">已隔离</span>' : `<span class="agent-chip ${tone}">${label}</span>`;
    const running = agent.running ? '<span class="agent-chip is-ok">运行中</span>' : '';
    return `<article class="agent-card${agent.installed ? '' : ' is-off'}${agent.isolated ? ' is-isolated' : ''}">
      <div class="agent-card-head"><strong>${escapeHtml(agent.name)}</strong>${running}${chip}</div>
      <p class="agent-note">${note}</p>
      <div class="agent-actions">${connect}${disconnect}<button class="button button-outline" type="button" data-agent-copy="${escapeHtml(agent.key)}">复制提示词</button></div>
    </article>`;
  }).join('');
  $$('[data-agent-connect]').forEach((button) => button.addEventListener('click', () => connectAgent(button.dataset.agentConnect, button)));
  $$('[data-agent-disconnect]').forEach((button) => button.addEventListener('click', () => disconnectAgent(button.dataset.agentDisconnect, button)));
  $$('[data-agent-copy]').forEach((button) => button.addEventListener('click', () => copyPrompt(button.dataset.agentCopy)));
}

async function loadAgents() {
  try { renderAgents((await fetchJson('/api/agents')).agents || []); }
  catch (error) {
    $('#agent-grid').innerHTML = `<div class="empty-state"><div class="empty-mark">A</div><h3>扫描失败</h3><p>${escapeHtml(error.message)}</p></div>`;
    notify(error.message, true);
  }
  fetchJson('/api/watcher').then((w) => {
    const note = $('#agents-watcher-note');
    if (note && w) {
      const last = w.last_run ? ` · 最近捕获 ${w.last_run.captured} 条` : '';
      note.textContent = `自动捕获：${w.enabled ? '后台运行中，接入的 Agent 对话会被自动记录提炼' : '已关闭（MEMBRAIN_WATCH=0）'}${last}`;
    }
  }).catch(() => {});
}

// Auto-rescan so install/uninstall shows up without clicking 重新扫描.
setInterval(() => {
  const panel = document.querySelector('[data-view-panel="agents"]');
  if (panel && panel.classList.contains('is-active')) loadAgents();
}, 45000);
window.addEventListener('focus', () => {
  const panel = document.querySelector('[data-view-panel="agents"]');
  if (panel && panel.classList.contains('is-active')) loadAgents();
});

const HEALTH_KINDS = {
  duplicate: ['疑似重复', 'is-warn'], conflict: ['待裁决冲突', 'is-warn'],
  noisy: ['疑似噪声', ''], forget: ['遗忘候选', 'is-off'],
};

async function loadHealth() {
  const grid = $('#health-findings');
  if (!grid) return;
  try {
    const report = await fetchJson('/api/health');
    $('#health-score').textContent = `${report.score} 分`;
    const counts = Object.entries(report.counts || {});
    $('#health-summary').textContent = counts.length
      ? counts.map(([kind, count]) => `${HEALTH_KINDS[kind]?.[0] || kind} ${count} 项`).join(' · ')
      : `一切正常，没有发现需要处理的问题。${report.dismissed ? `（已忽略 ${report.dismissed} 项）` : ''}`;
    if (!report.findings.length) {
      grid.innerHTML = '<div class="empty-state"><div class="empty-mark">H</div><h3>很健康</h3><p>没有重复、冲突或需要处理的问题。</p></div>';
      return;
    }
    grid.innerHTML = report.findings.map((finding) => {
      const [label, tone] = HEALTH_KINDS[finding.kind] || [finding.kind, ''];
      const [a, b] = finding.fact_ids.length === 2 ? finding.fact_ids : [finding.fact_ids[0], ''];
      const supersede = finding.kind === 'duplicate'
        ? `<button class="button button-dark" type="button" data-health-supersede="${finding.fingerprint}" data-keep="${escapeHtml(a)}" data-drop="${escapeHtml(b)}">合并并替代</button>` : '';
      const retract = (finding.kind === 'noisy' || finding.kind === 'forget')
        ? `<button class="button button-outline" type="button" data-health-retract="${finding.fingerprint}" data-drop="${escapeHtml(finding.fact_ids[0])}">撤回</button>` : '';
      return `<article class="health-finding">
        <div class="health-finding-head"><span class="agent-chip ${tone}">${label}</span><span class="health-fp">${finding.fact_ids.length} 条</span></div>
        <p class="health-summary-line">${escapeHtml(finding.summary)}</p>
        <p class="health-suggestion">${escapeHtml(finding.suggestion)}</p>
        <div class="agent-actions">${supersede}${retract}
          <button class="button button-outline" type="button" data-health-dismiss="${finding.fingerprint}">忽略</button>
        </div>
      </article>`;
    }).join('');
    $$('[data-health-supersede]').forEach((button) => button.addEventListener('click', () => healthAction('supersede', button)));
    $$('[data-health-retract]').forEach((button) => button.addEventListener('click', () => healthAction('retract', button)));
    $$('[data-health-dismiss]').forEach((button) => button.addEventListener('click', () => healthAction('dismiss', button)));
  } catch (error) {
    grid.innerHTML = `<div class="empty-state"><div class="empty-mark">H</div><h3>扫描失败</h3><p>${escapeHtml(error.message)}</p></div>`;
    notify(error.message, true);
  }
}

async function healthAction(action, button) {
  const payload = { action, fingerprint: button.dataset.healthSupersede || button.dataset.healthRetract || button.dataset.healthDismiss,
    keep_id: button.dataset.keep || '', drop_id: button.dataset.drop || '' };
  if (action === 'supersede' && (!payload.keep_id || !payload.drop_id)) { notify('该发现缺少成对的记忆。', true); return; }
  if (action !== 'dismiss') {
    const verb = action === 'supersede' ? '合并并替代' : '撤回';
    if (!window.confirm(`确定要${verb}这条记忆？原始文件会保留在生命周期记录中，可以追溯。`)) return;
  }
  button.disabled = true;
  try {
    const result = await fetchJson('/api/health/action', { method: 'POST', body: JSON.stringify(payload) });
    notify(result.note || (action === 'dismiss' ? '已忽略，下次扫描不再提示。' : '已处理。'));
    await loadHealth();
  } catch (error) { notify(error.message, true); button.disabled = false; }
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

async function disconnectAgent(key, button) {
  const agent = (state.agents || []).find((item) => item.key === key);
  const name = agent ? agent.name : key;
  const confirmed = window.confirm(
    `断开 ${name} 的接入？\n\n` +
    '· 该 Agent 产生的记忆会被隔离：不再出现在检索、召回和公共记忆里（数据不删除）。\n' +
    '· 该 Agent 的新对话不再被捕获。\n' +
    '· 重新接入后记忆恢复可见。\n\n确定要断开吗？');
  if (!confirmed) return;
  button.disabled = true;
  try {
    const result = await fetchJson(`/api/agents/${encodeURIComponent(key)}/disconnect`, { method: 'POST', body: '{}' });
    notify(result.manual ? result.note || '配置需要手动移除；记忆隔离已生效。'
      : result.config_removed ? `已断开，原配置已备份为 ${result.backup}。重启该软件后生效` : '已断开接入，记忆隔离已生效。');
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
  const sessions = (prev.sessions || []).filter((s) => s.status !== 'noise');
  const bySource = new Map();
  sessions.forEach((s) => {
    if (!bySource.has(s.source)) bySource.set(s.source, []);
    bySource.get(s.source).push(s);
  });
  const groupsHtml = [...bySource.entries()].map(([source, rows]) => {
    const selectable = rows.filter((s) => s.status !== 'already');
    const rowsHtml = rows.slice(0, 80).map((s) => {
      const statusChip = s.status === 'already'
        ? '<span class="agent-chip is-ok">已导入</span>'
        : s.status === 'skipped'
          ? '<span class="agent-chip is-off">已忽略</span>' : '';
      const checkbox = s.status === 'new' || s.status === 'skipped'
        ? `<input type="checkbox" class="session-check" data-session-key="${escapeHtml(s.key)}" ${s.status === 'new' ? 'checked' : ''}>`
        : '';
      return `<label class="session-row"><span class="session-row-main">${checkbox}<span class="session-title">${escapeHtml(s.title || '（无标题）')}</span></span>
        <span class="session-row-meta">${escapeHtml(s.date)} · ${s.turns} 轮 ${statusChip}</span></label>`;
    }).join('');
    return `<div class="session-group">
      <div class="session-group-head"><label class="check-inline"><input type="checkbox" class="session-group-check" data-session-group="${escapeHtml(source)}" ${selectable.some((s) => s.status === 'new') ? 'checked' : ''}> ${escapeHtml(rows[0].label || source)}</label>
      <span class="import-note">${rows.length} 个会话${rows.length > 80 ? '（只显示前 80）' : ''}</span></div>
      <div class="session-group-body">${rowsHtml}</div></div>`;
  }).join('');
  const actionable = (prev.new || 0) > 0 || sessions.some((s) => s.status === 'skipped');
  card.innerHTML = `<h3>Agent 会话导入 · LLM 清洗筛选</h3>
    <p class="import-note">自动发现本机 Agent 历史会话（源只读）。勾选要导入的，由 AI 识别分类、起标题、写一句话摘要后进资料库；<strong>不勾选即不导入</strong>；「忽略选中」后永久不再提示（可恢复）。纯问候/空会话/系统回执等噪声自动丢弃。</p>
    <p class="import-note" id="watcher-note">新对话由后台自动捕获并提炼，无需重复导入。</p>
    <div class="import-actions">
      <button class="button button-dark" id="session-import-start" type="button" ${actionable ? '' : 'disabled'}>导入选中</button>
      <button class="button button-outline" id="session-import-skip" type="button">忽略选中</button>
      <button class="button button-outline" id="session-import-unskip" type="button">恢复全部忽略</button>
      <label class="check-inline"><input type="checkbox" id="session-import-distill"> 同时提炼为短记忆 <span class="import-note">（默认关闭：历史会话大多为闲聊和临时上下文，勾选后 LLM 才会把值得长期保存的信息写成记忆条目；不勾则只进资料库）</span></label>
      <span class="import-note" id="session-import-status"></span>
    </div>
    ${groupsHtml || '<div class="side-empty">没有发现可导入的历史会话。</div>'}`;
  $('#session-import-start')?.addEventListener('click', () => startSessionImport('import'));
  $('#session-import-skip')?.addEventListener('click', () => startSessionImport('skip'));
  $('#session-import-unskip')?.addEventListener('click', () => startSessionImport('unskip'));
  card.querySelectorAll('[data-session-group]').forEach((box) => box.addEventListener('change', () => {
    const group = box.closest('.session-group');
    if (!group) return;
    group.querySelectorAll('.session-check').forEach((one) => { one.checked = box.checked; });
  }));
  fetchJson('/api/watcher').then((w) => {
    const note = $('#watcher-note');
    if (note && w) {
      const last = w.last_run ? `，最近一次捕获 ${w.last_run.captured} 条` : '';
      note.textContent = `新对话由后台自动捕获并提炼（捕获服务${w.enabled ? '运行中' : '已关闭'}${last}），无需重复导入。`;
    }
  }).catch(() => {});
}

async function startSessionImport(mode) {
  const button = $('#session-import-start');
  if (button) button.disabled = true;
  try {
    const checked = [...document.querySelectorAll('.session-check:checked')].map((el) => el.dataset.sessionKey);
    if (mode === 'unskip') {
      await fetchJson('/api/import/sessions/start', { method: 'POST', body: JSON.stringify({ skips_only: true, clear_skips: true }) });
      await loadImport();
      notify('已恢复全部忽略的会话');
      return;
    }
    if (mode === 'import' && !checked.length) { notify('请先勾选要导入的会话。', true); return; }
    if (mode === 'skip' && !checked.length) { notify('请先勾选要忽略的会话。', true); return; }
    if (mode === 'skip') {
      await fetchJson('/api/import/sessions/start', { method: 'POST', body: JSON.stringify({ skips_only: true, skip: checked }) });
      notify(`已忽略 ${checked.length} 个会话`);
      await loadImport();
      return;
    }
    const body = { distill_facts: Boolean($('#session-import-distill')?.checked), only: checked };
    await fetchJson('/api/import/sessions/start', { method: 'POST', body: JSON.stringify(body) });
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
      : '<div class="empty-state library-empty"><div class="empty-mark">档</div><h3>资料库还是空的</h3><p>展开上方的“数据导入”，可以导入本机 Agent 历史会话，或配置可选的历史来源。</p></div>';
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
      <div class="detail-section"><div class="detail-section-label">完整内容 · ${entry.total_chars} 字</div>
        <pre class="library-body">${escapeHtml(entry.text)}</pre>
        ${pager}
        ${hasMore ? '<p class="panel-subtitle">长文已分段；也可让 Agent 用 read_archive 分段读取。</p>' : ''}</div>
      <div class="detail-section"><div class="detail-meta"><div><span class="meta-label">来源</span><span class="meta-value">${escapeHtml(entry.origin || '—')}</span></div><div><span class="meta-label">范围</span><span class="meta-value">${escapeHtml(entry.scope || '—')}</span></div><div><span class="meta-label">日期</span><span class="meta-value">${escapeHtml(entry.date || '—')}</span></div></div></div>`;
    content.querySelectorAll('[data-lib-page]').forEach((btn) => btn.addEventListener('click', () => selectLibraryEntry(id, Number(btn.dataset.libPage))));
  } catch (error) { notify(error.message, true); }
}

/* ---- RAG knowledge library (uploaded documents) ---- */
state.ragDocs = [];
state.ragSelected = null;

async function loadRag() {
  try {
    const data = await fetchJson('/api/rag/documents');
    state.ragDocs = data.documents || [];
    renderRagList();
  } catch (error) {
    const list = $('#rag-list');
    if (list) list.innerHTML = `<div class="side-empty">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderRagList(items) {
  const list = $('#rag-list');
  if (!list) return;
  const rows = items || state.ragDocs || [];
  $('#rag-count').textContent = rows.length;
  if (!rows.length) {
    list.innerHTML = '<div class="side-empty">还没有文档。把文件拖进上方区域试试。</div>';
    return;
  }
  list.innerHTML = rows.map((doc) => `
    <button class="rag-item${state.ragSelected === doc.id ? ' is-selected' : ''}" type="button" data-rag-id="${escapeHtml(doc.id)}">
      <span class="rag-item-title">${escapeHtml(doc.title)}</span>
      <span class="rag-item-meta">${escapeHtml(doc.category)} · ${doc.chunks} 段</span>
    </button>`).join('');
  list.querySelectorAll('[data-rag-id]').forEach((el) => el.addEventListener('click', () => openRagDoc(el.dataset.ragId)));
}

async function openRagDoc(id) {
  try {
    const doc = await fetchJson(`/api/rag/doc/${encodeURIComponent(id)}`);
    state.ragSelected = id;
    renderRagList();
    const detail = $('#rag-detail');
    if (!detail) return;
    detail.classList.remove('is-hidden');
    detail.innerHTML = `
      <div class="detail-top"><div><h2 class="detail-title">${escapeHtml(doc.title)}</h2>
      <p class="detail-id">${escapeHtml(doc.uri || doc.id)} · ${escapeHtml(doc.category)}</p></div>
      <button class="button button-danger" id="rag-delete-button">删除</button></div>
      <div class="detail-section"><div class="detail-section-label">内容分段 · ${doc.chunks}（检索时按段落定位）</div>
        ${(doc.chunk_list || []).map((c) => `<div class="rag-chunk"><span class="rag-chunk-seq">#${c.seq + 1}</span><p>${escapeHtml(c.text)}</p></div>`).join('')}
      </div>`;
    $('#rag-delete-button')?.addEventListener('click', async () => {
      if (!window.confirm('删除这篇资料？')) return;
      await fetchJson('/api/rag/delete', { method: 'POST', body: JSON.stringify({ id }) });
      state.ragSelected = null;
      $('#rag-detail').classList.add('is-hidden');
      notify('资料已删除');
      loadRag();
    });
  } catch (error) { notify(error.message, true); }
}

async function uploadRagFiles(fileList) {
  const files = [...(fileList || [])];
  if (!files.length) return;
  if (state.ragUploading) { notify('正在上传中，请稍候。'); return; }
  state.ragUploading = true;
  const button = $('#rag-upload-button');
  if (button) { button.disabled = true; button.textContent = '导入中…'; }
  notify(`正在导入 ${files.length} 个文件（读取内容、分段、建索引）…`);
  let ok = 0;
  for (const file of files) {
    try {
      const buf = new Uint8Array(await file.arrayBuffer());
      let binary = '';
      for (let i = 0; i < buf.length; i += 0x8000) {
        binary += String.fromCharCode(...buf.subarray(i, i + 0x8000));
      }
      await fetchJson('/api/rag/upload', { method: 'POST', body: JSON.stringify({ filename: file.name, data: btoa(binary) }) });
      ok += 1;
    } catch (error) { notify(`${file.name}：${error.message}`, true); }
  }
  state.ragUploading = false;
  if (button) { button.disabled = false; button.textContent = '选择文件'; }
  notify(`已导入 ${ok} / ${files.length} 个文件`);
  loadRag();
}

async function ragSearch(query) {
  const q = (query || '').trim();
  if (!q) { loadRag(); return; }
  try {
    const data = await fetchJson(`/api/rag/search?q=${encodeURIComponent(q)}`);
    const hits = data.hits || [];
    const list = $('#rag-list');
    $('#rag-count').textContent = hits.length;
    list.innerHTML = hits.length ? hits.map((hit) => `
      <button class="rag-item" type="button" data-rag-id="${escapeHtml(hit.resource_id)}">
        <span class="rag-item-title">${escapeHtml(hit.title)} · 段 ${hit.seq + 1} · ${hit.score}</span>
        <span class="rag-item-meta">${escapeHtml(hit.text.slice(0, 120))}…</span>
      </button>`).join('') : '<div class="side-empty">没有命中。</div>';
    list.querySelectorAll('[data-rag-id]').forEach((el) => el.addEventListener('click', () => openRagDoc(el.dataset.ragId)));
  } catch (error) { notify(error.message, true); }
}

function switchLibraryTab(tab) {
  if (state.lib) state.lib.tab = tab;
  document.querySelectorAll('[data-lib-tab]').forEach((el) => el.classList.toggle('is-active', el.dataset.libTab === tab));
  $('#rag-pane')?.classList.toggle('is-hidden', tab !== 'rag');
  $('#library')?.classList.toggle('is-hidden', tab !== 'archive');
}

function initRagUi() {
  if (state.ragUiInit) return;
  state.ragUiInit = true;
  document.querySelectorAll('[data-lib-tab]').forEach((el) =>
    el.addEventListener('click', () => switchLibraryTab(el.dataset.libTab)));
  $('#rag-upload-button')?.addEventListener('click', () => $('#rag-file-input')?.click());
  $('#rag-file-input')?.addEventListener('change', (event) => { uploadRagFiles(event.target.files); event.target.value = ''; });
  const drop = $('#rag-drop');
  if (drop) {
    drop.addEventListener('dragover', (event) => { event.preventDefault(); drop.classList.add('is-drag'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('is-drag'));
    drop.addEventListener('drop', (event) => {
      event.preventDefault();
      drop.classList.remove('is-drag');
      uploadRagFiles(event.dataTransfer?.files);
    });
    drop.addEventListener('click', () => $('#rag-file-input')?.click());
  }
  let ragTimer;
  $('#rag-search')?.addEventListener('input', (event) => {
    window.clearTimeout(ragTimer);
    ragTimer = window.setTimeout(() => ragSearch(event.target.value.trim()), 260);
  });
  switchLibraryTab(state.lib?.tab === 'archive' ? 'archive' : 'rag');
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

const VIEW_LABELS = { chat: 'Agent 对话', overview: '记忆总览', memories: '全部记忆', conflicts: '待处理冲突', episodes: '捕获记录', diffs: '会话变更', library: '资料库', agents: 'Agent 接入', health: '记忆健康', model: '模型连接' };

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
  if (panelName === 'library') { initRagUi(); loadRag(); loadLibrary(); }
  if (panelName === 'agents') loadAgents();
  if (panelName === 'health') loadHealth();
}

function refreshCurrentView() {
  loadData();
  if (state.currentView === 'diffs') loadDiffs();
  if (state.currentView === 'library') { loadRag(); loadLibrary(); }
  if (state.currentView === 'agents') loadAgents();
  if (state.currentView === 'health') loadHealth();
}

let searchTimer;
function bind() {
  $('#agents-copy-all')?.addEventListener('click', copyAllPrompts);
  $('#agents-scan-button').addEventListener('click', loadAgents);
  $('#agents-show-all')?.addEventListener('change', (event) => {
    try { localStorage.setItem('membrain.agentShowAll', event.target.checked ? '1' : '0'); } catch { /* storage may be disabled */ }
    renderAgents(state.agents || []);
  });
  let libraryTimer;
  $('#library-search').addEventListener('input', () => { window.clearTimeout(libraryTimer); libraryTimer = window.setTimeout(loadLibrary, 300); });
  $('#refresh-button').addEventListener('click', refreshCurrentView);
  $('#add-button').addEventListener('click', () => openAction('add'));
  $('#extract-button').addEventListener('click', startExtract);
  $('#reindex-button').addEventListener('click', reindex);
  $('#new-chat-button').addEventListener('click', newChat);
  $('#chat-history-button')?.addEventListener('click', toggleChatHistory);
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
  $('#config-reload-button')?.addEventListener('click', reloadConfig);
  $('#recall-verify-button')?.addEventListener('click', startRecallVerify);
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
  $('#open-diffs-button')?.addEventListener('click', () => switchView('diffs'));
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
