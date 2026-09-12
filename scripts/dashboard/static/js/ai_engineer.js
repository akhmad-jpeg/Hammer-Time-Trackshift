/**
 * AI Race Engineer — Pit Wall Strategy Transceiver (Frontend Controller)
 * 
 * Synthesizes live telemetry, tire delta, battery State-of-Charge (SOC),
 * fuel burn, track pace sensitivity, and multi-policy evaluations into
 * authoritative F1 team radio transmissions and definitive tactical calls.
 * 
 * Securely connects to user's LLM API key (OpenAI, Anthropic Claude, Google Gemini,
 * OpenRouter, or local endpoint) stored directly in browser localStorage.
 * Falls back deterministically if API key is on hold.
 */

// Global state cache
window.lastAiEngineerData = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    initAiEngineer();
});

function initAiEngineer() {
    const key = localStorage.getItem('ht_llm_api_key') || '';
    const provider = localStorage.getItem('ht_llm_provider') || 'auto';
    const model = localStorage.getItem('ht_llm_model') || '';
    const baseUrl = localStorage.getItem('ht_llm_base_url') || '';
    const notes = localStorage.getItem('ht_llm_notes') || '';
    const autoConsult = localStorage.getItem('ht_llm_auto_consult') === 'true';

    // Populate drawer inputs if present
    const keyInput = document.getElementById('ai-cfg-key');
    if (keyInput) keyInput.value = key;

    const provSelect = document.getElementById('ai-cfg-provider');
    if (provSelect) provSelect.value = provider;

    const modelInput = document.getElementById('ai-cfg-model');
    if (modelInput) modelInput.value = model;

    const baseInput = document.getElementById('ai-cfg-base');
    if (baseInput) baseInput.value = baseUrl;

    const notesInput = document.getElementById('ai-cfg-notes');
    if (notesInput) notesInput.value = notes;

    const autoCheck = document.getElementById('ai-eng-auto-toggle');
    if (autoCheck) autoCheck.checked = autoConsult;

    updateKeyStatusBadge();
}

function updateKeyStatusBadge() {
    const key = localStorage.getItem('ht_llm_api_key') || '';
    const badge = document.getElementById('ai-eng-key-status');
    if (!badge) return;

    if (key && key.trim().length > 4) {
        const prov = localStorage.getItem('ht_llm_provider') || 'auto';
        badge.innerHTML = `<span style="color:#00ff88">● ${prov.toUpperCase()} READY</span>`;
    } else {
        badge.innerHTML = `<span style="color:#ffd700">● KEY ON HOLD</span>`;
    }
}

function toggleAiEngineerConfig() {
    const drawer = document.getElementById('ai-eng-config-drawer');
    if (!drawer) return;
    const isOpen = drawer.style.display !== 'none';
    drawer.style.display = isOpen ? 'none' : 'block';
    if (!isOpen) {
        // Focus key input if opening and empty
        const keyInput = document.getElementById('ai-cfg-key');
        if (keyInput && !keyInput.value) keyInput.focus();
    }
}

function togglePasswordVisibility(id) {
    const input = document.getElementById(id);
    if (!input) return;
    input.type = input.type === 'password' ? 'text' : 'password';
}

function saveAiEngineerConfig() {
    const key = (document.getElementById('ai-cfg-key')?.value || '').trim();
    const provider = document.getElementById('ai-cfg-provider')?.value || 'auto';
    const model = (document.getElementById('ai-cfg-model')?.value || '').trim();
    const baseUrl = (document.getElementById('ai-cfg-base')?.value || '').trim();
    const notes = (document.getElementById('ai-cfg-notes')?.value || '').trim();

    localStorage.setItem('ht_llm_api_key', key);
    localStorage.setItem('ht_llm_provider', provider);
    localStorage.setItem('ht_llm_model', model);
    localStorage.setItem('ht_llm_base_url', baseUrl);
    localStorage.setItem('ht_llm_notes', notes);

    updateKeyStatusBadge();

    const statusEl = document.getElementById('ai-cfg-status');
    if (statusEl) {
        statusEl.innerHTML = `<span style="color:#00ff88;font-weight:700">✓ Config saved securely to local storage.</span>`;
        setTimeout(() => {
            if (statusEl) statusEl.innerHTML = '';
        }, 3000);
    }
}

function clearAiEngineerKey() {
    localStorage.removeItem('ht_llm_api_key');
    const keyInput = document.getElementById('ai-cfg-key');
    if (keyInput) keyInput.value = '';
    updateKeyStatusBadge();

    const statusEl = document.getElementById('ai-cfg-status');
    if (statusEl) {
        statusEl.innerHTML = `<span style="color:#ff9f43">Key cleared. Deterministic engine fallback active.</span>`;
        setTimeout(() => {
            if (statusEl) statusEl.innerHTML = '';
        }, 3000);
    }
}

function toggleAiAutoConsult(checked) {
    localStorage.setItem('ht_llm_auto_consult', checked ? 'true' : 'false');
}

/**
 * Consult the AI Race Engineer.
 * Pulls current unified scenario and sends to backend LLM transceiver endpoint.
 */
async function consultAiRaceEngineer(isAuto = false) {
    const btn = document.getElementById('ai-eng-consult-btn');
    const container = document.getElementById('ai-engineer-window');

    if (!container) return;

    // Validate race scenario
    const st = (typeof ucState === 'function') ? ucState() : null;
    if (st && st.error) {
        if (!isAuto) {
            container.innerHTML = `
                <div class="ai-eng-window">
                    <div style="color:#ff6b6b;font-family:'Share Tech Mono',monospace;font-size:0.85em;">
                        ⚠️ INCOMPLETE SCENARIO: ${st.error}. Configure drivers and track above first.
                    </div>
                </div>`;
            container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        return;
    }

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="radio-dot" style="background:#ffd700;box-shadow:0 0 8px #ffd700"></span> 📻 SYNTHESIZING CALL...`;
    }

    // Show high-tech scanning skeleton
    container.innerHTML = `
        <div class="ai-eng-window">
            <div class="ai-eng-topbar">
                <div class="ai-eng-channel">
                    <span class="radio-dot"></span>
                    <span class="ai-eng-callsign">AI RACE ENGINEER // PIT WALL TRANSCEIVER</span>
                    <span class="ai-eng-freq">CH-1 STRATEGY</span>
                </div>
                <div class="ai-eng-badges">
                    <span class="ai-badge ai-badge-source">SCANNING TELEMETRY...</span>
                </div>
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.8em;color:#ffd700;padding:20px;text-align:center;">
                <div style="margin-bottom:8px;">📡 Interrogating Multi-Policy Decision Engine & Telemetry Stream...</div>
                <div style="font-size:0.7em;color:#889;">Synthesizing tyre wear rates, fuel burn delta (-0.1355 s/lap), and isotonic overtake calibration...</div>
            </div>
        </div>`;

    if (!isAuto) {
        container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Build payload
    const key = localStorage.getItem('ht_llm_api_key') || '';
    const provider = localStorage.getItem('ht_llm_provider') || 'auto';
    const model = localStorage.getItem('ht_llm_model') || null;
    const baseUrl = localStorage.getItem('ht_llm_base_url') || null;
    const onlineNotes = (document.getElementById('ai-cfg-notes')?.value || localStorage.getItem('ht_llm_notes') || '').trim();

    const payload = {
        state: st ? {
            leader_code: st.leader,
            chaser_code: st.chaser,
            track_name: st.track,
            year: st.year,
            start_lap: st.start_lap,
            race_length: st.race_length,
            gap_before_s: st.gap_before_s,
            leader_tyre_compound: st.leader_tyre_compound,
            chaser_tyre_compound: st.chaser_tyre_compound,
            leader_tyre_age: st.leader_tyre_age,
            chaser_tyre_age: st.chaser_tyre_age,
            battery_pct: st.battery_pct,
            battery_band_pct: st.battery_band_pct,
            reserve_target_pct: st.reserve_target_pct,
            reserve_target_mj: st.reserve_target_mj,
            perspective: st.perspective || 'chaser',
        } : {},
        call_result: window.lastUcData || null,
        api_key: key || null,
        provider: provider,
        model: model || null,
        base_url: baseUrl || null,
        online_context: onlineNotes || null,
    };

    try {
        const res = await fetch('/api/ai-race-engineer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.error) throw new Error(data.error);

        window.lastAiEngineerData = data;
        renderAiEngineerCard(data, payload.state);

    } catch (err) {
        container.innerHTML = `
            <div class="ai-eng-window">
                <div class="ai-eng-topbar">
                    <div class="ai-eng-channel">
                        <span style="color:#ff6b6b;font-size:1.1em;">⚠️</span>
                        <span class="ai-eng-callsign">TRANSCEIVER LINK FAULT</span>
                    </div>
                </div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.78em;color:#ffb3b3;padding:10px 0;">
                    <div>Error contacting AI Race Engineer: <b>${escapeHtml(err.message)}</b></div>
                    <div style="margin-top:10px;">
                        <button class="action-btn" style="width:auto;margin:0;padding:6px 14px;font-size:0.8em" onclick="consultAiRaceEngineer()">⟳ RETRY TRANSCEIVER</button>
                        <button class="action-btn" style="width:auto;margin:0;background:#1c1c28;padding:6px 14px;font-size:0.8em" onclick="toggleAiEngineerConfig()">⚙️ CHECK CONFIG</button>
                    </div>
                </div>
            </div>`;
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<span class="radio-dot"></span> 📻 CONSULT AI RACE ENGINEER`;
        }
    }
}

/**
 * Render the full AI Race Engineer Pit Wall Transceiver Card
 */
function renderAiEngineerCard(data, state) {
    const container = document.getElementById('ai-engineer-window');
    if (!container) return;

    const isChaser = (state.perspective || 'chaser') === 'chaser';
    const driver = isChaser ? (state.chaser_code || 'LEC') : (state.leader_code || 'RUS');
    const opponent = isChaser ? (state.leader_code || 'RUS') : (state.chaser_code || 'LEC');
    const seatLabel = isChaser ? 'CHASER (ATTACK)' : 'LEADER (DEFENCE)';

    const isOffline = data.source === 'deterministic_engine_fallback' || data.mode === 'deterministic_offline';
    const sourceBadge = isOffline
        ? `<span class="ai-badge ai-badge-source offline">🛡️ DETERMINISTIC RACECRAFT ENGINE</span>`
        : `<span class="ai-badge ai-badge-source">⚡ LIVE LLM: ${(data.provider || 'AI').toUpperCase()} ${(data.model || '').toUpperCase()}</span>`;

    const latencyBadge = data.latency_ms
        ? `<span class="ai-badge" style="background:#151520;color:#aaa;border:1px solid #333">${data.latency_ms.toFixed(0)} ms</span>`
        : '';

    const confScore = data.confidence_score_pct || 80;
    const confLevel = data.confidence_level || (confScore >= 75 ? 'HIGH' : 'MEDIUM');
    const confClass = confScore >= 75 ? 'high' : '';

    // Directives HTML
    const directivesList = (data.driver_directives || []).map((dir, idx) => `
        <li class="ai-directive-item">
            <span class="ai-directive-num">D-${idx + 1}</span>
            <span>${escapeHtml(dir)}</span>
        </li>
    `).join('');

    // Telemetry Proof Matrix HTML
    const proofCards = (data.telemetry_proof || []).map((proof) => {
        const parts = proof.split(':');
        const title = parts.length > 1 ? parts[0] : 'Telemetry Proof';
        const body = parts.length > 1 ? parts.slice(1).join(':') : proof;
        return `
            <div class="ai-evidence-card">
                <div class="ai-evidence-lbl">${escapeHtml(title)}</div>
                <div class="ai-evidence-val" style="font-size:0.95em;">${escapeHtml(body.trim())}</div>
            </div>
        `;
    }).join('');

    container.innerHTML = `
        <div class="ai-eng-window">
            <!-- Transceiver Top Bar -->
            <div class="ai-eng-topbar">
                <div class="ai-eng-channel">
                    <span class="radio-dot"></span>
                    <span class="ai-eng-callsign">AI RACE ENGINEER // PIT WALL TRANSCEIVER</span>
                    <span class="ai-eng-freq">FREQ: CH-1 PIT TO CAR</span>
                </div>
                <div class="ai-eng-badges">
                    <span class="ai-badge" style="background:rgba(255,255,255,0.05);color:#ddd;border:1px solid #444">${seatLabel}</span>
                    ${sourceBadge}
                    ${latencyBadge}
                    <span class="ai-badge ai-badge-conf ${confClass}">${confScore}% CONFIDENCE [${confLevel}]</span>
                </div>
            </div>

            <!-- Authentic Team Radio Banner -->
            <div class="ai-radio-card">
                <div class="ai-radio-hdr">
                    <div class="ai-radio-tag">
                        <span>📻 PIT-TO-CAR RADIO // CALLSIGN: ${escapeHtml(driver)}</span>
                    </div>
                    <div class="ai-eq-bars" title="Audio transmission active">
                        <div class="ai-eq-bar"></div>
                        <div class="ai-eq-bar"></div>
                        <div class="ai-eq-bar"></div>
                        <div class="ai-eq-bar"></div>
                        <div class="ai-eq-bar"></div>
                    </div>
                </div>
                <div class="ai-radio-quote">
                    "${escapeHtml(data.radio_transmission || '')}"
                </div>
            </div>

            <!-- Definitive Tactical Call Card -->
            <div class="ai-def-call">
                <div>
                    <div class="ai-def-mode">${escapeHtml(data.definitive_call || 'THE CALL')}</div>
                    <div class="ai-def-sub">
                        TARGET: ${escapeHtml(driver)} VS ${escapeHtml(opponent)} · CIRCUIT: ${escapeHtml(state.track_name || 'Circuit')} · LAP ${state.start_lap || 1}/${state.race_length || 57}
                    </div>
                </div>
                <div class="ai-def-badge">
                    <div class="ai-def-score">${confScore}%</div>
                    <div class="ai-def-score-lbl">CONVICTION SCORE</div>
                </div>
            </div>

            <!-- Telemetry & ML Evidence Matrix -->
            <div class="ai-section-title">
                <span>⚡ TELEMETRY &amp; MACHINE LEARNING GROUNDING MATRIX</span>
            </div>
            <div class="ai-evidence-grid">
                ${proofCards}
            </div>

            <!-- Strategic Rationale / Trade-off Explanation -->
            <div class="ai-section-title">
                <span>📋 CHIEF STRATEGIST TACTICAL RATIONALE</span>
            </div>
            <div class="ai-rationale-text">
                ${escapeHtml(data.tactical_rationale || '').replace(/\n\n/g, '<br><br>')}
            </div>

            <!-- Driver Action Directives -->
            <div class="ai-section-title">
                <span>🎯 TURN-BY-TURN DRIVER DIRECTIVES</span>
            </div>
            <ul class="ai-directives-list">
                ${directivesList}
            </ul>

            <!-- Contingency Protocol -->
            <div class="ai-contingency-card">
                <b>⚠️ CONTINGENCY PROTOCOL:</b> ${escapeHtml(data.contingency_protocol || 'Maintain standard battery management and observe opponent reaction.')}
            </div>

            <!-- Footer / Status note -->
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;margin-top:14px;font-family:'Share Tech Mono',monospace;font-size:0.62em;color:#667;">
                <span>${escapeHtml(data.note || 'Pit Wall Strategy Transceiver')}</span>
                <span style="cursor:pointer;color:#00d2be;text-decoration:underline;" onclick="toggleAiEngineerConfig()">⚙️ Configure LLM Key / Provider</span>
            </div>
        </div>
    `;
}

// Utility: HTML Escaping
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
