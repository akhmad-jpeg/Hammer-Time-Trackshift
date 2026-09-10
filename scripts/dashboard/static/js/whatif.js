let selectedPTyre = null, selectedPTrack = null;
let selectedSTyre = null, selectedSTrack = null;
let selectedEvent = null;
let sessionFastestMs = Infinity;

function switchTab(name, btn) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.f1-nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
    if (name === 'drivers') loadDriverOptions();
}

// Hammer Time modular deck: open/close a module by id (all controls stay in the DOM).
function htToggle(modId, btn) {
    const mod = document.getElementById(modId);
    if (!mod) return;
    const open = mod.classList.toggle('open');
    if (btn) btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open && modId === 'ht-validation' && !mod.dataset.loaded) {
        mod.dataset.loaded = '1';
        loadEnergyBenchmark();
    }
}

// ── Hammer Time redesign helpers ─────────────────────────────────────────
function htVal(id) { const el = document.getElementById(id); return el ? el.value : ''; }

function htDriverLastName(code) {
    const list = (typeof ovDrivers !== 'undefined') ? ovDrivers : [];
    const d = list.find(x => x.code === code);
    return d ? String(d.name || code).split(' ').slice(-1)[0] : (code || '—');
}

function htTrackShort(t) {
    return String(t || '').replace(/\s+(Circuit|International Racing Course)$/i, '').toUpperCase();
}

// Live-update the Race Context Bar (and the hero line / tuner inherit chips)
// from the Race Call inputs — the single source of truth for the scenario.
function htUpdateContext() {
    const L = htVal('ov-leader'), C = htVal('ov-chaser'), T = htVal('ov-track'), Y = htVal('ov-year');
    const lap = htVal('ov-lap') || '—';
    const gap = htVal('ov-gap');
    const lt = htVal('ov-ltyre'), ct = htVal('ov-ctyre');
    const la = htVal('ov-lage'), ca = htVal('ov-cage');
    const battle = document.getElementById('ctx-battle');
    if (battle) battle.textContent = (L || '—') + ' → ' + (C || '—');
    const meta = document.getElementById('ctx-meta');
    if (meta) meta.textContent = (T ? htTrackShort(T) : '—') + ' · ' + (Y || '—') + ' · LAP ' + lap;
    const g = document.getElementById('ctx-gap');
    if (g) g.textContent = 'GAP ' + (gap ? gap + 's' : '—');
    const ty = document.getElementById('ctx-tyres');
    if (ty) ty.textContent = 'TYRES ' + (lt || '—') + ' ' + (la || '?') + 'L / ' + (ct || '—') + ' ' + (ca || '?') + 'L';
    const raw = htVal('ov-ers-batt');
    const bt = document.getElementById('ctx-batt');
    if (bt) bt.textContent = 'BATT ' + (raw === '' ? 'AUTO (~62%)' : Math.round(100 * (parseFloat(raw) || 0) / 4.0) + '%');
    const hero = document.getElementById('ht-hero-line');
    if (hero) {
        hero.textContent = (L && C && T)
            ? htDriverLastName(L) + ' leads ' + htDriverLastName(C) + ' by ' + (gap || '—') + 's on Lap ' + lap + '.'
            : 'Set the scenario and the race state, then make the call.';
    }
    const bi = document.getElementById('bt-inherited');
    if (bi) bi.textContent = (L || '—') + ' vs ' + (C || '—') + ' · ' + (T || '—') + ' ' + (Y || '—') +
        ' · LAP ' + lap + ' · GAP ' + (gap || '—') + 's · ' + (lt || '—') + ' ' + (la || '?') + 'L vs ' + (ct || '—') + ' ' + (ca || '?') + 'L';
}

function htEditScenario(btn) {
    const mod = document.getElementById('ht-scenario');
    if (!mod) return;
    const open = mod.classList.toggle('open');
    if (btn) btn.textContent = open ? 'CLOSE SCENARIO' : 'EDIT SCENARIO';
    if (open) mod.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function elShow(which) {
    const comp = document.getElementById('el-compare'), sand = document.getElementById('el-sandbox');
    const tc = document.getElementById('el-tab-compare'), ts = document.getElementById('el-tab-sandbox');
    if (!comp || !sand) return;
    comp.style.display = which === 'compare' ? '' : 'none';
    sand.style.display = which === 'sandbox' ? '' : 'none';
    if (tc) tc.classList.toggle('active', which === 'compare');
    if (ts) ts.classList.toggle('active', which === 'sandbox');
}

function htMenuToggle(listId) {
    const list = document.getElementById(listId);
    if (list) list.style.display = (list.style.display === 'none') ? 'flex' : 'none';
}

// Fill both Energy Lab session fields from the Race Call scenario (resolving
// the sessions first if the Replay module hasn't resolved them yet).
function htInheritSession() {
    const sid = htVal('ovr-leader');
    if (sid) {
        ['e-session', 'sb-session'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = sid;
        });
        const lap = parseInt(htVal('ov-lap'), 10);
        if (Number.isFinite(lap) && lap > 1) {
            ['e-lap', 'sb-lap'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.value = lap;
            });
        }
    } else {
        resolveRaceSessions();   // resolves from the scenario and inherits
    }
}

// LIVE SYNC — fill the Race Call state from the live telemetry feed instead
// of manual inputs: tyres, ages, current lap and the timing gap (measured at
// the line on the two cars' latest shared lap).  Falls back to the two most
// recently live drivers when the selected pair has no laps in the window.
function htLiveSyncSetSelect(id, v) {
    const el = document.getElementById(id);
    if (!el || v == null || v === '') return;
    if (![...el.options].some(o => o.value === String(v))) {
        const o = document.createElement('option');
        o.value = v; o.textContent = v;
        el.appendChild(o);
    }
    el.value = v;
    el.dispatchEvent(new Event('change', { bubbles: true }));
}

async function htLiveSync(btn) {
    const status = document.getElementById('live-sync-status');
    const setStatus = (msg, col) => {
        if (status) { status.textContent = msg; status.style.color = col || '#9ab'; status.style.display = msg ? 'block' : 'none'; }
    };
    if (btn) { btn.disabled = true; btn.textContent = 'SYNCING…'; }
    setStatus('');
    const apply = d => {
        htLiveSyncSetSelect('ov-leader', d.leader.code);
        htLiveSyncSetSelect('ov-chaser', d.chaser.code);
        if (d.track) htLiveSyncSetSelect('ov-track', d.track);
        if (d.year) document.getElementById('ov-year').value = d.year;
        document.getElementById('ov-lap').value = Math.max(d.leader.lap, d.chaser.lap) || 1;
        htLiveSyncSetSelect('ov-ltyre', d.leader.tyre);
        htLiveSyncSetSelect('ov-ctyre', d.chaser.tyre);
        document.getElementById('ov-lage').value = d.leader.tyre_age;
        document.getElementById('ov-cage').value = d.chaser.tyre_age;
        if (d.gap_s != null) setGapVal(Math.max(0.1, Math.round(Math.abs(d.gap_s) * 10) / 10));
        htUpdateContext();
    };
    try {
        const leader = document.getElementById('ov-leader').value;
        const chaser = document.getElementById('ov-chaser').value;
        let res = await fetch('/api/live/battle-state?leader=' + encodeURIComponent(leader) + '&chaser=' + encodeURIComponent(chaser));
        let data = await res.json();
        if (data.error) throw new Error(data.error);
        if (!data.live && (data.live_now || []).length >= 2 && (data.stale || []).length) {
            res = await fetch('/api/live/battle-state?leader=' + encodeURIComponent(data.live_now[0]) + '&chaser=' + encodeURIComponent(data.live_now[1]));
            data = await res.json();
            if (data.error) throw new Error(data.error);
        }
        if (!data.live) {
            setStatus('No live telemetry in the last 10 minutes (stale: ' + (data.stale || []).join(', ') + ') — start a capture first.', '#ff6b6b');
            return;
        }
        // If the picked "leader" is actually ahead-in-reverse (negative
        // gap), swap the roles so the gap stays leader-relative.
        let swapped = false;
        if (data.gap_s != null && data.gap_s < 0) {
            const t = data.leader; data.leader = data.chaser; data.chaser = t;
            data.gap_s = Math.abs(data.gap_s);
            swapped = true;
        }
        apply(data);
        const L = data.leader, C = data.chaser;
        setStatus('● LIVE — ' + L.code + ' ahead of ' + C.code + ' by ' +
            (data.gap_s != null ? data.gap_s.toFixed(1) + 's' : '?') +
            ' on L' + Math.max(L.lap, C.lap) +
            ' (' + (L.tyre || '?') + ' ' + L.tyre_age + 'L vs ' + (C.tyre || '?') + ' ' + C.tyre_age + 'L)' +
            (swapped ? ' · roles auto-swapped (your chaser pick was actually ahead)' : '') +
            (data.same_race ? '' : ' · ⚠ drivers live on DIFFERENT tracks'),
            data.same_race ? '#00c853' : '#ffd700');
    } catch (e) {
        setStatus('Live sync error: ' + e.message, '#ff6b6b');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '● LIVE SYNC'; }
    }
}

let htModelSummary = null;
function htSetModelStatus(s) {
    htModelSummary = s;
    const word = document.getElementById('mc-word');
    const dot = document.getElementById('mc-dot');
    const healthy = s && (s.battle_laps || 0) > 0;
    if (word) word.textContent = healthy ? 'HEALTHY' : 'NO DATA';
    if (dot) dot.className = 'mc-dot ' + (healthy ? 'ok' : 'bad');
}

function htToggleModel() {
    const p = document.getElementById('ht-model-panel');
    if (!p) return;
    if (p.style.display !== 'none') { p.style.display = 'none'; return; }
    const s = htModelSummary;
    p.innerHTML = !s
        ? '<div class="chart-note">Model summary not loaded yet.</div>'
        : '<div class="mp-title">MODEL HEALTH</div>' +
          '<div class="mp-row"><span>Battle samples</span><b>' + (s.battle_laps || 0).toLocaleString() + '</b></div>' +
          '<div class="mp-row"><span>Overtake labels</span><b>' + (s.overtake_labels != null ? s.overtake_labels : '—') + (s.overtake_rate != null ? ' (' + (100 * s.overtake_rate).toFixed(1) + '%)' : '') + '</b></div>' +
          '<div class="mp-row"><span>Closing MAE</span><b>' + (s.closing_mae != null ? s.closing_mae + 's' : '—') + '</b></div>' +
          '<div class="mp-row"><span>Overtake AUC</span><b>' + (s.overtake_auc != null ? s.overtake_auc : '—') + '</b></div>' +
          '<div class="mp-row"><span>Training date</span><b>' + ((s.trained_at || '').slice(0, 16).replace('T', ' ') || '—') + '</b></div>' +
          '<div class="mp-row"><span>Coverage</span><b><span class="mc-dot ok">●</span> Good</b></div>';
    p.style.display = 'block';
}

// Close popovers when clicking anywhere else.
document.addEventListener('click', e => {
    const menu = document.getElementById('ovr-export-list');
    if (menu && menu.style.display !== 'none' && !e.target.closest('.export-menu')) menu.style.display = 'none';
    const mp = document.getElementById('ht-model-panel');
    if (mp && mp.style.display !== 'none' && !e.target.closest('.ctx-bar')) mp.style.display = 'none';
});

let ovDrivers = [];
let ovTracks = [];

function fillSelect(id, items, valueKey, labelFn, placeholder) {
    const sel = document.getElementById(id);
    sel.innerHTML = '';
    if (placeholder) {
        const o = document.createElement('option');
        o.value = ''; o.textContent = placeholder;
        sel.appendChild(o);
    }
    items.forEach(it => {
        const o = document.createElement('option');
        o.value = valueKey ? it[valueKey] : it;
        o.textContent = labelFn(it);
        sel.appendChild(o);
    });
}

async function loadOvertakeOptions() {
    const err = document.getElementById('ov-err');
    try {
        const res = await fetch('/api/overtake/options');
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        ovDrivers = data.drivers || [];
        ovTracks = data.tracks || [];
        const leader = document.getElementById('ov-leader');
        const chaser = document.getElementById('ov-chaser');
        fillSelect('ov-leader', ovDrivers, 'code', d => d.name + ' (' + d.code + ')', 'Pick leader...');
        fillSelect('ov-chaser', ovDrivers, 'code', d => d.name + ' (' + d.code + ')', 'Pick chaser...');
        if (ovDrivers.length >= 2) {
            leader.value = ovDrivers[0].code;
            chaser.value = ovDrivers[1].code;
            const lc = ovDrivers.map(d => d.code);
            if (lc.includes('HAM') && lc.includes('VER')) {
                leader.value = 'HAM'; chaser.value = 'VER';
            } else if (lc.includes('LEC') && lc.includes('RUS')) {
                leader.value = 'LEC'; chaser.value = 'RUS';
            }
        }
        fillSelect('ov-track', ovTracks, '', t => t, 'Pick track...');
        const tyres = data.tyres || ['Soft', 'Medium', 'Hard', 'Intermediate', 'Wet'];
        fillSelect('ov-ltyre', tyres, '', t => t, null);
        fillSelect('ov-ctyre', tyres, '', t => t, null);
        document.getElementById('ov-ltyre').value = 'Medium';
        document.getElementById('ov-ctyre').value = 'Medium';
        if (data.summary) htSetModelStatus(data.summary);
        htUpdateContext();
    } catch (e) {
        err.textContent = 'Overtake options error: ' + e.message;
        err.style.display = 'block';
    }
}

async function runOvertakeSim() {
    const err = document.getElementById('bt-err');
    err.style.display = 'none';
    const btn = [...document.querySelectorAll('button')].find(b => /SIMULATE BATTLE/.test(b.innerText || ''));
    if (btn) { btn.disabled = true; btn.textContent = 'MODELLING...'; }
    try {
        const fuelDiff = parseFloat(document.getElementById('ov-fueldiff').value);
        const energyDiff = parseFloat(document.getElementById('ov-energydiff').value);
        const body = {
            leader_code: document.getElementById('ov-leader').value,
            chaser_code: document.getElementById('ov-chaser').value,
            track_name: document.getElementById('ov-track').value,
            year: parseInt(document.getElementById('ov-year').value, 10) || null,
            lap_number: parseInt(document.getElementById('ov-lap').value, 10) || 20,
            gap_before_s: parseFloat(document.getElementById('ov-gap').value) || 0.8,
            leader_tyre_compound: document.getElementById('ov-ltyre').value,
            chaser_tyre_compound: document.getElementById('ov-ctyre').value,
            leader_tyre_age: parseInt(document.getElementById('ov-lage').value, 10) || 0,
            chaser_tyre_age: parseInt(document.getElementById('ov-cage').value, 10) || 0,
            fuel_diff_kg: Number.isFinite(fuelDiff) ? fuelDiff : -3.0,
            energy_diff_mj: Number.isFinite(energyDiff) ? energyDiff : 0.0,
            sim_laps: parseInt(document.getElementById('ov-simlaps').value, 10) || 1,
        };
        const res = await fetch('/api/overtake/sim', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        const out = document.getElementById('ov-out');
        const s = data.single;
        const pd = data.pace_gap_detail || {};
        const leadM = pd.leader_model ? (pd.leader_model[1] || 'aggregate') : '?';
        const chasM = pd.chaser_model ? (pd.chaser_model[1] || 'aggregate') : '?';
        const probPct = Math.round(100 * s.overtake_probability);
        const likely = probPct >= 50;
        const sim0 = (data.sim && data.sim.laps && data.sim.laps.length) ? data.sim : null;
        const lastGap = sim0 ? sim0.laps[sim0.laps.length - 1].gap_before_s : null;
        const statRow = (label, val, col) =>
            `<div class="hc-stat"><span class="hcs-l">${label}</span><span class="hcs-v"${col ? ` style="color:${col}"` : ''}>${val}</span></div>`;
        let html =
            `<div class="hero-call" style="margin-top:14px">` +
            `<div class="hc-big">${probPct}%</div>` +
            `<div class="hc-big-l">P(OVERTAKE) · LAP ${body.lap_number} · ${body.track_name.toUpperCase()} ${body.year || ''}</div>` +
            (likely
                ? `<div class="bt-verdict pass">OVERTAKE LIKELY — LAP ${body.lap_number}</div>`
                : `<div class="bt-verdict nopass">NO CLEAN OVERTAKE${sim0 ? ` — GAP ${lastGap.toFixed(2)}s AFTER ${sim0.laps.length} LAPS` : ''}</div>`) +
            `<div class="hc-stats">` +
            statRow('CLOSING RATE', (s.closing_rate_s > 0 ? '+' : '') + s.closing_rate_s + 's/lap', s.closing_rate_s >= 0 ? '#00c853' : '#ff6b6b') +
            (lastGap != null ? statRow('PROJECTED GAP', lastGap.toFixed(2) + 's') : '') +
            statRow('PACE GAP (LEADER − CHASER)', (data.pace_gap_s >= 0 ? '+' : '') + data.pace_gap_s + 's') +
            `</div>` +
            `<div class="bt-why">${likely
                ? 'The battle converts — the chaser gains ' + (s.closing_rate_s > 0 ? '+' + s.closing_rate_s.toFixed(2) + 's/lap' : 'enough') + ' on ' + body.chaser_tyre_compound + ' age ' + body.chaser_tyre_age + ' vs ' + body.leader_tyre_compound + ' age ' + body.leader_tyre_age + '.'
                : 'The chaser ' + (s.closing_rate_s > 0 ? 'is closing at +' + s.closing_rate_s.toFixed(2) + 's/lap' : 'is not closing — the gap is holding or growing') + '. ' + (data.pace_gap_s > 0 ? 'A pace edge exists on this tyre state.' : 'No pace edge on this tyre state — try a fresher or softer tyre, or an energy delta.')}` +
            `</div></div>`;
        if (!s.track_covered) {
            html += `<div class="chart-note" style="margin-top:8px;color:#ff6b6b">⚠ Track unseen in overtake training — probability is uncalibrated here; treat as the pace-gap fallback.</div>`;
        }
        if (data.sim && data.sim.laps.length) {
            const sim = data.sim;
            const BATTLE_COLS = 'grid-template-columns:52px 90px 110px 130px 44px 1fr;';
            const battleHead =
                `<div class="tbl-row tbl-head" style="${BATTLE_COLS}">` +
                `<span>LAP</span>` +
                `<span>GAP</span>` +
                `<span>CLOSING/LAP</span>` +
                `<span>P(OVERTAKE)</span>` +
                `<span>PCT</span>` +
                `<span>EVENT</span></div>`;
            const rows = sim.laps.map(l => {
                const pct = Math.round(100 * l.overtake_probability);
                const bar = `<div style="height:10px;background:linear-gradient(90deg,#ff6b6b 0 ${pct}%,#1c1c24 ${pct}% 100%);border-radius:3px;min-width:60px;width:100%"></div>`;
                const pass = l.passed ? '<span class="pass-chip">▲ Overtake</span>' : '';
                return `<div class="tbl-row${l.passed ? ' pass-row' : ''}" style="${BATTLE_COLS}">` +
                    `<span>L${l.lap}</span>` +
                    `<span>${l.gap_before_s.toFixed(2)}s</span>` +
                    `<span>${(l.closing_rate_s > 0 ? '+' : '') + l.closing_rate_s.toFixed(2)}s/lap</span>` +
                    `<span>${bar}</span>` +
                    `<span>${pct}%</span>` +
                    `<span>${pass}</span></div>`;
            }).join('');
            const passLaps = sim.laps.filter(l => l.passed).map(l => l.lap);
            const verdict = sim.pass_lap
                ? `<div class="pass-verdict"><span>▲ Overtake on lap</span><span class="pass-lap-big">${sim.pass_lap}</span>` +
                  (passLaps.length > 1 ? `<span class="pass-extra">also on laps ${passLaps.filter(l => l !== sim.pass_lap).join(', ')}</span>` : '') +
                  `</div>`
                : `<div class="pass-verdict no-pass">NO PASS — battle ${sim.ended === 'blown_open' ? 'blown open' : 'ran its laps'}</div>`;
            html += `<details class="ht-hint" style="margin-top:14px"><summary>ENGINEERING DETAILS — battle roll-forward (${chasM} vs ${leadM} year models)</summary>` +
                `<div class="panel-title" style="margin-top:10px">Battle Sim — ${sim.laps.length} laps <span class="pass-lap-marker" style="font-size:0.55em">red rows = overtake laps</span></div>${verdict}<div class="tbl-wrap">${battleHead}${rows}</div>` +
                `<div class="chart-note" style="margin-top:8px">Gap delta ${body.gap_before_s}s · pace gap (leader − chaser) ${data.pace_gap_s > 0 ? '+' : ''}${data.pace_gap_s}s · fuel diff ${body.fuel_diff_kg} kg · energy diff ${body.energy_diff_mj} MJ.</div>` +
                `</details>`;
        }
        out.innerHTML = html;
    } catch (e) {
        err.textContent = 'Error: ' + e.message;
        err.style.display = 'block';
    }
    if (btn) { btn.disabled = false; btn.textContent = 'SIMULATE BATTLE'; }
}

// ── P1 FULL-RACE OVERTAKE SIMULATOR (speed-trace aligned) ─────────────
let ovrGapChart = null;
let ovrSectorChart = null;
let ovrModeChart = null;
let lastRaceSim = null;

const OVR_MODE_COLORS = { balanced: '#ffd700', push: '#ff6b6b', liftcoast: '#00c853', stored: '#9aa' };
const TRUST_COLORS = { high: '#00c853', medium: '#ffd700', low: '#ff6b6b', insufficient: '#777' };
const trustDot = t => `<span style="color:${TRUST_COLORS[t] || '#777'}" title="trust: ${t}">●${(t || 'insufficient').toUpperCase()}</span>`;

// An ERS energy source is either a plain mode key (both cars) or an
// ASYMMETRIC duel key "<leader>><<chaser>" (e.g. "push>balanced" = leader
// attacks on Push while the chaser holds Balanced).  Colour by the leader
// car's spec so a duel reads against the attacking side; "stored>X" rows
// (real trace vs a what-if mode) inherit the stored grey.
// Tyre-health traffic light for the replay's DRIVERS column (shared
// health timeline with the main dashboard's Tyre Degradation chart).
const healthColor = h => h == null ? '#888' : h >= 60 ? '#00c853' : h >= 30 ? '#ffd700' : '#ff6b6b';
const tyreHealthSpan = h => h == null
    ? ''
    : `<span style="color:${healthColor(h)}" title="tyre health (Pirelli-style model)">&nbsp;·&nbsp;${Math.round(h)}%</span>`;

const srcColor = key => {
    if (!key) return '#ccc';
    const s = String(key);
    const i = s.indexOf('>');
    const lead = i > 0 ? s.slice(0, i) : s;
    return OVR_MODE_COLORS[lead] || '#ccc';
};
const srcLabel = key => {
    if (!key) return '';
    const s = String(key).toUpperCase();
    return s.indexOf('>') > 0 ? s.split('>').join(' > ') : s;
};
const srcTitle = key => {
    if (!key) return '';
    const s = String(key);
    const i = s.indexOf('>');
    if (i <= 0) return `both cars on ${s}`;
    const lm = s.slice(0, i), cm = s.slice(i + 1);
    const name = sp => sp === 'stored' ? 'as stored (real trace)' : `${sp} mode`;
    return `leader ${name(lm)} vs chaser ${name(cm)}`;
};
const fmtSpecMJ = v => (v == null ? 'real' : v.toFixed(1) + ' MJ');
const fmtSpecPct = v => (v == null ? 'real' : v.toFixed(1) + '%');

// Vertical "OVERTAKE L{n}" annotation for the P1 charts.  `marks` = array of
// { idx (index into the chart's laps), lap, color, label, datasetIndex }.
// The x pixel is read from the rendered point of that dataset so it is exact
// regardless of axis auto-skip or banding.
function passLapAnnotationPlugin(marks) {
    return {
        id: 'passLapAnnotation',
        afterDatasetsDraw(chart) {
            const { ctx, chartArea } = chart;
            if (!chartArea || !marks.length) return;
            const top = chartArea.top + 2;
            const bottom = chartArea.bottom;
            marks.forEach(mk => {
                const meta = chart.getDatasetMeta(mk.datasetIndex || 0);
                const pt = meta && meta.data && meta.data[mk.idx];
                const x = pt && isFinite(pt.x) ? pt.x : null;
                if (x === null) return;
                const col = mk.color || '#ff6b6b';
                ctx.save();
                // Dashed vertical line through the pass lap.
                ctx.strokeStyle = col;
                ctx.globalAlpha = 0.85;
                ctx.lineWidth = 1.5;
                ctx.setLineDash([5, 4]);
                ctx.beginPath();
                ctx.moveTo(x, top);
                ctx.lineTo(x, bottom);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = 1;
                // Small pill label above the line.
                const label = mk.label || 'OVERTAKE L' + mk.lap;
                ctx.font = "700 8.5px 'Share Tech Mono', monospace";
                const w = Math.round(ctx.measureText(label).width) + 8;
                const lx = Math.max(chartArea.left + 2, Math.min(x - w / 2, chartArea.right - w - 2));
                ctx.fillStyle = 'rgba(20,20,26,0.92)';
                ctx.fillRect(lx, top, w, 13);
                ctx.strokeStyle = col;
                ctx.lineWidth = 1;
                ctx.strokeRect(lx + 0.5, top + 0.5, w - 1, 12);
                ctx.fillStyle = col;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, lx + w / 2, top + 7);
                ctx.restore();
            });
        }
    };
}

function circuitHeatmapSvg(heat, size, base) {
    // Schematic circuit: the 12 speed-trace segments drawn as arc bands
    // around a ring, in around-the-lap order — so each braking zone sits
    // at its approximate circuit position.  Yellow band = a lap's hot
    // corner.  Segment 1 starts at 12 o'clock.
    // When `base` (a reference heat array of the same segments) is given,
    // each band renders the DELTA vs base instead of the absolute mass:
    // green = this source shifts MORE pass mass into the zone than stored,
    // red = it shifts LESS.  Every ring is the same size so they read as a
    // comparable row.
    const n = (heat && heat.length) || 12;
    const cx = size / 2, cy = size / 2;
    const r0 = size * 0.36, sw = size * 0.13;
    const C0 = 2 * Math.PI * r0;
    const seg = C0 / n;
    const delta = base && base.length === n;
    const vals = heat.map((h, j) => delta
        ? h.total_probability - (base[j] ? base[j].total_probability : 0)
        : h.total_probability);
    const maxAbs = delta ? Math.max.apply(null, vals.map(v => Math.abs(v))) : 0;
    const maxTot = delta ? 0 : (heat.length ? Math.max.apply(null, vals) : 0);
    let arcs = '';
    heat.forEach((h, j) => {
        const v = vals[j];
        let stroke, a, label;
        if (delta) {
            stroke = v >= 0 ? '#00c853' : '#ff6b6b';
            a = maxAbs > 0 ? (0.14 + 0.86 * Math.abs(v) / maxAbs) : 0.10;
            label = `Δ pass prob ${v >= 0 ? '+' : ''}${v.toFixed(4)} vs stored`;
        } else {
            stroke = 'rgba(255,107,107,1)';
            a = maxTot > 0 ? (0.10 + 0.90 * v / maxTot) : 0.10;
            label = `total pass prob ${v.toFixed(4)}`;
        }
        const off = C0 - (j + 0.5) * seg;
        arcs += `<circle cx="${cx}" cy="${cy}" r="${r0}" fill="none" stroke="${stroke}" stroke-opacity="${a}" stroke-width="${sw}" stroke-dasharray="${seg} ${C0}" stroke-dashoffset="${off}" transform="rotate(-90 ${cx} ${cy})"><title>Corner zone ${h.segment} · ${Math.round(100 * h.fraction_start)}–${Math.round(100 * h.fraction_end)}% of lap · ${label} · hot corner on ${h.hot_count} lap(s)</title></circle>`;
        if (h.hot_count > 0) {
            arcs += `<circle cx="${cx}" cy="${cy}" r="${r0 + sw / 2 + 2}" fill="none" stroke="#ffd700" stroke-width="2" stroke-dasharray="${seg} ${C0}" stroke-dashoffset="${off}" transform="rotate(-90 ${cx} ${cy})"><title>Hot corner zone ${h.segment} (${h.hot_count} lap(s))</title></circle>`;
        }
    });
    let labels = '';
    heat.forEach((h, j) => {
        const ang = 2 * Math.PI * (j + 0.5) / n - Math.PI / 2;
        const lx = cx + (r0 + sw / 2 + size * 0.06) * Math.cos(ang);
        const ly = cy + (r0 + sw / 2 + size * 0.06) * Math.sin(ang);
        labels += `<text x="${lx}" y="${ly + size * 0.016}" text-anchor="middle" font-size="${size * 0.042}" fill="#999" font-family="Share Tech Mono, monospace">${h.segment}</text>`;
    });
    return `<svg viewBox="0 0 ${size} ${size}" style="max-width:100%;height:auto">` +
        `<circle cx="${cx}" cy="${cy}" r="${r0 + sw / 2 + 1}" fill="none" stroke="#ffffff18" stroke-width="1"/>` +
        `<circle cx="${cx}" cy="${cy}" r="${r0 - sw / 2 - 1}" fill="none" stroke="#ffffff12" stroke-width="1"/>` +
        arcs + labels +
        `<text x="${cx}" y="${cy - r0 - sw / 2 - size * 0.045}" text-anchor="middle" font-size="${size * 0.05}" fill="#ffd700" font-family="Share Tech Mono, monospace">S/F</text>` +
        `</svg>`;
}

function sectorBarPct(pct, color) {
    const c = Math.max(0, Math.min(100, pct));
    return `<div style="width:56px;height:9px;background:linear-gradient(90deg,${color} 0 ${c}%,#1c1c24 ${c}% 100%);border-radius:3px;display:inline-block"></div>`;
}

async function resolveRaceSessions() {
    const err = document.getElementById('ovr-err');
    err.style.display = 'none';
    const leader = document.getElementById('ov-leader').value;
    const chaser = document.getElementById('ov-chaser').value;
    const track = document.getElementById('ov-track').value;
    const year = parseInt(document.getElementById('ov-year').value, 10) || 2026;
    if (!leader || !chaser || !track) {
        err.textContent = 'Pick a leader, chaser and track first (P0 options above).';
        err.style.display = 'block';
        return null;
    }
    try {
        const res = await fetch(`/api/overtake/sessions?leader=${encodeURIComponent(leader)}&chaser=${encodeURIComponent(chaser)}&track=${encodeURIComponent(track)}&year=${year}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        if (data.missing && data.missing.length) {
            err.textContent = 'No race session found for: ' + data.missing.join(', ') +
                ' at ' + track + ' ' + year + ' (try another year/track).';
            err.style.display = 'block';
            return null;
        }
        const s = data.sessions || {};
        document.getElementById('ovr-leader').value = (s[leader] || {}).session_id || '';
        document.getElementById('ovr-chaser').value = (s[chaser] || {}).session_id || '';
        // Energy Lab inherits the resolved leader session so its tools run on
        // the same race the wall is analysing (spec: no repeated session ids).
        const leaderSid = (s[leader] || {}).session_id;
        if (leaderSid) {
            ['e-session', 'sb-session'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.value = leaderSid;
            });
        }
        const ctx = document.getElementById('ovr-context');
        if (s[leader] && s[chaser]) {
            const span = Math.min(s[leader].max_lap || 0, s[chaser].max_lap || 0);
            ctx.textContent = `${leader} (${s[leader].name}) session #${s[leader].session_id} · ${chaser} (${s[chaser].name}) session #${s[chaser].session_id} · ${data.track} ${data.year} · race laps 1–${span} · Energy Lab session inherited`;
        }
        return data;
    } catch (e) {
        err.textContent = 'Resolve sessions error: ' + e.message;
        err.style.display = 'block';
        return null;
    }
}

async function scanSeasonCalibration() {
    // P0-model calibration across every race the leader/chaser pair share
    // this season: closing-rate predictions vs the real lap-time deltas.
    // First call runs a light sim per race (~2-3s each, then cached).
    const err = document.getElementById('ovr-err');
    err.style.display = 'none';
    const leader = document.getElementById('ov-leader').value;
    const chaser = document.getElementById('ov-chaser').value;
    const year = parseInt(document.getElementById('ov-year').value, 10) || 2026;
    const out = document.getElementById('ovr-cal-out');
    const btn = document.getElementById('ovr-cal-scan');
    if (!leader || !chaser) {
        err.textContent = 'Pick a leader and chaser first (P0 options above).';
        err.style.display = 'block';
        return;
    }
    if (btn) { btn.disabled = true; btn.textContent = 'SCANNING...'; }
    out.innerHTML = `<div class="chart-note">Scanning every ${year} race where ${leader} and ${chaser} both have sessions — light sim per race (~2-3s each, then cached for this pair/season)...</div>`;
    try {
        const res = await fetch(`/api/overtake/calibration?leader=${encodeURIComponent(leader)}&chaser=${encodeURIComponent(chaser)}&year=${year}`);
        const d = await res.json();
        if (d.error) throw new Error(d.error);
        const SCAN_COLS = 'grid-template-columns:repeat(8,minmax(0,1fr));';
        const rows = d.races.map(rc => {
            if (rc.error) {
                return `<div class="tbl-row" style="grid-template-columns:1fr auto;">` +
                    `<span title="${rc.track}">${rc.track}</span>` +
                    `<span style="color:#ff6b6b;white-space:normal" title="${rc.error}">⚠ ${rc.error}</span></div>`;
            }
            const flips = (rc.flip_laps || []).slice(0, 5).map(l => 'L' + l).join(', ') +
                ((rc.flip_laps || []).length > 5 ? ` +${rc.flip_laps.length - 5}` : '');
            const netTxt = rc.net_direction_ok === false
                ? '<b style="color:#ff6b6b">⚠ net flips</b>'
                : `net ${(rc.net_model_gain_s || 0).toFixed(1)} vs ${(rc.net_actual_gain_s || 0).toFixed(1)}s`;
            return `<div class="tbl-row" style="${SCAN_COLS}">` +
                `<span title="${rc.track}">${rc.track}</span>` +
                `<span title="trust tier from agreement / R² / net direction">${trustDot(rc.trust)}</span>` +
                `<span title="scored laps">${rc.n}</span>` +
                `<span title="% of laps where the model's sign matches reality">${Math.round(100 * (rc.agreement_pct || 0))}%</span>` +
                `<span title="R² of predicted closing vs real delta">${(rc.r2 || 0).toFixed(2)}</span>` +
                `<span title="OLS slope (1 = scale-true)">${(rc.slope || 0).toFixed(2)}</span>` +
                `<span title="${flips}">${rc.sign_flips || 0}${flips ? ' (' + flips + ')' : ''}</span>` +
                `<span title="cumulative model gain vs cumulative real gain">${netTxt}</span></div>`;
        }).join('');
        const tc = d.trust_counts || {};
        const calHead =
            `<div class="tbl-row tbl-head" style="${SCAN_COLS}">` +
            `<span>TRACK</span>` +
            `<span>TRUST</span>` +
            `<span>LAPS</span>` +
            `<span>AGREE</span>` +
            `<span>R²</span>` +
            `<span>SLOPE</span>` +
            `<span>FLIPS</span>` +
            `<span>NET (model vs real)</span></div>`;
        out.innerHTML =
            `<div class="panel-title" style="margin-top:8px">Season calibration — ${d.leader} vs ${d.chaser} ${d.year} (which races the P0 projection can be trusted on)</div>` +
            `<div style="background:#101014;border:1px solid #ffffff14;border-radius:8px;padding:8px 10px">${calHead}${rows || '<div class="chart-note">No shared race sessions this season.</div>'}</div>` +
            `<div class="chart-note" style="margin-top:4px">high ${tc.high || 0} · medium ${tc.medium || 0} · low ${tc.low || 0} · insufficient ${tc.insufficient || 0} — ●high = ≥65% sign agreement, R² ≥ 0.25, matching net direction; races flagged ⚠ net flips contradict the real lap evidence (scenario, not prediction)</div>`;
    } catch (e) {
        out.innerHTML = `<div class="chart-note" style="color:#ff6b6b">Calibration scan error: ${e.message}</div>`;
    }
    if (btn) { btn.disabled = false; btn.textContent = 'SCAN SEASON CALIBRATION'; }
}

// ── RACE CALL (live co-pilot): current tyre state + gap in → forward projection + the call ──
let lastLiveCompare = null;
const LIVE_WINDOW_S = 1.2;   // deck: P0 overtake signal only lives inside ~DRS range
const LIVE_PASS_CUM = 0.8;   // deck: cumulative P(overtake) treated as the projected pass (raised from 0.5 after the race-call backtest over-calling passes that never happened)
// Conversion phrasing is grounded in backtest_race_calls.py (HAM/VER 2021
// sample): a projected pass converted ~12% of the time and >=80%-confidence
// calls ~7%.  Re-run the harness over the full sweep and update these.
const BT_PASS_CONVERT_RATE = 0.12;
const BT_HIGH_CONF_CONVERT_RATE = 0.07;
// The three compared ERS settings, as per-sector deploy-delta vectors (MJ) —
// the same vocabulary as the Energy Sandbox.  Balanced is the flat neutral
// shape; Push / Lift & Coast apply the calibrated spread to all three
// sectors (net ±0.15 MJ/lap).  The sector sliders add a custom 4th setting.
const ERS_SECTOR_SPREAD = 0.05;
const PRESET_SCENARIOS = [
    { key: 'lcoast', label: 'Lift & Coast', color: '#4fc3f7',
      deltas: [-ERS_SECTOR_SPREAD, -ERS_SECTOR_SPREAD, -ERS_SECTOR_SPREAD] },
    { key: 'balanced', label: 'Balanced', color: '#ffd700',
      deltas: [0, 0, 0] },
    { key: 'push', label: 'Push', color: '#ff6b6b',
      deltas: [ERS_SECTOR_SPREAD, ERS_SECTOR_SPREAD, ERS_SECTOR_SPREAD] },
];

function scenNet(sc) {
    return (sc.deltas || []).reduce(function (a, b) { return a + b; }, 0);
}
function scenIsBalanced(sc) { return Math.abs(scenNet(sc)) < 1e-9; }
function socEndOf(r) {
    const s = r.summary || {};
    return s.chaser_soc_end_pct != null ? s.chaser_soc_end_pct + '%' : '—';
}

function setGapVal(v) {
    const g = document.getElementById('ov-gap');
    if (g) g.value = v;
    const l = document.getElementById('ov-gap-val');
    if (l) l.textContent = v + 's';
    const side = document.getElementById('ov-gap-side');
    if (side) side.textContent = v + 's';
    htUpdateContext();
}

function ovSectorDeltas() {
    return ['ov-d1', 'ov-d2', 'ov-d3'].map(function (id) {
        const el = document.getElementById(id);
        return el ? (parseFloat(el.value) || 0) : 0;
    });
}
function ovSectorInput() {
    const ds = ovSectorDeltas();
    ['ov-v1', 'ov-v2', 'ov-v3'].forEach(function (id, i) {
        const el = document.getElementById(id);
        if (el) el.textContent = (ds[i] >= 0 ? '+' : '') + ds[i].toFixed(2);
    });
    const sum = ds.reduce(function (a, b) { return a + b; }, 0);
    const el = document.getElementById('ov-sum');
    if (el) {
        let net = 'pure reallocation — store-neutral, pace shape only';
        if (sum > 0.005) net = "net deploy from the chaser's 4.0 MJ store (30% floor)";
        else if (sum < -0.005) net = "net bank to the chaser's 4.0 MJ store (full = stop lifting)";
        el.textContent = 'Σ Δ = ' + (sum >= 0 ? '+' : '') + sum.toFixed(2) +
            ' MJ/lap · ' + net + " — each sector's delta worth its own s/MJ";
    }
}
function setOvErsShape(d1, d2, d3) {
    [['ov-d1', d1], ['ov-d2', d2], ['ov-d3', d3]].forEach(function (pair) {
        const el = document.getElementById(pair[0]);
        if (el) el.value = pair[1];
    });
    ovSectorInput();
}

async function runRaceCall() {
    const err = document.getElementById('ov-err');
    if (err) err.style.display = 'none';
    const out = document.getElementById('ov-call');
    const btn = document.getElementById('call-btn');
    const leader = document.getElementById('ov-leader').value;
    const chaser = document.getElementById('ov-chaser').value;
    const track = document.getElementById('ov-track').value;
    if (!leader || !chaser || !track) {
        if (err) {
            err.textContent = 'Pick the leader, chaser and race first — fastest: Calendar tab → click the round.';
            err.style.display = 'block';
        }
        return;
    }
    const battMjRaw = document.getElementById('ov-ers-batt').value;
    const battPct = battMjRaw === ''
        ? null
        : Math.max(30, Math.min(100, 100 * parseFloat(battMjRaw) / 4.0));
    const base = {
        leader_code: leader,
        chaser_code: chaser,
        track_name: track,
        year: parseInt(document.getElementById('ov-year').value, 10) || null,
        start_lap: parseInt(document.getElementById('ov-lap').value, 10) || 1,
        race_length: parseInt(document.getElementById('ov-racelaps').value, 10) || 57,
        gap_before_s: parseFloat(document.getElementById('ov-gap').value) || 0.8,
        leader_tyre_compound: document.getElementById('ov-ltyre').value,
        chaser_tyre_compound: document.getElementById('ov-ctyre').value,
        leader_tyre_age: parseInt(document.getElementById('ov-lage').value, 10) || 0,
        chaser_tyre_age: parseInt(document.getElementById('ov-cage').value, 10) || 0,
    };
    if (battPct != null) base.chaser_battery_pct = battPct;
    // The compared settings: the three preset vectors (Balanced / Push /
    // Lift & Coast) plus the slider's current sector shape when it differs
    // from every preset.
    const customDeltas = ovSectorDeltas();
    const scenarios = PRESET_SCENARIOS.map(p => ({
        key: p.key, label: p.label, color: p.color, deltas: p.deltas.slice(),
        custom: false
    }));
    const dup = PRESET_SCENARIOS.find(p =>
        p.deltas.every((d, i) => Math.abs(d - customDeltas[i]) < 1e-9));
    if (!dup) {
        const net = customDeltas.reduce((a, b) => a + b, 0);
        scenarios.push({
            key: 'custom', custom: true,
            label: Math.abs(net) < 0.005 ? 'Balanced'
                 : (net > 0 ? 'Push' : 'Lift & Coast'),
            color: net >= 0 ? '#ff6b6b' : '#4fc3f7',
            deltas: customDeltas
        });
    }

    if (btn) { btn.disabled = true; btn.textContent = 'MODELLING ' + scenarios.length + ' SETTINGS…'; }
    if (out) out.innerHTML =
        `<div class="chart-note">Running the race forward from L${base.start_lap} of ${base.race_length} under <b>Lift &amp; Coast / Balanced / Push</b>${scenarios.some(s => s.custom) ? ' + your sector shape' : ''} — leader assumed Balanced, tyre ages ticking up, no pit-stop / SC model…</div>`;
    try {
        const results = await Promise.all(scenarios.map(async sc => {
            const res = await fetch('/api/overtake/live', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(Object.assign({}, base, { chaser_ers_deltas: sc.deltas }))
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            return { scenario: sc, live: data.live };
        }));
        lastLiveCompare = results;
        renderLiveCompare(results);
    } catch (e) {
        if (err) { err.textContent = 'Race call error: ' + e.message; err.style.display = 'block'; }
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = "▲ WHAT'S THE CALL?"; }
    }
}

function livePct(p) { return Math.round(100 * Math.min(1, Math.max(0, p || 0))); }

// Pick THE CALL from the compared settings: the lightest ERS setting that
// still projects a clean pass; else best window; else conserve-to-the-flag.
// Net-zero shapes (Balanced, or a store-neutral reallocation) anchor first.
// A 'hold' verdict (the battery/posture gate) is treated as a non-conversion:
// the walk reached the target but only by draining the store to its floor or
// while the strategist asked to conserve, so it must not win the ranking.
function chooseCall(scenarios) {
    const byVerdict = (v) => scenarios.filter(x => x.live.call.verdict === v);
    const passers = byVerdict('attack').sort((a, b) => {
        const aw = scenIsBalanced(a.scenario) ? -Infinity : scenNet(a.scenario);
        const bw = scenIsBalanced(b.scenario) ? -Infinity : scenNet(b.scenario);
        return aw - bw;
    });
    if (passers.length) return { kind: 'pass', run: passers[0], passers };
    const attempts = byVerdict('attempt').sort(
        (a, b) => (b.live.summary.best_lap_probability || 0)
                - (a.live.summary.best_lap_probability || 0));
    if (attempts.length) return { kind: 'attempt', run: attempts[0], attempts };
    const holds = byVerdict('hold').sort(
        (a, b) => (b.live.summary.best_lap_probability || 0)
                - (a.live.summary.best_lap_probability || 0));
    if (holds.length) return { kind: 'hold', run: holds[0], holds };
    const balanced = scenarios.find(s => scenIsBalanced(s.scenario)) || scenarios[0];
    return { kind: 'none', run: balanced };
}

function renderLiveCompare(scenarios) {
    const out = document.getElementById('ov-call');
    if (!out) return;
    const picked = chooseCall(scenarios);
    const pick = picked.run;
    const sc = pick.scenario;
    const r = pick.live;
    const m = r.meta || {};
    const c = r.call || {};
    const s = r.summary || {};
    const lt = (m.tyres || {}).leader || {};
    const ct = (m.tyres || {}).chaser || {};
    const ersMeta = m.ers || {};
    const battIn = document.getElementById('ov-ers-batt');
    const battOverridePct = battIn && battIn.value !== ''
        ? Math.round(100 * Math.max(0, Math.min(4, parseFloat(battIn.value) || 0)) / 4.0)
        : null;
    const battStart = battOverridePct != null ? battOverridePct + '%' :
        (ersMeta.chaser_battery_start_pct != null
            ? ersMeta.chaser_battery_start_pct + '%' : '~62%');
    const net = s.avg_pace_gap_s || 0;
    const netTxt = (net >= 0 ? '+' : '') + net.toFixed(3) + 's/lap ' +
        (net >= 0 ? 'toward' : 'away');

    // ALTERNATIVES — compact "what happens if I choose this instead?" cards
    // (the recommendation itself lives in the hero above, not in this row).
    const altCards = scenarios.filter(x => x !== pick).map(x => {
        const rr = x.live;
        const cc = rr.call || {};
        const ss = rr.summary || {};
        let resTxt;
        if (cc.verdict === 'attack') {
            resTxt = `<b style="color:#ff6b6b">PASS L${cc.pass_lap}</b>`;
        } else if (cc.verdict === 'hold') {
            resTxt = `<span style="color:#ffb300">HOLD${cc.pass_lap ? ` — PASS L${cc.pass_lap} DRAINS STORE` : ' — SAVE'}</span>`;
        } else if (cc.verdict === 'attempt') {
            resTxt = `window ~L${cc.window_open_lap || '—'} · no pass`;
        } else {
            resTxt = 'no window';
        }
        const risky = (ss.ers_energy_limited_laps || 0) > 0;
        const line = (l, v) => `<div class="ac-line"><span>${l}</span><b>${v}</b></div>`;
        return `<div class="alt-card">` +
            `<div class="ac-mode" style="color:${x.scenario.color}">${x.scenario.label}${x.scenario.custom ? ' (yours)' : ''}</div>` +
            line('OVERTAKE', resTxt) +
            line('CUM P', livePct(cc.cumulative_probability) + '%') +
            line('BATTERY AT FLAG', ss.chaser_soc_end_pct != null ? socEndOf(rr) : 'n/a') +
            line('FINAL GAP', ss.projected_final_gap_s != null ? '+' + ss.projected_final_gap_s + 's' : '—') +
            line('RISK', risky
                ? '<span style="color:#ffb300">⚠ BATTERY RISK</span>'
                : (cc.verdict === 'attempt' ? '<span style="color:#ffd700">MEDIUM</span>' : '<span style="color:#00c853">LOW</span>')) +
            `</div>`;
    }).join('');

    // THE CALL narrative.
    let verdictHtml, subHtml, notes = [];
    const scNet = scenNet(sc);
    if (picked.kind === 'pass') {
        const others = scenarios.filter(x => x !== pick);
        if (scenIsBalanced(sc)) {
            const cons = scenarios.find(x => scenNet(x.scenario) < -1e-9);
            const consPasses = cons && cons.live.call.verdict === 'attack';
            const consS = cons && cons.live.summary || {};
            verdictHtml = `<div class="pass-verdict"><span>▲ OVERTAKE ON LAP</span>` +
                `<span class="pass-lap-big">${c.pass_lap}</span>` +
                `<span>— NO ERS NEEDED · run Balanced</span></div>`;
            subHtml = `Balanced alone projects the clean pass on <b>L${c.pass_lap}</b> (cum P ${livePct(c.cumulative_probability)}%) — the tyre edge does the work, no deployment required.`;
            if (consPasses) {
                subHtml += ` Even <b>${cons.scenario.label}</b> still gets there${consS.ers_banked_mj != null ? ' and banks ~' + consS.ers_banked_mj.toFixed(2) + ' MJ' : ''}, but on a thinner margin (cum P ${livePct(cons.live.call.cumulative_probability)}%) — take the pass on Balanced and keep the store.`;
            }
        } else if (scNet < -1e-9) {
            verdictHtml = `<div class="pass-verdict"><span>▲ OVERTAKE ON LAP</span>` +
                `<span class="pass-lap-big">${c.pass_lap}</span>` +
                `<span>— WHILE CONSERVING</span></div>`;
            subHtml = `<b>${sc.label}</b> still makes the pass while banking the store (${battStart} → ${socEndOf(r)}) — the cleanest outcome when it converts.`;
        } else {
            const endPct = s.chaser_soc_end_pct;
            verdictHtml = `<div class="pass-verdict"><span>▲ OVERTAKE ON LAP</span>` +
                `<span class="pass-lap-big">${c.pass_lap}</span>` +
                `<span>— MUST PUSH to make it happen</span></div>`;
            subHtml = `Balanced shows no clean pass, but <b>${sc.label}</b> deployment closes the window: expected on <b>L${c.pass_lap}</b> with the store at <b>${endPct != null ? endPct + '%' : '?'}</b> after.`;
            if (s.ers_energy_limited_laps) {
                notes.push(`⚠ The push ran the store to the 30% floor on ${s.ers_energy_limited_laps} lap(s) — the pace reverts to Balanced there, so the pass timing already assumes the edge fades.`);
            }
        }
        const waste = others.find(x => scenNet(x.scenario) > scNet);
        if (waste && scenNet(waste.scenario) > 1e-9
                && waste.live.call.verdict === 'attack') {
            notes.push(`${waste.scenario.label} buys nothing extra here (same pass, battery ${socEndOf(waste.live)}) — don't spend energy a lighter setting already earns.`);
        }
    } else if (picked.kind === 'attempt') {
        verdictHtml = `<div class="pass-verdict no-pass" style="color:#ffb300;border-color:#ffb30055">` +
            `<span>ATTACKABLE — NOT A CONVERTED PASS</span></div>`;
        subHtml = `Every setting gets the pair inside ~${LIVE_WINDOW_S}s${c.window_open_lap ? ' (from L' + c.window_open_lap + ')' : ''}, but none converts before the flag: cumulative P tops out at <b>${livePct(c.cumulative_probability)}%</b>. In the backtest a projected pass converted only ~${Math.round(100 * BT_PASS_CONVERT_RATE)}% of the time — treat this as <b>a window worth attacking, not a guaranteed move</b>, and only spend energy if the position is worth the risk.`;
        if (scNet < -1e-9) {
            notes.push(`Conserving (${sc.label}) already opens the window — banking the store while staying close is the low-risk play.`);
        }
    } else {
        verdictHtml = `<div class="pass-verdict no-pass"><span>NO CLEAN OVERTAKE</span>` +
            `<span class="pass-extra">— CONSERVE TO THE FLAG</span></div>`;
        const push = scenarios.find(x => scenNet(x.scenario) > 1e-9);
        const cons = scenarios.find(x => scenNet(x.scenario) < -1e-9);
        subHtml = `None of the settings opens a real window before the flag${scenIsBalanced(sc) ? '' : " — the gap just isn't closable on this tyre state"} (projected ${s.projected_final_gap_s}s apart at the flag, ${netTxt}).`;
        if (push) {
            const ps = push.live.summary || {};
            const pc = push.live.call || {};
            subHtml += ` Pushing changes none of that: it would spend <b>${ps.ers_deployed_mj != null ? ps.ers_deployed_mj.toFixed(2) + ' MJ' : 'energy'}</b> to still show ${pc.verdict === 'none' ? 'no window' : 'no pass'} — and the backtest's own high-confidence calls converted only ~${Math.round(100 * BT_HIGH_CONF_CONVERT_RATE)}% of the time, so there is no pass to buy here.`;
        }
        if (cons) {
            const cs = cons.live.summary || {};
            subHtml += ` The call is <b>${cons.scenario.label} to the flag</b> — bank the store (${battStart} → ${socEndOf(cons.live)}${cs.ers_banked_mj != null ? ', ~' + cs.ers_banked_mj.toFixed(2) + ' MJ banked' : ''}) for the next fight instead of burning it on a dead battle.`;
        }
    }
    if (m.extrapolation_note) {
        notes.push(`⚠ Tyres run to the flag — projected ${Math.max(s.projected_flag_tyre_ages[0], s.projected_flag_tyre_ages[1])} laps old; the model is extrapolating past realistic stint life (no pit-stop model). Re-run after each stop.`);
    }
    const noteHtml = notes.length
        ? notes.map(n => `<div class="chart-note" style="color:#ffb300;margin-top:6px">${n}</div>`).join('')
        : '';

    // Detailed facts + lap table for the recommended run.
    const laps = r.laps || [];
    const fmtDelta = (d) => (d >= 0 ? '+' : '') + d.toFixed(2);
    const ersShapeTxt = sc.deltas ? sc.deltas.map((d, i) => 'S' + (i + 1) + ' ' + fmtDelta(d)).join(' · ')
        + ` (net ${fmtDelta(scNet)} MJ/lap)` : '—';
    const facts = [
        ['BATTERY START', battStart + (battOverridePct != null
            ? ' (your input — 30% floor enforced)'
            : ' (working-band default — set it above when you have a live SOC estimate)')],
        ['CHASER ERS SHAPE', ersShapeTxt],
        ['ERS S/MJ ON TRACK', ersMeta.sec_per_mj != null ? ersMeta.sec_per_mj.toFixed(3) + ' s/MJ (per sector ' + (ersMeta.sector_pace_s_per_mj || []).map((p, i) => 'S' + (i + 1) + ' ' + p.toFixed(3)).join(' / ') + ')' : '—'],
        ['MODELS', m.models || 'aggregate career pace'],
        ['NET PACE EDGE (' + sc.label + ')', netTxt],
        ['CLOSEST APPROACH', s.closest_lap ? `${s.min_gap_s}s on L${s.closest_lap}` : '—'],
        ['BEST SINGLE-LAP P', s.best_probability_lap ? `${livePct(s.best_lap_probability)}% on L${s.best_probability_lap}` : '—'],
        ['TYRES AT THE FLAG', `${m.leader} ${lt.compound} → ${s.projected_flag_tyre_ages[0]} laps · ${m.chaser} ${ct.compound} → ${s.projected_flag_tyre_ages[1]} laps`],
    ];
    const factHtml = facts.map(f =>
        `<div style="display:flex;justify-content:space-between;gap:14px;padding:3px 0;border-bottom:1px solid #ffffff0d">` +
        `<span style="font-family:'Share Tech Mono',monospace;font-size:0.62em;letter-spacing:1px;color:#778">${f[0]}</span>` +
        `<span style="color:#ddd;text-align:right">${f[1]}</span></div>`).join('');
    const COLS = 'grid-template-columns:52px 74px 96px 1fr;';
    const rows = laps.map((l, i) => {
        const prev = i > 0 ? laps[i - 1].gap_before_s : null;
        const gapCol = prev == null ? '#ddd' : (l.gap_before_s < prev ? '#00c853' : '#ff6b6b');
        const paceTxt = (l.pace_gap_s >= 0 ? '+' : '') + l.pace_gap_s.toFixed(2);
        const winTd = l.in_window
            ? `<span>P <b style="color:${l.overtake_probability >= 0.3 ? '#ff6b6b' : '#ffd700'}">${livePct(l.overtake_probability)}%</b> · Σ ${livePct(l.cumulative_probability)}%</span>`
            : '<span style="color:#556">—</span>';
        const isPass = l.lap === c.pass_lap;
        return `<div class="tbl-row${isPass ? ' pass-row' : ''}" style="${COLS}">` +
            `<span>${isPass ? '▲ ' : ''}L${l.lap}</span>` +
            `<span style="color:${gapCol}">${l.gap_before_s.toFixed(2)}s</span>` +
            `<span style="color:${l.pace_gap_s >= 0 ? '#00c853' : '#ff6b6b'}">${paceTxt}</span>` +
            winTd + `</div>`;
    }).join('');
    const winTxt = c.window_open_lap
        ? `attack window opens L${c.window_open_lap}`
        : 'pair never gets inside ~' + LIVE_WINDOW_S + 's';

    // HERO — the call itself: mode, one-line outcome, four numbers, risk.
    const heroTag = picked.kind === 'pass'
        ? `▲ Clean overtake projected — Lap ${c.pass_lap}.`
        : picked.kind === 'hold'
        ? (c.verdict_reason
            ? `HOLD — ${c.verdict_reason}.`
            : `HOLD — the window converts but this posture drains the store to its floor; attack from a posture that keeps a reserve.`)
        : picked.kind === 'attempt'
        ? `No clean overtake — attack window opens ~L${c.window_open_lap || '—'}. Only spend energy if the position is worth the risk.`
        : 'No clean overtake predicted — protect the battery and attack later.';
    const heroRisk = (s.ers_energy_limited_laps || picked.kind === 'attempt' || picked.kind === 'hold')
        ? { t: s.ers_energy_limited_laps ? '⚠ BATTERY RISK' : (picked.kind === 'hold' ? '● HOLD — BATTERY GATE' : 'HIGH RISK'), c: '#ffb300' }
        : (scenNet(sc) > 1e-9 ? { t: 'MEDIUM RISK', c: '#ffd700' } : { t: '● LOW RISK', c: '#00c853' });
    const hstat = (label, val, col) =>
        `<div class="hc-stat"><span class="hcs-l">${label}</span><span class="hcs-v"${col ? ` style="color:${col}"` : ''}>${val}</span></div>`;
    const heroHtml =
        `<div class="hero-call">` +
        `<div class="hc-mode" style="color:${sc.color}">${sc.label}</div>` +
        `<div class="hc-tag">${heroTag}</div>` +
        `<div class="hc-stats">` +
        hstat('OVERTAKE PROBABILITY', livePct(c.cumulative_probability) + '%') +
        (picked.kind === 'pass' ? hstat('PASS LAP', 'L' + c.pass_lap) : '') +
        hstat('BATTERY AT FLAG', s.chaser_soc_end_pct != null ? socEndOf(r) : '—') +
        hstat('EXPECTED FINAL GAP', s.projected_final_gap_s != null ? '+' + s.projected_final_gap_s + 's' : '—') +
        hstat('RISK', heroRisk.t, heroRisk.c) +
        `</div></div>`;
    const altHtml = altCards
        ? `<div class="alt-head">ALTERNATIVES — WHAT HAPPENS IF I CHOOSE THIS INSTEAD?</div><div class="alt-row">${altCards}</div>`
        : '';
    out.innerHTML =
        heroHtml + altHtml +
        `<details class="ht-hint" style="margin-top:14px"><summary>ENGINEERING DETAILS — evidence under THE CALL (${sc.label}) · ${winTxt}</summary>` +
        `<div style="margin-top:10px">${verdictHtml}${subHtml}${noteHtml}</div>` +
        `<div class="call-ctx" style="margin-top:8px">${m.leader} vs ${m.chaser} · ${m.track}${m.year ? ' ' + m.year : ''} · LIVE from L${m.start_lap} of ${m.race_length} · gap ${m.gap_before_s}s · leader ${lt.compound} ${lt.age} laps (Balanced) · chaser ${ct.compound} ${ct.age} laps — one run per ERS setting</div>` +
        `<div class="panel-title" style="margin-top:10px">Under THE CALL (${sc.label}) · PACE EDGE + = chaser faster</div>` +
        `<div style="margin:10px 0;background:#101014;border:1px solid #ffffff14;border-radius:8px;padding:6px 10px">${factHtml}</div>` +
        `<div style="background:#101014;border:1px solid #ffffff14;border-radius:8px;padding:6px 10px;max-height:280px;overflow-y:auto">` +
        `<div class="tbl-row tbl-head" style="${COLS}">` +
        `<span>LAP</span><span>GAP</span><span>PACE EDGE</span><span>IN WINDOW (P → Σ)</span></div>` +
        rows +
        `</div>` +
        `<div class="chart-note" style="margin-top:8px">Each alternative = the same live race under a different chaser ERS sector shape (leader Balanced) — the deltas are worth each sector's measured s/MJ. Conversion odds are from the backtest sample; re-run scripts/backtest_race_calls.py on the full sweep to refresh them. Real races add SCs, mistakes and traffic the models don't see.</div>` +
        `</details>`;
}

async function runFullRaceSim() {
    const err = document.getElementById('ovr-err');
    err.style.display = 'none';
    const btn = [...document.querySelectorAll('button')].find(b => /SIMULATE FULL RACE/.test(b.innerText || ''));
    if (btn) { btn.disabled = true; btn.textContent = 'MODELLING RACE...'; }
    try {
        const resolved = await resolveRaceSessions();
        if (!resolved) return;
        const body = {
            full_race: true,
            leader_session_id: parseInt(document.getElementById('ovr-leader').value, 10) || 0,
            chaser_session_id: parseInt(document.getElementById('ovr-chaser').value, 10) || 0,
            start_lap: parseInt(document.getElementById('ovr-start').value, 10) || 1,
            gap_before_s: parseFloat(document.getElementById('ovr-gap').value) || 1.5,
            max_laps: parseInt(document.getElementById('ovr-maxlaps').value, 10) || 80,
        };
        const modes = ['balanced', 'push', 'liftcoast'].filter(m =>
            document.getElementById('ovr-m-' + m).checked);
        if (modes.length) body.modes = modes;
        const ovrLMode = document.getElementById('ovr-lmode').value;
        const ovrCMode = document.getElementById('ovr-cmode').value;
        if (ovrLMode && ovrCMode && ovrLMode !== ovrCMode) {
            body.mode_pairs = [[ovrLMode, ovrCMode]];
        }
        const endRaw = document.getElementById('ovr-end').value;
        if (endRaw) body.end_lap = parseInt(endRaw, 10) || undefined;
        if (!body.leader_session_id || !body.chaser_session_id) {
            err.textContent = 'Leader and chaser session IDs are required — RESOLVE SESSIONS (or type them in).';
            err.style.display = 'block';
            return;
        }
        const res = await fetch('/api/overtake/sim', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderFullRace(data.full_race);
    } catch (e) {
        err.textContent = 'Error: ' + e.message;
        err.style.display = 'block';
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'SIMULATE FULL RACE'; }
    }
}

function renderFullRace(r) {
    const out = document.getElementById('ovr-out');
    const m = r.meta || {};
    const e = r.energy || { real_laps: 0, imputed_laps: 0 };
    const passLaps = (r.laps || []).filter(l => l.passed).map(l => l.lap);
    const verdict = r.pass_lap
        ? `<div class="pass-verdict"><span>▲ Overtake on lap</span><span class="pass-lap-big">${r.pass_lap}</span>` +
          `<span>— most likely S${r.pass_sector}</span>` +
          (passLaps.length > 1 ? `<span class="pass-extra">also on laps ${passLaps.filter(l => l !== r.pass_lap).join(', ')}</span>` : '') +
          `</div>`
        : `<div class="pass-verdict no-pass">NO PASS — final gap ${r.final_gap_s}s after ${r.laps_simulated} laps</div>`;
    const energyNote = (e.real_laps + e.imputed_laps) > 0
        ? `<div class="chart-note" style="margin-top:6px;color:${e.imputed_laps ? '#ffd700' : '#00c853'}">energy_diff: real for ${e.real_laps} lap(s) · imputed ${e.imputed_laps} (run the energy simulator for these sessions to make the P0 model fully ERS-aware)</div>`
        : '';
    // PIT STOP MODEL — real stops applied as gap jumps at each one-sided
    // stop (leader stops → chaser closes; chaser stops → gap grows; a
    // leader whose gap goes negative rejoined behind = pit-stop overtake).
    const pitEvents = (r.pit_stops && r.pit_stops.length) ? r.pit_stops
        : (r.laps || []).filter(l => l.pit_stop).map(l => l.pit_stop);
    const pitNote = pitEvents.length
        ? `<div class="chart-note" style="margin-top:6px;color:#00d2be">Ⓟ PIT STOPS MODELED — ` +
          pitEvents.map(p => `<b>${p.code}</b> L${p.lap} (${p.compound || '?'}, ~${p.pit_loss_s}s${p.estimated ? ' est' : ''}${p.overtake ? ', <b style="color:#ff6b6b">position swap</b>' : ''})`).join(' · ') +
          ` · loss ≈ each driver's own slow pit lap vs neighbouring green laps; the pair's gap jumps at every one-sided stop.</div>`
        : '';
    // NEUTRALISATION MODEL — SC / VSC / RedFlag laps: pace model skipped,
    // field bunches (gap compresses toward the train), restart resumes from
    // the bunched gap.  Windows are lap-anchored into strategy_events.
    const neutralWins = r.neutralisations || [];
    const neutralGroups = [];
    neutralWins.forEach(n => {
        const g = neutralGroups[neutralGroups.length - 1];
        if (g && g.type === n.type && n.lap === g.end + 1) g.end = n.lap;
        else neutralGroups.push({ type: n.type, start: n.lap, end: n.lap });
    });
    const neutralNote = neutralGroups.length
        ? `<div class="chart-note" style="margin-top:6px;color:#4fc3f7">Ⓝ NEUTRALISATIONS MODELED — ` +
          neutralGroups.map(g => `<b>${g.type}</b> L${g.start}${g.end > g.start ? '–' + g.end : ''}`).join(' · ') +
          ` · the field bunches under the flag (the pair's gap compresses, no overtaking modeled) and the restart resumes from the bunched gap. Windows are lap-anchored into strategy_events for every later run.</div>`
        : '';

    const PERLAP_COLS = 'grid-template-columns:44px 88px 92px 44px 120px 120px 120px 1.2fr 0.8fr 150px;';
    const perlapHead =
        `<div class="tbl-row tbl-head" style="${PERLAP_COLS}">` +
        `<span>LAP</span>` +
        `<span>GAP</span>` +
        `<span>CLOSING</span>` +
        `<span>P%</span>` +
        `<span>S1</span>` +
        `<span>S2</span>` +
        `<span>S3</span>` +
        `<span>DRIVERS (TYRE·AGE)</span>` +
        `<span>HOT ZONE</span>` +
        `<span>PASS</span></div>`;
    const rows = r.laps.map(l => {
        const pct = Math.round(100 * l.overtake_probability);
        const s1 = l.sectors[0], s2 = l.sectors[1], s3 = l.sectors[2];
        const pass = l.passed
            ? `<span class="pass-chip">▲ Overtake S${l.pass_sector}</span>`
            : '';
        const gapCol = l.closing_rate_s >= 0 ? '#00c853' : '#ff6b6b';
        return `<div class="tbl-row${l.passed ? ' pass-row' : ''}" style="${PERLAP_COLS}">` +
            `<span>L${l.lap}</span>` +
            `<span>${l.gap_before_s.toFixed(2)}s</span>` +
            `<span style="color:${gapCol}">${(l.closing_rate_s > 0 ? '+' : '') + l.closing_rate_s.toFixed(2)}s</span>` +
            `<span>${pct}%</span>` +
            `<span>${sectorBarPct(100 * s1.probability, '#ffd700')} S1</span>` +
            `<span>${sectorBarPct(100 * s2.probability, '#ff9f43')} S2</span>` +
            `<span>${sectorBarPct(100 * s3.probability, '#ff6b6b')} S3</span>` +
            `<span>${l.leader.code} <span style="color:${TYRE_COLORS[l.leader.tyre] || '#00d2be'}">${l.leader.tyre}</span>${Math.round(l.leader.tyre_age)}${tyreHealthSpan(l.leader.tyre_health)} · ${l.chaser.code} <span style="color:${TYRE_COLORS[l.chaser.tyre] || '#00d2be'}">${l.chaser.tyre}</span>${Math.round(l.chaser.tyre_age)}${tyreHealthSpan(l.chaser.tyre_health)}</span>` +
            `<span style="color:#ffd700">${l.hot_zone ? '▸S' + l.hot_zone.sector + '@' + Math.round(100 * l.hot_zone.fraction) + '%' : ''}</span>` +
            `<span>${pass}${l.pit_stop ? ` <span class="pit-chip" title="Pit stop modeled: loss ~${l.pit_stop.pit_loss_s}s → gap ${l.pit_stop.overtake ? 'swap' : 'jump'}">Ⓟ ${l.pit_stop.code}${l.pit_stop.overtake ? ' · swap' : ''}</span>` : ''}${l.neutral ? ` <span class="sc-chip" title="Neutralisation: field bunches, gap compresses, no overtaking modeled">Ⓝ ${l.neutral.type}</span>` : ''}</span></div>`;
    }).join('');

    const totals = (r.sector_totals || []).map(t =>
        `S${t.sector} ${(100 * t.share).toFixed(0)}% of pass mass`).join(' · ');

    // Corner heatmap: one cell per speed-trace segment, intensity = total
    // pass-probability mass across the race; yellow border = the lap's hot
    // corner at least once (segment n covers lap fractions n/12..(n+1)/12,
    // i.e. the same track region every lap).
    const heat = r.corner_heat || [];
    const maxTot = heat.length ? Math.max.apply(null, heat.map(h => h.total_probability)) : 0;
    const heatCells = heat.map(h => {
        const a = maxTot > 0 ? (0.08 + 0.92 * h.total_probability / maxTot) : 0.08;
        const hot = h.hot_count > 0;
        return `<div title="Corner zone ${h.segment} · ${Math.round(100 * h.fraction_start)}–${Math.round(100 * h.fraction_end)}% of lap · total pass prob ${h.total_probability.toFixed(4)} · hot corner on ${h.hot_count} lap(s)" style="flex:1;height:26px;background:rgba(255,107,107,${a});border:${hot ? '1px solid #ffd700' : '1px solid #ffffff1a'};border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:9px;color:#eee">${h.segment}</div>`;
    }).join('');
    const heatStrip = heat.length
        ? `<div class="panel-title" style="margin-top:18px">Corner heatmap — braking-zone pass mass by speed-trace segment</div>` +
          `<div style="display:flex;gap:2px;margin-top:8px">${heatCells}</div>` +
          `<div class="chart-note" style="margin-top:4px">12 segments/lap (segment n = ${Math.round(100 / heat.length)}% slice of the lap, same track region every lap) · intensity = total overtake probability mass · yellow border = was a lap's hot corner · hover for values</div>`
        : '';

    // ERS sources: the stored race_state trace plus each requested mode.
    // Each source gets ONE circuit ring at the SAME size so they read as a
    // comparable row (only energy_diff changes between them).
    const mc = r.mode_compare || [];
    const modeLapsByLap = {};
    mc.forEach(m => {
        modeLapsByLap[m.mode] = {};
        (r.modes[m.mode].laps || []).forEach(l => {
            modeLapsByLap[m.mode][l.lap] = l.overtake_probability;
        });
    });
    const ringSources = [{
        mode: 'stored', color: srcColor('stored'), corner_heat: heat,
        pass_lap: r.pass_lap, final_gap_s: r.final_gap_s,
    }].concat(mc.map(m => ({
        mode: m.mode, color: srcColor(m.mode),
        corner_heat: m.corner_heat, pass_lap: m.pass_lap, final_gap_s: m.final_gap_s,
    })));
    const circuitRow = heat.length
        ? `<div class="panel-title" style="margin-top:18px">Circuit overlay — pass-mass Δ per ERS source</div>` +
          `<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px">${ringSources.map((o, i) => {
              const isStored = o.mode === 'stored';
              const ring = isStored
                  ? circuitHeatmapSvg(o.corner_heat, 150)
                  : circuitHeatmapSvg(o.corner_heat, 150, heat);
              const sub = isStored
                  ? 'stored trace (absolute)'
                  : 'Δ vs stored';
              return `<div style="text-align:center;width:150px">` +
                  `<div title="${srcTitle(o.mode)}" style="font-family:'Share Tech Mono',monospace;font-size:0.66em;color:${o.color};margin-bottom:2px">${srcLabel(o.mode)}</div>` +
                  ring +
                  `<div style="font-family:'Share Tech Mono',monospace;font-size:0.6em;color:#777;margin-top:2px">${sub} · ${o.pass_lap ? '<span class="pass-lap-marker">▲ PASS L' + o.pass_lap + '</span>' : 'no pass · ' + o.final_gap_s + 's'}</div></div>`;
          }).join('')}</div>` +
          `<div class="chart-note" style="margin-top:4px">Schematic ring: each band is one speed-trace segment in around-the-lap order (1 = start/finish, clockwise) · yellow band = a lap's hot corner · stored ring = absolute pass-probability mass (red intensity) · each other ring = Δ vs stored for that row above (green = more pass mass moved into the zone, red = less; “X &gt; Y” = asymmetric duel) · all rings the same size · hover for values</div>`
        : '';

    // ERS mode head-to-head: comparison table + overlaid per-lap P(overtake).
    let modeSection = '';
    if (mc.length) {
        const MODE_COLS = 'grid-template-columns:88px 108px 104px 104px 112px 118px 1fr;';
        const modeHead =
            `<div class="tbl-row tbl-head" style="${MODE_COLS}">` +
            `<span>MODE</span>` +
            `<span>MEAN P</span>` +
            `<span>MAX P</span>` +
            `<span>DEPLOY</span>` +
            `<span>BATT END</span>` +
            `<span>ΔE</span>` +
            `<span>RESULT</span></div>`;
        const rowsHtml = modeHead + mc.map(m => {
            const c = srcColor(m.mode);
            const pass = m.pass_lap ? `<span class="pass-chip">▲ Overtake L${m.pass_lap}</span>` : `<span style="color:#777">no pass (final ${m.final_gap_s}s)</span>`;
            const asym = !m.symmetric && (m.leader_mode || m.chaser_mode);
            const dep = asym
                ? `L ${fmtSpecMJ(m.leader_deployed_mj)} · C ${fmtSpecMJ(m.chaser_deployed_mj)}`
                : `${fmtSpecMJ(m.deployed_total_mj)}`;
            const batt = asym
                ? `L ${fmtSpecPct(m.leader_final_battery_pct)} · C ${fmtSpecPct(m.chaser_final_battery_pct)}`
                : `${m.final_battery_pct != null ? m.final_battery_pct.toFixed(1) + '%' : '—'}`;
            const clip = (m.energy_clipped_laps || 0) > 0
                ? `<div style="color:#ffb300;font-size:0.95em" title="energy_diff beyond the P0 models' trained range was capped at [-2.0, +0.6] MJ on these laps">ΔE capped ${m.energy_clipped_laps} lap(s)</div>`
                : '';
            return `<div class="tbl-row${m.pass_lap ? ' pass-row' : ''}" style="${MODE_COLS}">` +
                `<span style="color:${c}" title="${srcTitle(m.mode)}"><b>${srcLabel(m.mode)}</b></span>` +
                `<span>${m.mean_probability.toFixed(4)}</span>` +
                `<span>${m.max_probability.toFixed(4)}</span>` +
                `<span title="total energy deployed over the race (real = the stored trace's actual race_state values)">${dep}</span>` +
                `<span title="battery at the flag per driver">${batt}</span>` +
                `<span>${m.energy_diff_mean_mj > 0 ? '+' : ''}${m.energy_diff_mean_mj.toFixed(3)} MJ</span>` +
                `<span>${pass}${clip}</span></div>`;
        }).join('');
        modeSection =
            `<div class="panel-title" style="margin-top:18px">ERS head-to-head — symmetric modes &amp; asymmetric LEADER &gt; CHASER duels</div>` +
            `<div class="tbl-wrap">${rowsHtml}</div>` +
            `<div style="position:relative;height:200px;margin-top:10px"><canvas id="ovrModeChartCanvas"></canvas></div>` +
            `<div class="chart-note" style="margin-top:4px">Each row is one full-race pass where ONLY the energy_diff feature changed. Plain rows = both cars on that mode; “X &gt; Y” = leader on X, chaser on Y (their per-lap P(overtake) lines are overlaid above) · ΔE = mean (chaser − leader) battery each lap · asymmetric rows show each car's deployed MJ / end battery · “real” = that side used its actual race_state trace · energy_diff beyond the models' trained range is capped at [−2.0, +0.6] MJ (laps flagged on the row)</div>`;
    }

    // P4 projected finishing order: per-ERS-source classification.  The
    // projected times are anchored on the winner's real race total with the
    // sim's final modeled gap applied (margin ≡ final gap), so the card is
    // consistent with the gap chart; real totals + modeled gained seconds
    // stay visible as evidence columns.
    const lb = r.leaderboard;
    let leaderboardSection = '';
    let calibSection = '';
    if (lb) {
        const fmtTime = s => {
            const neg = s < 0;
            const x = Math.abs(s);
            const h = Math.floor(x / 3600), mn = Math.floor((x % 3600) / 60), sc = x % 60;
            return (neg ? '-' : '') + h + ':' + String(mn).padStart(2, '0') +
                ':' + sc.toFixed(3).padStart(6, '0');
        };
        const scored = lb.sources.stored ? lb.sources.stored.laps_scored : r.laps_simulated;
        const P4_COLS = 'grid-template-columns:88px 190px 150px 120px 130px 1fr 90px;';
        const p4Head =
            `<div class="tbl-row tbl-head" style="${P4_COLS}">` +
            `<span>SOURCE</span>` +
            `<span>P1</span>` +
            `<span>P2</span>` +
            `<span>SWAP</span>` +
            `<span>PASS</span>` +
            `<span>GAINED</span>` +
            `<span>TRUST</span></div>`;
        const lbRows = p4Head + (lb.order || []).map(src => {
            const s = lb.sources[src];
            if (!s || !s.order || s.order.length < 2) return '';
            const color = srcColor(src);
            const p1 = s.order[0], p2 = s.order[1];
            const swapped = p1.code !== lb.initial_leader;
            const passTxt = s.pass_lap
                ? `<span class="pass-chip">▲ Overtake L${s.pass_lap}</span>`
                : `<span style="color:#666">no pass</span>`;
            return `<div class="tbl-row" style="${P4_COLS}">` +
                `<span style="color:${color}" title="${srcTitle(src)}"><b>${srcLabel(src)}</b></span>` +
                `<span>P1 <b style="color:#ffd700">${p1.code}</b> <span style="color:#aaa">${fmtTime(p1.projected_s)}</span></span>` +
                `<span>P2 ${p2.code} <span style="color:#aaa">+${s.margin_s.toFixed(3)}s</span></span>` +
                `<span>${swapped ? '<b style="color:#ff6b6b">⬆ SWAP</b>' : '<span style="color:#666">holds</span>'}</span>` +
                `<span>${passTxt}</span>` +
                `<span>${p1.code} ${p1.gained_s.toFixed(2)}s · ${p2.code} ${p2.gained_s.toFixed(2)}s</span>` +
                `<span>${trustDot(s.trust)}</span></div>`;
        }).join('');
        leaderboardSection =
            `<div class="panel-title" style="margin-top:18px">Projected finishing order (P4) — per ERS source</div>` +
            `<div class="tbl-wrap">${lbRows}</div>` +
            `<div class="chart-note" style="margin-top:4px">Times anchored on the winner's real race total over ${scored} scored laps + the sim's final modeled gap (margin ≡ final gap) · gained = seconds the model had that driver gain while chasing · ⬆ SWAP = the mode's deployment flips the classification · ● = calibration trust (below)</div>`;

        // Model-vs-reality calibration: the P0 closing-rate predictions
        // regressed against the drivers' REAL lap-time deltas on these laps.
        // agreement % = how often the model's sign matches reality; flips =
        // laps where the model says the chaser gains but reality shows the
        // chaser losing (or vice versa); net direction ⚠ = the model's
        // overall verdict contradicts the real lap evidence, so the
        // projected leaderboard above is a scenario, not a prediction.
        const calRows = (lb.order || []).map(src => {
            const s = lb.sources[src];
            const c = s && s.calibration;
            if (!c || !c.n) return '';
            const color = srcColor(src);
            const flips = (c.flip_laps || []).slice(0, 8).map(l => 'L' + l).join(', ') +
                ((c.flip_laps || []).length > 8 ? ` +${c.flip_laps.length - 8} more` : '');
            const netTxt = c.net_direction_ok
                ? `<span style="color:#666">net model ${c.net_model_gain_s.toFixed(1)}s vs real ${c.net_actual_gain_s.toFixed(1)}s</span>`
                : `<span style="color:#ff6b6b"><b>⚠ net direction contradicts real laps</b></span>`;
            return `<div style="display:flex;align-items:center;gap:10px;padding:4px 0;border-bottom:1px solid #ffffff0d;font-family:'Share Tech Mono',monospace;font-size:0.68em;flex-wrap:wrap">` +
                `<span style="width:88px;color:${color}" title="${srcTitle(src)}"><b>${srcLabel(src)}</b></span>` +
                trustDot(c.trust) +
                `<span style="width:58px">n=${c.n}</span>` +
                `<span style="width:88px">agree ${Math.round(100 * c.agreement_pct)}%</span>` +
                `<span style="width:74px">R² ${c.r2.toFixed(2)}</span>` +
                `<span style="width:80px">slope ${c.slope.toFixed(2)}</span>` +
                `<span style="width:150px">flips ${c.sign_flips}${flips ? ' (' + flips + ')' : ''}</span>` +
                `${netTxt}</div>`;
        }).join('');
        const calibHead =
            `<div style="display:flex;align-items:center;gap:10px;padding:3px 0;border-bottom:1px solid #ffffff1f;font-family:'Share Tech Mono',monospace;font-size:0.6em;color:#999;letter-spacing:1px;flex-wrap:wrap">` +
            `<span style="width:88px">ERS SRC</span>` +
            `<span style="width:74px">TRUST</span>` +
            `<span style="width:58px">LAPS</span>` +
            `<span style="width:88px">AGREE</span>` +
            `<span style="width:74px">R²</span>` +
            `<span style="width:80px">SLOPE</span>` +
            `<span style="width:150px">FLIPS</span>` +
            `<span>NET (model vs real)</span></div>`;
        calibSection =
            `<div class="panel-title" style="margin-top:18px">Model calibration — closing rate vs real lap-time deltas</div>` +
            `<div style="background:#101014;border:1px solid #ffffff14;border-radius:8px;padding:8px 10px">${calibHead}${calRows || '<div class="chart-note">No scored laps to calibrate against.</div>'}</div>` +
            `<div class="chart-note" style="margin-top:4px">Regresses each lap's predicted closing rate against the drivers' actual lap-time deltas (leader lap − chaser lap). agree = model sign matches reality · flips = sign-flip laps · slope 1 = scale-true (lower = model overstates, negative = inverted) · ●high needs ≥65% agreement, R² ≥ 0.25 and matching net direction</div>`;
    }

    out.innerHTML =
        `<div class="strat-recommend" style="margin-top:14px;display:block">` +
        `<div class="strat-rec-title">${m.leader ? m.leader.code : '?'} (#${m.leader ? m.leader.session_id : '?'}) vs ${m.chaser ? m.chaser.code : '?'} (#${m.chaser ? m.chaser.session_id : '?'}) — ${m.track || ''} ${m.date || ''} · laps ${m.start_lap}–${m.end_lap}</div>` +
        `<div class="strat-rec-reason">sector totals: ${totals || 'n/a'} · final gap ${r.final_gap_s}s over ${r.laps_simulated} lap(s)</div>` +
        `</div>` +
        `${verdict}${energyNote}${pitNote}${neutralNote}` +
        `<div class="panel-title" style="margin-top:18px">Gap progression (speed-trace aligned intra-lap path in tooltip)</div>` +
        `<div style="position:relative;height:220px;margin-top:8px"><canvas id="ovrGapChartCanvas"></canvas></div>` +
        `<div class="panel-title" style="margin-top:18px">Overtake % by sector — stacked per lap</div>` +
        `<div style="position:relative;height:220px;margin-top:8px"><canvas id="ovrSectorChartCanvas"></canvas></div>` +
        `<details class="ht-hint" style="margin-top:14px"><summary>ENGINEERING DETAILS — ERS &amp; track evidence</summary>` +
        `${heatStrip}${circuitRow}${modeSection}${leaderboardSection}${calibSection}` +
        `</details>` +
        `<details class="ht-hint" style="margin-top:10px"><summary>VIEW LAP DATA — ${r.laps.length} laps</summary>` +
        `<div class="panel-title" style="margin-top:10px">Per-lap <span class="pass-lap-marker" style="font-size:0.55em">red rows = overtake laps</span></div>` +
        `<div class="tbl-wrap">${perlapHead}${rows}</div></details>`;

    const laps = r.laps.map(l => l.lap);
    // Pass laps shared by the gap / sector charts: dashed "OVERTAKE L{n}"
    // line plus an emphasised data point so the lap jumps out.
    const passLapsMain = [];
    r.laps.forEach((l, i) => { if (l.passed) passLapsMain.push({ idx: i, lap: l.lap }); });
    const gapMarks = passLapsMain.map(m => ({ idx: m.idx, lap: m.lap, color: '#ff6b6b', label: 'OVERTAKE L' + m.lap }));
    const pointBg = r.laps.map(l => (l.passed ? '#ff6b6b'
        : (l.neutral ? '#4fc3f7' : 'rgba(255,215,0,0.35)')));
    const pointR = r.laps.map(l => (l.passed ? 6 : (l.neutral ? 4 : 2)));
    const pointBw = r.laps.map(l => (l.passed ? 2 : 0));
    if (ovrGapChart) { ovrGapChart.destroy(); ovrGapChart = null; }
    if (ovrSectorChart) { ovrSectorChart.destroy(); ovrSectorChart = null; }
    // Keep the result for CSV/JSON export and enable the export buttons.
    lastRaceSim = r;
    const csvBtn = document.getElementById('ovr-export-csv');
    const jsonBtn = document.getElementById('ovr-export-json');
    if (csvBtn) csvBtn.disabled = false;
    if (jsonBtn) jsonBtn.disabled = false;

    const gctx = document.getElementById('ovrGapChartCanvas');
    if (gctx) {
        ovrGapChart = new Chart(gctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: laps,
                datasets: [{
                    label: 'gap delta (s)',
                    data: r.laps.map(l => l.gap_before_s),
                    borderColor: '#ffd700', backgroundColor: 'rgba(255,215,0,0.08)',
                    fill: true, tension: 0.2,
                    pointRadius: pointR, pointBackgroundColor: pointBg,
                    pointBorderColor: r.laps.map(l => (l.passed ? '#fff' : 'transparent')),
                    pointBorderWidth: pointBw, pointHoverRadius: 7,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    y: { title: { display: true, text: 'gap (s)' }, grid: { color: '#ffffff12' } },
                    x: { title: { display: true, text: 'lap' }, grid: { color: '#ffffff12' } },
                },
                plugins: { legend: { labels: { color: '#ccc', font: { size: 10 } } } },
            },
            plugins: [passLapAnnotationPlugin(gapMarks)],
        });
    }
    const sctx = document.getElementById('ovrSectorChartCanvas');
    if (sctx) {
        ovrSectorChart = new Chart(sctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: laps,
                datasets: [
                    { label: 'S1', data: r.laps.map(l => l.sectors[0].probability), backgroundColor: '#ffd700' },
                    { label: 'S2', data: r.laps.map(l => l.sectors[1].probability), backgroundColor: '#ff9f43' },
                    { label: 'S3', data: r.laps.map(l => l.sectors[2].probability), backgroundColor: '#ff6b6b' },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { stacked: true, title: { display: true, text: 'lap' }, grid: { color: '#ffffff12' } },
                    y: { stacked: true, title: { display: true, text: 'P(overtake)' }, grid: { color: '#ffffff12' } },
                },
                plugins: { legend: { labels: { color: '#ccc', font: { size: 10 } } } },
            },
            plugins: [passLapAnnotationPlugin(gapMarks)],
        });
    }
    // ERS mode head-to-head: overlaid per-lap P(overtake) per mode + stored.
    const mctx = document.getElementById('ovrModeChartCanvas');
    if (mctx) {
        if (ovrModeChart) { ovrModeChart.destroy(); ovrModeChart = null; }
        const datasets = [{
            label: 'stored (race_state)', data: r.laps.map(l => l.overtake_probability),
            borderColor: OVR_MODE_COLORS.stored, borderWidth: 2, pointRadius: 0, tension: 0.2, fill: false,
        }];
        (r.mode_compare || []).forEach(m => {
            datasets.push({
                label: srcLabel(m.mode), data: r.laps.map(l => (modeLapsByLap[m.mode] || {})[l.lap] ?? null),
                borderColor: srcColor(m.mode), borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: false,
            });
        });
        // Annotate each source's own pass lap in that source's colour so the
        // chart shows WHICH deployment overtakes WHEN.
        const modePassMarks = [];
        const pushMark = (lap, di, color, label) => {
            const idx = laps.indexOf(lap);
            if (idx >= 0) modePassMarks.push({ idx: idx, lap: lap, datasetIndex: di, color: color, label: label });
        };
        if (r.pass_lap) pushMark(r.pass_lap, 0, '#ff6b6b', 'OVERTAKE L' + r.pass_lap);
        (r.mode_compare || []).forEach((m, mi) => {
            if (m.pass_lap) pushMark(m.pass_lap, mi + 1, srcColor(m.mode), srcLabel(m.mode) + ' PASS L' + m.pass_lap);
        });
        ovrModeChart = new Chart(mctx.getContext('2d'), {
            type: 'line',
            data: { labels: laps, datasets: datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    y: { title: { display: true, text: 'P(overtake)' }, grid: { color: '#ffffff12' } },
                    x: { title: { display: true, text: 'lap' }, grid: { color: '#ffffff12' } },
                },
                plugins: { legend: { labels: { color: '#ccc', font: { size: 10 } } } },
            },
            plugins: [passLapAnnotationPlugin(modePassMarks)],
        });
    }
}

function downloadFile(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 500);
}

function exportRaceCsv() {
    const r = lastRaceSim;
    if (!r) return;
    const m = r.meta || {};
    const header = ['lap', 'gap_before_s', 'gap_after_s', 'closing_rate_s', 'pace_gap_s',
                    'overtake_probability', 's1_probability', 's2_probability', 's3_probability',
                    'energy_diff_mj', 'fuel_diff_kg', 'leader', 'leader_tyre', 'leader_tyre_age', 'leader_tyre_health',                    'chaser', 'chaser_tyre', 'chaser_tyre_age', 'chaser_tyre_health',
                    'passed', 'hot_zone', 'pit_stop', 'neutral'];
    const esc = v => {
        if (v === null || v === undefined) return '';
        const s = String(v);
        return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    const rows = r.laps.map(l => [
        l.lap, l.gap_before_s, l.gap_after_s, l.closing_rate_s, l.pace_gap_s,
        l.overtake_probability, l.sectors[0].probability, l.sectors[1].probability, l.sectors[2].probability,
        l.energy_diff_mj, l.fuel_diff_kg,
        l.leader.code, l.leader.tyre, l.leader.tyre_age, l.leader.tyre_health,
        l.chaser.code, l.chaser.tyre, l.chaser.tyre_age, l.chaser.tyre_health,
        l.passed ? 1 : 0,
        l.hot_zone ? `S${l.hot_zone.sector}@${Math.round(100 * l.hot_zone.fraction)}%` : '',
        l.pit_stop ? `${l.pit_stop.code} ${l.pit_stop.compound || ''} loss~${l.pit_stop.pit_loss_s}s${l.pit_stop.overtake ? ' swap' : ''}` : '',
        l.neutral ? l.neutral.type : ''
    ]);
    let csv = [header.join(','), ...rows.map(rr => rr.map(esc).join(','))].join('\n');
    // P4: append the projected finishing order (per ERS source) to the export.
    if (r.leaderboard) {
        const lb = r.leaderboard;
        const lines = ['', 'PROJECTED FINISHING ORDER (per ERS source)',
            'source,position,code,projected_s,real_s,gained_s,margin_s,pass_lap'];
        (lb.order || []).forEach(src => {
            const s = lb.sources[src];
            if (!s) return;
            (s.order || []).forEach(row => {
                lines.push([src, row.position, row.code, row.projected_s,
                    row.real_s, row.gained_s, s.margin_s,
                    s.pass_lap || ''].map(esc).join(','));
            });
        });
        csv += '\n' + lines.join('\n');
    }
    // Model calibration: closing-rate regression stats per ERS source.
    if (r.leaderboard && r.leaderboard.order) {
        const calLines = ['', 'MODEL CALIBRATION (closing rate vs real lap-time deltas)',
            'source,n,trust,agreement_pct,pearson,r2,slope,intercept,net_model_gain_s,net_actual_gain_s,net_direction_ok,sign_flips,flip_laps'];
        r.leaderboard.order.forEach(src => {
            const s = r.leaderboard.sources[src];
            const c = s && s.calibration;
            if (!c || !c.n) return;
            calLines.push([src, c.n, c.trust, c.agreement_pct, c.pearson, c.r2,
                c.slope, c.intercept, c.net_model_gain_s, c.net_actual_gain_s,
                c.net_direction_ok ? 1 : 0, c.sign_flips,
                (c.flip_laps || []).join(';')].map(esc).join(','));
        });
        csv += '\n' + calLines.join('\n');
    }
    const base = `${m.leader ? m.leader.code : 'x'}_vs_${m.chaser ? m.chaser.code : 'y'}_${(m.track || '').replace(/[^A-Za-z0-9]+/g, '_')}`;
    downloadFile(`${base}_race.csv`, csv, 'text/csv');
}

function exportRaceJson() {
    const r = lastRaceSim;
    if (!r) return;
    const m = r.meta || {};
    const base = `${m.leader ? m.leader.code : 'x'}_vs_${m.chaser ? m.chaser.code : 'y'}_${(m.track || '').replace(/[^A-Za-z0-9]+/g, '_')}`;
    downloadFile(`${base}_race.json`, JSON.stringify(r, null, 2), 'application/json');
}

// Auto-refresh: when the energy simulator rewrites race_state for one of the
// displayed race's sessions, re-run the full race so the energy_diff counter
// (real vs imputed) drops live without a page reload.

// ── VALIDATION — energy benchmark (deck's −1.8s claim + every stored race) ──
// Serves the fleet summary (backtests/energy_fleet_summary.json, written by
// scripts/benchmark_energy_fleet.py) plus full per-race artifacts, and
// re-runs the selected race's benchmark on demand.  Monaco 2023 is the deck
// anchor: its artifact carries the deck-claim metadata, so the reproduction
// badge shows exactly there and nowhere else.
let bmEnergyChart = null;
let bmFleetData = null;      // last fleet summary payload
let bmCurrentRace = null;    // {key, track, year} of the selected race

const bmEsc = v => String(v == null ? '' : v).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const bmNormTrack = s => String(s || '').trim().toLowerCase();

// INIT — first open of the Validation module (wired in htToggle).
async function loadEnergyBenchmark() {
    const out = document.getElementById('bm-energy-out');
    const err = document.getElementById('bm-energy-err');
    if (err) err.style.display = 'none';
    if (out) out.innerHTML = '<div class="chart-note">Loading fleet summary…</div>';
    try {
        const res = await fetch('/api/benchmark/energy/fleet');
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        bmFleetData = data;
        renderFleetTable(data);
        populateRaceSelects(data.races || []);
    } catch (e) {
        if (out) out.innerHTML = '';
        if (err) { err.textContent = 'Error: ' + e.message; err.style.display = 'block'; }
    }
}

function populateRaceSelects(races) {
    const ts = document.getElementById('bm-track'), ys = document.getElementById('bm-year');
    if (!ts || !ys) return;
    const tracks = [...new Set(races.map(r => r.track))].sort((a, b) => a.localeCompare(b));
    ts.innerHTML = tracks.map(t => `<option value="${bmEsc(t)}">${bmEsc(t)}</option>`).join('');
    // Default to Monaco (the deck anchor), else the alphabetically first.
    ts.value = tracks.find(t => bmNormTrack(t) === 'monaco') || tracks[0];
    bmYearOptions();
}

function bmYearOptions() {
    const ts = document.getElementById('bm-track'), ys = document.getElementById('bm-year');
    if (!ts || !ys || !bmFleetData) return;
    const track = ts.value;
    const years = [...new Set((bmFleetData.races || []).filter(r => r.track === track).map(r => r.year))].sort();
    ys.innerHTML = years.map(y => `<option value="${bmEsc(y)}">${bmEsc(y)}</option>`).join('');
    ys.value = (bmNormTrack(track) === 'monaco' && years.includes('2023')) ? '2023' : (years[years.length - 1] || '');
    bmSelectRace();
}

async function bmSelectRace() {
    const ts = document.getElementById('bm-track'), ys = document.getElementById('bm-year');
    if (!ts || !ys || !bmFleetData || !ts.value || !ys.value) return;
    const races = (bmFleetData.races || []).filter(r => r.track === ts.value && r.year === ys.value);
    if (!races.length) return;
    const race = races[races.length - 1];   // latest date within the season
    bmCurrentRace = { key: race.key, track: race.track, year: race.year };
    const label = document.getElementById('bm-race-label');
    if (label) label.textContent = race.track + ' · ' + race.date + ' · ' + race.sessions + ' drivers';
    await loadEnergyRace(race.key);
}

async function loadEnergyRace(key) {
    const out = document.getElementById('bm-energy-out');
    const err = document.getElementById('bm-energy-err');
    if (err) err.style.display = 'none';
    if (out) out.innerHTML = '<div class="chart-note">Loading race benchmark…</div>';
    try {
        const res = await fetch('/api/benchmark/energy/race?key=' + encodeURIComponent(key));
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderEnergyBenchmark(data);
    } catch (e) {
        if (out) out.innerHTML = '';
        if (err) { err.textContent = 'Error: ' + e.message; err.style.display = 'block'; }
    }
}

async function rerunEnergyBenchmark(btn) {
    const err = document.getElementById('bm-energy-err');
    if (err) err.style.display = 'none';
    const q = bmCurrentRace
        ? `?track=${encodeURIComponent(bmCurrentRace.track)}&year=${bmCurrentRace.year}` : '';
    if (btn) { btn.disabled = true; btn.textContent = '⟳ RE-RUNNING…'; }
    try {
        const res = await fetch('/api/benchmark/energy/run' + q, { method: 'POST' });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderEnergyBenchmark(data);
    } catch (e) {
        const out = document.getElementById('bm-energy-out');
        if (out) out.innerHTML = '';
        if (err) { err.textContent = 'Error: ' + e.message; err.style.display = 'block'; }
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '⟳ RE-RUN BENCHMARK'; }
    }
}

async function bmReload() {
    if (bmCurrentRace) await loadEnergyRace(bmCurrentRace.key);
}

async function rebuildFleetSummary(btn) {
    const err = document.getElementById('bm-energy-err');
    if (err) err.style.display = 'none';
    if (btn) { btn.disabled = true; btn.textContent = '⟳ REBUILDING…'; }
    try {
        const res = await fetch('/api/benchmark/energy/fleet/run', { method: 'POST' });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        const sel = {
            track: (document.getElementById('bm-track') || {}).value,
            year: (document.getElementById('bm-year') || {}).value,
        };
        bmFleetData = data;
        renderFleetTable(data);
        populateRaceSelects(data.races || []);
        const ts = document.getElementById('bm-track'), ys = document.getElementById('bm-year');
        if (sel.track && ts && [...ts.options].some(o => o.value === sel.track)) {
            ts.value = sel.track;
            const years = [...new Set((bmFleetData.races || []).filter(r => r.track === sel.track).map(r => r.year))].sort();
            ys.innerHTML = years.map(y => `<option value="${bmEsc(y)}">${bmEsc(y)}</option>`).join('');
            ys.value = (sel.year && years.includes(sel.year)) ? sel.year : (years[years.length - 1] || '');
            bmSelectRace();
        }
    } catch (e) {
        const out = document.getElementById('bm-energy-out');
        if (out) out.innerHTML = '';
        if (err) { err.textContent = 'Error: ' + e.message; err.style.display = 'block'; }
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '⟳ REBUILD FLEET SUMMARY'; }
    }
}

function renderFleetTable(data) {
    const out = document.getElementById('bm-fleet-out');
    if (!out) return;
    const races = data.races || [];
    const head = data.headline || {};
    if (!races.length) {
        out.innerHTML = '<div class="chart-note">No races in the fleet summary.</div>';
        return;
    }
    const cols = 'grid-template-columns:minmax(140px,1fr) 56px 40px 48px 74px 84px 84px 90px;';
    const rows = races.map((r, i) => {
        const top = i < 5;
        const basis = r.pace_basis && r.pace_basis.indexOf('flat') === 0 ? 'F' : 'M';
        return `<div class="tbl-row" style="${cols}${top ? 'border-left:3px solid #ffd700;' : ''}">` +
            `<span title="${bmEsc(r.track + ' · ' + r.date + ' · ' + r.sessions + ' drivers')}">${bmEsc(r.track)}</span>` +
            `<span>${bmEsc(r.year)}</span>` +
            `<span>${r.sessions}</span>` +
            `<span>${r.laps}</span>` +
            `<span title="pace basis: ${bmEsc(r.pace_basis)}">${Number(r.pace_s_per_mj).toFixed(4)}<span style="color:#667"> ${basis}</span></span>` +
            `<span style="color:${r.closing_phase_improvement_s_mean < 0 ? '#00c853' : '#ff6b6b'}">${Number(r.closing_phase_improvement_s_mean).toFixed(2)}s</span>` +
            `<span style="color:${r.full_race_improvement_s_mean < 0 ? '#00c853' : '#ff6b6b'}">${Number(r.full_race_improvement_s_mean).toFixed(2)}s</span>` +
            `<span>${bmEsc(r.strategy_mode)}</span></div>`;
    }).join('');
    out.innerHTML =
        `<div class="hc-stats" style="gap:26px;margin-bottom:10px">` +
        `<div class="hc-stat"><span class="hcs-l">RACES</span><span class="hcs-v">${races.length}</span></div>` +
        `<div class="hc-stat"><span class="hcs-l">CLOSING MEAN</span><span class="hcs-v" style="color:#00c853">${Number(head.closing_phase_improvement_s_mean).toFixed(2)}s</span></div>` +
        `<div class="hc-stat"><span class="hcs-l">FULL-RACE MEAN</span><span class="hcs-v" style="color:#00c853">${Number(head.full_race_improvement_s_mean).toFixed(2)}s</span></div>` +
        `</div>` +
        `<div class="tbl-wrap"><div class="tbl-row tbl-head" style="${cols}">` +
        `<span>TRACK</span><span>YEAR</span><span>N</span><span>LAPS</span><span>PACE</span><span>CLOSING</span><span>FULL RACE</span><span>MODE</span></div>${rows}</div>` +
        `<div class="chart-note" style="margin-top:8px">Sorted by closing-phase gain (biggest first, gold edge = top 5). PACE suffix M = measured per-track s/MJ · F = flat-constant fallback (track not in the measured profile). Click a row to load that race.</div>`;
    // Row click -> select that race in the pickers.
    out.querySelectorAll('.tbl-row:not(.tbl-head)').forEach((row, i) => {
        row.style.cursor = 'pointer';
        row.addEventListener('click', () => {
            const r = races[i];
            const ts = document.getElementById('bm-track'), ys = document.getElementById('bm-year');
            if (!ts || !ys) return;
            ts.value = r.track;
            const years = [...new Set((bmFleetData.races || []).filter(x => x.track === r.track).map(x => x.year))].sort();
            ys.innerHTML = years.map(y => `<option value="${bmEsc(y)}">${bmEsc(y)}</option>`).join('');
            ys.value = r.year;
            bmSelectRace();
        });
    });
}

function renderEnergyBenchmark(data) {
    const out = document.getElementById('bm-energy-out');
    const err = document.getElementById('bm-energy-err');
    const meta = document.getElementById('bm-energy-meta');
    if (!out) return;
    if (err) err.style.display = 'none';

    const head = data.headline || {};
    const cfg = data.config || {};
    const sessions = data.sessions || [];
    const src = data._source || {};
    const closeMean = head.closing_phase_improvement_s_mean;
    const fullMean = head.full_race_improvement_s_mean;
    const hasDeck = head.deck_claim_s !== undefined;
    const reproduced = !!head.deck_claim_reproduced;
    const genAt = (data.meta && data.meta.generated_at) || '';

    const stat = (label, value, sub, color) =>
        `<div class="hc-stat"><span class="hcs-l">${label}</span><span class="hcs-v"${color ? ` style="color:${color}"` : ''}>${value}</span>` +
        (sub ? `<span style="font-size:0.6em;color:#667">${sub}</span>` : '') + `</div>`;

    // Closing-window tyre health (backend-annotated from the same model the
    // Tyre Degradation chart plots) tints each driver's row: green = plenty
    // of tyre life left, yellow = worn, red = tyres essentially gone in the
    // closing phase (the projection is extrapolating past realistic stint
    // life).  Grey = no tyre data for that driver.
    const cols = 'grid-template-columns:64px 56px 96px 120px 120px 1fr 190px;';
    const rows = sessions.map(s => {
        const h = s.closing_tyre_health_pct;
        const tint = h == null ? null
            : h >= 60 ? 'rgba(0,200,83,0.10)'
            : h >= 30 ? 'rgba(255,215,0,0.10)'
            : 'rgba(255,107,107,0.14)';
        const edge = h == null ? ''
            : `border-left:3px solid ${h >= 60 ? '#00c853' : h >= 30 ? '#ffd700' : '#ff6b6b'};`;
        const healthCell = h == null
            ? '<span style="color:#667">—</span>'
            : `<span style="color:${h >= 60 ? '#00c853' : h >= 30 ? '#ffd700' : '#ff6b6b'}" ` +
              `title="tyre health at the closing-phase start (L${s.closing_phase_start_lap ?? ''}) — same model as the Tyre Degradation chart">` +
              `${Math.round(h)}% ${bmEsc(s.closing_tyre_compound || '')}${s.closing_tyre_age != null ? '·age' + Math.round(s.closing_tyre_age) : ''}</span>`;
        return `<div class="tbl-row" style="${cols}${tint ? 'background:' + tint + ';' : ''}${edge}">` +
            `<span>${s.driver_code}</span>` +
            `<span>L${s.laps}</span>` +
            `<span>${Number(s.pace_s_per_mj).toFixed(4)}</span>` +
            `<span style="color:#00c853">${Number(s.closing_phase_improvement_s).toFixed(2)}s</span>` +
            `<span style="color:#00c853">${Number(s.full_race_improvement_s).toFixed(2)}s</span>` +
            `<span>${s.strategy_mode} vs ${s.flat_out_mode}</span>` +
            healthCell + `</div>`;
    }).join('');

    let html =
        `<div class="hc-stats" style="gap:34px;margin-bottom:14px">` +
        stat('CLOSING PHASE · final ' + (cfg.closing_laps || 15) + ' laps',
            (closeMean >= 0 ? '+' : '') + closeMean.toFixed(2) + 's',
            'mean across ' + sessions.length + ' drivers' + (hasDeck ? ' · deck claims −1.8s' : ''),
            closeMean < 0 ? '#00c853' : '#ff6b6b') +
        stat('FULL RACE',
            (fullMean >= 0 ? '+' : '') + fullMean.toFixed(2) + 's',
            hasDeck ? 'strategy − flat-out · deck number is conservative' : 'strategy − flat-out',
            fullMean < 0 ? '#00c853' : '#ff6b6b') +
        (hasDeck
            ? stat('DECK CLAIM',
                reproduced ? 'REPRODUCED ✓' : 'MISSED ✗',
                '|−1.8 − closing mean| < 0.25s',
                reproduced ? '#ffd700' : '#ff6b6b')
            : '') +
        `</div>` +
        `<div class="tbl-wrap"><div class="tbl-row tbl-head" style="${cols}">` +
        `<span>DRIVER</span><span>LAPS</span><span>PACE</span><span>CLOSING</span><span>FULL RACE</span><span>PAIRING</span><span>TYRE HEALTH @ CLOSING</span></div>${rows}</div>` +
        `<div class="chart-note" style="margin-top:10px">Row tint = that driver's tyre health at the closing-phase start (green ≥60% · yellow 30–60% · red &lt;30% · grey = no data) — the same Pirelli-style model the main dashboard's <b>Tyre Degradation</b> chart plots, so the two panels share one health timeline. Red rows mean the strategy projection is extrapolating past realistic stint life in the closing window.</div>` +
        (genAt
            ? `<div class="chart-note" style="margin-top:10px">Artifact generated ${genAt} · ${(data.meta && data.meta.script) || 'scripts/benchmark_energy_strategy.py'} · deterministic — same DB yields the same numbers, so the figure above is the script's own output.</div>`
            : '');
    out.innerHTML = html;
    if (meta) meta.textContent = src.artifact ? ('artifact ' + src.artifact + (src.mtime_iso ? ' · ' + src.mtime_iso : '')) : '';

    // Cumulative per-lap improvement (strategy − flat-out) for the first
    // driver, split at the closing-phase window so the deck window is visible.
    const first = sessions[0];
    if (first && first.per_lap && first.per_lap.lap && window.Chart) {
        const laps = first.per_lap.lap;
        const strat = first.per_lap.credits[first.strategy_mode];
        const flat = first.per_lap.credits[first.flat_out_mode];
        if (strat && flat && strat.length === laps.length) {
            const cum = [];
            let acc = 0;
            for (let i = 0; i < laps.length; i++) { acc += flat[i] - strat[i]; cum.push(+acc.toFixed(3)); }
            const closeLaps = cfg.closing_laps || 15;
            drawBmEnergyChart(laps, cum, laps[laps.length - closeLaps] || laps[0], first.driver_code, closeLaps);
        }
    } else {
        const card = document.getElementById('bm-energy-chart-card');
        if (card) card.style.display = 'none';
    }
}

function drawBmEnergyChart(laps, cum, closeStartLap, driver, closeLaps) {
    const card = document.getElementById('bm-energy-chart-card');
    const canvas = document.getElementById('bmEnergyChart');
    if (!card || !canvas) return;
    const closeIdx = laps.findIndex(l => l >= closeStartLap);
    const pre = cum.map((v, i) => (closeIdx < 0 || i < closeIdx) ? v : null);
    const win = cum.map((v, i) => (closeIdx >= 0 && i >= closeIdx) ? v : null);
    if (bmEnergyChart) bmEnergyChart.destroy();
    bmEnergyChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: laps,
            datasets: [
                { label: 'Race body', data: pre, borderColor: '#9aa', borderDash: [4, 3], pointRadius: 0, tension: 0.2 },
                { label: 'Closing phase (final ' + closeLaps + ' laps)', data: win, borderColor: '#ffd700', borderWidth: 2, pointRadius: 0, tension: 0.2 },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#ccc', font: { family: 'Share Tech Mono', size: 11 } } },
                tooltip: { callbacks: { label: c => 'Δ ' + c.parsed.y.toFixed(2) + 's' } },
                title: {
                    display: true,
                    text: driver + ' · cumulative race-time improvement (strategy − flat-out)',
                    color: '#eee', font: { family: 'Barlow Condensed', size: 14, weight: '700' },
                },
            },
            scales: {
                x: { title: { display: true, text: 'Lap', color: '#99a' }, ticks: { color: '#99a' }, grid: { color: '#ffffff0d' } },
                y: { title: { display: true, text: 'cumulative Δ (s)', color: '#99a' }, ticks: { color: '#99a' }, grid: { color: '#ffffff0d' } },
            },
        },
    });
    card.style.display = 'block';
}
