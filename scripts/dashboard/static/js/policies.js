// TACTICAL OPTIONS — the multi-policy decision engine UI.
//
// This tab is a pure VIEW over /api/strategy/policies: the scenario
// (leader / chaser / track / tyres / gap / lap) is inherited from the
// Hammer Time tab's Race Call inputs — the single source of truth — and
// only the decision-engine-specific knobs (battery override, reserve
// target) live here.  The response's per-policy score components are all
// rendered, so the panel answers "why not the alternative?" without a
// second request.

// ── Tab plumbing ─────────────────────────────────────────────────────────

// switchTab lives in whatif.js; this helper lets buttons elsewhere jump to
// a tab without a nav button element in hand.
function switchTabTo(name) {
    const btn = [...document.querySelectorAll('.f1-nav-btn')]
        .find(b => (b.getAttribute('onclick') || '').includes("'" + name + "'"));
    if (btn) switchTab(name, btn);
}

// Refresh the context bar from the Hammer Time inputs (called on tab open
// and whenever the scenario inputs change).
function poUpdateContext() {
    const L = htVal('ov-leader'), C = htVal('ov-chaser'), T = htVal('ov-track');
    const Y = htVal('ov-year'), lap = htVal('ov-lap') || '—', gap = htVal('ov-gap');
    const battle = document.getElementById('po-battle');
    if (battle) battle.textContent = (L || '—') + ' → ' + (C || '—');
    const meta = document.getElementById('po-meta');
    if (meta) meta.textContent =
        (T ? htTrackShort(T) : '—') + ' · ' + (Y || '—') + ' · LAP ' + lap +
        ' · INHERITED FROM HAMMER TIME';
    const g = document.getElementById('po-gap');
    if (g) g.textContent = 'GAP ' + (gap ? gap + 's' : '—');
    const chip = document.getElementById('po-posture-chip');
    if (chip) {
        // In the leader seat the posture lever is hidden (the engine models
        // the threat itself), so the chip would be stale — swap its meaning.
        const persp = document.querySelector('.po-persp-btn.active');
        const leaderView = persp && persp.getAttribute('data-perspective') === 'leader';
        if (leaderView) {
            chip.textContent = 'SEAT: DEFENDING (LEADER)';
            chip.style.borderColor = '#ffd700';
            chip.style.color = '#ffd700';
        } else {
            const def = document.querySelector('.po-posture-btn[data-posture="defensive_boost"]');
            const defensive = def && def.classList.contains('active');
            chip.textContent = 'LEADER: ' + (defensive ? 'DEFENDING' : 'PASSIVE');
            chip.style.borderColor = defensive ? '#ff6b6b' : '';
            chip.style.color = defensive ? '#ff6b6b' : '';
        }
    }
    const raw = htVal('ov-ers-batt');
    const own = htVal('po-batt-in');
    const b = document.getElementById('po-batt');
    const leaderView = (document.querySelector('.po-persp-btn.active') || {})
        .getAttribute && (document.querySelector('.po-persp-btn.active') || {}).getAttribute('data-perspective') === 'leader';
    if (b) {
        // Battery is a synthesized estimate: show the uncertainty band
        // (floor ±2% at the anchor, +0.5%/lap of drift, capped ±8% — the
        // same constants the backend's battery_uncertainty_band applies).
        const lap = parseInt(htVal('ov-lap'), 10) || 1;
        const band = Math.min(8, 2 + 0.5 * Math.max(0, lap - 1));
        let mean;
        if (own !== '') mean = Math.round(parseFloat(own) || 0);
        else if (raw !== '') mean = Math.round(100 * (parseFloat(raw) || 0) / 4.0);
        else mean = 62;
        b.textContent = (leaderView ? 'OUR BATT ' : 'BATT ') + mean + '% ±' + band.toFixed(1) + '%';
    }
}

// ── Payload assembly ─────────────────────────────────────────────────────

function poState() {
    const leader = htVal('ov-leader'), chaser = htVal('ov-chaser');
    const track = htVal('ov-track');
    if (!leader || !chaser || !track) {
        return { error: 'Pick the leader, chaser and race on the Hammer Time tab first (Calendar tab → click a round is fastest).' };
    }
    const body = {
        leader_code: leader,
        chaser_code: chaser,
        track_name: track,
        year: parseInt(htVal('ov-year'), 10) || null,
        start_lap: parseInt(htVal('ov-lap'), 10) || 1,
        race_length: parseInt(htVal('ov-racelaps'), 10) || 57,
        gap_before_s: parseFloat(htVal('ov-gap')) || 0.8,
        leader_tyre_compound: htVal('ov-ltyre') || 'Medium',
        chaser_tyre_compound: htVal('ov-ctyre') || 'Medium',
        leader_tyre_age: parseInt(htVal('ov-lage'), 10) || 0,
        chaser_tyre_age: parseInt(htVal('ov-cage'), 10) || 0,
    };
    // Battery: this tab's own override (% of the 4 MJ store) wins; else the
    // Race Call's MJ field; else the engine's working-band default.
    const own = htVal('po-batt-in');
    const mj = htVal('ov-ers-batt');
    if (own !== '') body.chaser_battery_pct = Math.max(0, Math.min(100, parseFloat(own) || 0));
    else if (mj !== '') body.chaser_battery_pct = Math.max(0, Math.min(100, 100 * (parseFloat(mj) || 0) / 4.0));
    const reserve = parseInt(htVal('po-reserve'), 10);
    if (Number.isFinite(reserve) && reserve > 0) {
        body.reserve_target_mj = Math.round(reserve / 100.0 * 4.0 * 100) / 100;
    }
    // Perspective: CHASER (we are attacking) or LEADER (we are defending).
    // The battery override's meaning flips with it — from the leader's
    // seat it is OUR battery, and the threat's battery is a separate input.
    const perspBtn = document.querySelector('.po-persp-btn.active');
    const perspective = (perspBtn && perspBtn.getAttribute('data-perspective')) || 'chaser';
    body.perspective = perspective;
    if (perspective === 'leader') {
        // Threat disclosure input: the attacking chaser's battery (blank =
        // the same mid-race default the threat model uses).
        const threat = htVal('po-threat-batt');
        if (threat !== '') body.threat_battery_pct = Math.max(0, Math.min(100, parseFloat(threat) || 0));
        // The reactive-posture lever is a chaser-seat concept; the leader
        // engine models the threat posture itself.
    } else {
        // Leader posture: the adversarial demo lever.  DEFENSIVE BOOST means
        // the leader counter-deploys from its own store — every closing-based
        // policy is scored against an opponent that reacts.
        const defBtn = document.querySelector('.po-posture-btn.active');
        body.leader_posture = (defBtn && defBtn.getAttribute('data-posture')) || 'balanced';
    }
    return { body };
}

// ── Rendering ────────────────────────────────────────────────────────────

function poPct(p) { return Math.round(100 * Math.min(1, Math.max(0, p || 0))); }

function poHeroRiskClass(card) {
    if (card.feasible === false) return { t: 'INFEASIBLE', c: '#ff6b6b' };
    if (card.battery_margin_pct != null && card.battery_margin_pct < 10)
        return { t: '⚠ THIN BATTERY MARGIN', c: '#ffb300' };
    if (card.confidence >= 0.5) return { t: '● DECISIVE', c: '#00c853' };
    return { t: 'MARGINAL CALL', c: '#ffd700' };
}

function poRenderActionCard(data) {
    const card = data.recommendation.action_card;
    const runner = data.recommendation.runner_up;
    const risk = poHeroRiskClass(card);
    const st = data.state;
    const leaderView = data.perspective === 'leader';
    const stat = (l, v, col) =>
        `<div class="hc-stat"><span class="hcs-l">${l}</span><span class="hcs-v"${col ? ` style="color:${col}"` : ''}>${v}</span></div>`;
    // Threat row: from the leader's seat the number that matters is the
    // ATTACK's probability, colour-coded by how alive it is.
    const threatP = leaderView ? card.threat_probability : card.overtake_probability;
    const threatLabel = leaderView ? 'ATTACK THREAT P' : 'OVERTAKE PROBABILITY';
    const threatCol = !leaderView ? undefined
        : (threatP >= 0.75 ? '#ff6b6b' : (threatP >= 0.4 ? '#ffb300' : '#00c853'));
    return `
    <div class="hero-call">
        <div class="hc-mode" style="color:var(--gold)">${card.action}</div>
        <div class="hc-tag">${card.reason}</div>
        <div class="hc-stats">
            ${stat(threatLabel, poPct(threatP) + '%', threatCol)}
            ${leaderView
                ? stat('HOLDS POSITION UNTIL', card.converted !== false && card.pass_lap
                    ? 'L' + card.pass_lap + ' (' + (card.laps_held || 0) + ' laps held)'
                    : 'FLAG — position survives', threatP < 0.4 ? '#00c853' : undefined)
                : stat('PASS / DEPLOY LAP', 'L' + card.deploy_lap)}
            ${stat('ENERGY COST', card.energy_cost_mj.toFixed(2) + ' MJ' +
                (card.energy_pct ? ' (' + card.energy_pct + '%)' : ''))}
            ${stat('EXPECTED FINISH Δ', (card.expected_finish_delta_s >= 0 ? '+' : '') +
                card.expected_finish_delta_s.toFixed(2) + 's',
                card.expected_finish_delta_s < 0 ? '#00c853' : '#ff6b6b')}
            ${stat('BATTERY MARGIN', card.battery_margin_pct != null
                ? '+' + card.battery_margin_pct.toFixed(0) + '%'
                + (card.battery_margin_worst_pct != null
                    ? ' <span style="color:#889;font-size:0.75em">(worst ±band: '
                      + (card.battery_margin_worst_pct >= 0 ? '+' : '')
                      + card.battery_margin_worst_pct.toFixed(0) + '%)</span>'
                    : '')
                : '—')}
            ${stat('CONFIDENCE', Math.round(100 * card.confidence) + '%')}
        </div>
        <div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap;align-items:center">
            <span class="po-risk" style="color:${risk.c};border-color:${risk.c}">${risk.t}</span>
            ${leaderView && data.recommendation.no_hope_disclosure
                ? `<span class="po-risk" style="color:#ff6b6b;border-color:#ff6b6b">⚠ ATTACK LIKELY UNDER EVERY DEFENCE</span>` : ''}
            ${runner ? `<span style="font-family:'Share Tech Mono',monospace;font-size:0.62em;letter-spacing:1px;color:#889">
                vs runner-up ${runner.policy} (margin ${data.recommendation.decision_margin_s.toFixed(2)}s)</span>` : ''}
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.62em;letter-spacing:1px;color:#667">
                ${data.latency_ms.toFixed(0)} ms · deterministic — same state, same call</span>
        </div>
        <div class="chart-note" style="margin-top:10px">
            State: ${st.leader} vs ${st.chaser} · ${st.track} · lap ${st.start_lap}/${st.race_length} ·
            gap ${st.gap_before_s}s ·
            ${leaderView
                ? `OUR (leader) battery ${st.battery_pct}%${st.battery_band_pct != null ? ' ±' + st.battery_band_pct + '% (synthetic)' : ''} ·
                   threat model: chaser attacks at ${st.threat_assumption.chaser_ers}% lever, its battery ${st.threat_assumption.chaser_battery_pct}%`
                : `chaser battery ${st.battery_pct}%${st.battery_band_pct != null ? ' ±' + st.battery_band_pct + '% (synthetic)' : ''}`} ·
            reserve target ${st.reserve_target_pct}% of store. Score prices the
            battery at the <b>worst case</b> of its band; confidence degrades as the band widens.
        </div>
    </div>`;
}

function poRenderMatrix(data) {
    const rows = [...data.policies].sort((a, b) => a.score_s - b.score_s);
    const best = data.recommendation.action_card.action;
    const leaderView = data.perspective === 'leader';
    if (leaderView) return poRenderMatrixLeader(data, rows, best);
    const head = `<div class="tbl-row tbl-head" style="grid-template-columns:1.35fr .55fr .55fr .6fr .6fr .6fr 1.5fr">
        <span>POLICY</span><span>P(PASS)</span><span>PASS LAP</span><span>E MJ</span>
        <span>FIN Δ</span><span>SCORE</span><span>WHY / REJECTION</span></div>`;
    const body = rows.map(p => {
        const winner = p.policy === best;
        const fin = (p.score_components.pass_gain_s - p.score_components.battery_cost_s
                     - p.score_components.wear_cost_s - p.score_components.risk_cost_s
                     - p.score_components.latency_cost_s);
        const verdict = p.feasible
            ? (p.pass_lap ? `L${p.pass_lap}` : 'no pass')
            : 'REJECTED';
        const verdictCol = !p.feasible ? '#ff6b6b' : (p.pass_lap ? '#ff6b6b' : '#889');
        const why = p.feasible
            ? (p.why || '')
            : (p.infeasible_reason || 'infeasible');
        return `<div class="tbl-row po-row${winner ? ' po-winner' : ' po-loser'}" style="grid-template-columns:1.35fr .55fr .55fr .6fr .6fr .6fr 1.5fr" title="${why}">
            <span style="font-family:'Barlow Condensed',sans-serif;font-weight:700;letter-spacing:1px">${p.policy}${winner ? ' ★' : ''}</span>
            <span style="color:${p.overtake_probability >= 0.5 ? '#ff6b6b' : '#cdd'}">${poPct(p.overtake_probability)}%</span>
            <span style="color:${verdictCol}">${verdict}</span>
            <span>${p.energy_cost_mj.toFixed(2)}</span>
            <span style="color:${fin < 0 ? '#00c853' : '#ff6b6b'}">${fin >= 0 ? '+' : ''}${fin.toFixed(2)}s</span>
            <span style="color:${winner ? 'var(--gold)' : '#9ab'}">${p.score_s.toFixed(2)}</span>
            <span style="color:${p.feasible ? '#667' : '#ff9f43'}">${why}</span>
        </div>`;
    }).join('');
    // Cost breakdown for the winner — where the score actually went.
    const win = rows.find(p => p.policy === best);
    let breakdown = '';
    if (win) {
        const sc = win.score_components;
        breakdown = `<div class="chart-note" style="margin-top:10px">
            Score ledger — <b>${win.policy}</b>: pass +${sc.pass_gain_s.toFixed(2)}s ·
            battery −${sc.battery_cost_s.toFixed(2)}s · wear −${sc.wear_cost_s.toFixed(2)}s ·
            cliff −${sc.cliff_cost_s.toFixed(2)}s · risk −${sc.risk_cost_s.toFixed(2)}s ·
            deferral −${sc.latency_cost_s.toFixed(2)}s = <b>${win.score_s.toFixed(2)}</b>
            (lower wins).</div>`;
    }
    return `<div class="alt-head">POLICY COMPARISON — ALL FIVE, RANKED BY SCORE</div>
        <div class="tbl-wrap">${head}${body}</div>${breakdown}`;
}

// Leader-perspective matrix: the columns answer the defender's questions —
// does the attack land, how long does the position hold, what did defence cost?
function poRenderMatrixLeader(data, rows, best) {
    const head = `<div class="tbl-row tbl-head" style="grid-template-columns:1.3fr .5fr .5fr .5fr .55fr .6fr 1.45fr">
        <span>DEFENCE</span><span>THREAT</span><span>LANDS</span><span>HELD</span>
        <span>E MJ</span><span>SCORE</span><span>WHY / REJECTION</span></div>`;
    const body = rows.map(p => {
        const winner = p.policy === best;
        const lands = p.converted ? `L${p.pass_lap}` : 'FLAG';
        const landCol = p.converted ? '#ff6b6b' : '#00c853';
        const why = p.feasible ? (p.why || '') : (p.infeasible_reason || 'infeasible');
        return `<div class="tbl-row po-row${winner ? ' po-winner' : ' po-loser'}" style="grid-template-columns:1.3fr .5fr .5fr .5fr .55fr .6fr 1.45fr" title="${why}">
            <span style="font-family:'Barlow Condensed',sans-serif;font-weight:700;letter-spacing:1px">${p.policy}${winner ? ' ★' : ''}</span>
            <span style="color:${p.overtake_probability >= 0.75 ? '#ff6b6b' : (p.overtake_probability >= 0.4 ? '#ffb300' : '#00c853')}">${poPct(p.overtake_probability)}%</span>
            <span style="color:${landCol}">${lands}</span>
            <span>${p.laps_held != null ? p.laps_held + ' laps' : '—'}</span>
            <span>${(p.energy_cost_mj || 0).toFixed(2)}</span>
            <span style="color:${winner ? 'var(--gold)' : '#9ab'}">${p.score_s.toFixed(2)}</span>
            <span style="color:${p.feasible ? '#667' : '#ff9f43'}">${why}</span>
        </div>`;
    }).join('');
    const win = rows.find(p => p.policy === best);
    let breakdown = '';
    if (win) {
        const sc = win.score_components;
        breakdown = `<div class="chart-note" style="margin-top:10px">
            Score ledger — <b>${win.policy}</b>: position risk +${sc.position_risk_s.toFixed(2)}s ·
            hold credit −${sc.hold_credit_s.toFixed(2)}s · battery −${sc.battery_cost_s.toFixed(2)}s ·
            wear −${sc.wear_cost_s.toFixed(2)}s = <b>${win.score_s.toFixed(2)}</b>
            (lower wins; each lap the defence buys earns credit).</div>`;
    }
    return `<div class="alt-head">DEFENCE COMPARISON — THE ATTACK IS MODELLED AT ${data.scoring_constants.threat_chaser_ers}% LEVER</div>
        <div class="tbl-wrap">${head}${body}</div>${breakdown}`;
}

function poRender(data) {
    const out = document.getElementById('po-out');
    if (!out) return;
    const infeas = data.recommendation.infeasible_policies || [];
    out.innerHTML =
        poRenderActionCard(data) +
        poRenderMatrix(data) +
        (infeas.length
            ? `<div class="chart-note" style="color:#ff9f43;margin-top:8px">
               ⚠ ${infeas.length} of 5 policies breached the reserve constraint and were pruned:
               ${infeas.join(', ')}. That pruning IS the decision — the engine refuses to buy a
               pass with battery it will not have.</div>`
            : '');
}

// ── The call ─────────────────────────────────────────────────────────────

let poBusy = false;

async function runPolicyEngine() {
    if (poBusy) return;
    const err = document.getElementById('po-err');
    if (err) err.style.display = 'none';
    const st = poState();
    if (st.error) {
        if (err) { err.textContent = st.error; err.style.display = 'block'; }
        return;
    }
    poBusy = true;
    const btn = document.getElementById('po-run-btn');
    if (btn) btn.disabled = true;
    const out = document.getElementById('po-out');
    if (out) out.innerHTML = '<div class="chart-note">Evaluating five policies… (first call loads the pace models)</div>';
    try {
        const res = await fetch('/api/strategy/policies', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(st.body)
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        poRender(data);
    } catch (e) {
        if (err) { err.textContent = 'Engine error: ' + e.message; err.style.display = 'block'; }
        if (out && out.querySelector('.chart-note') && !out.querySelector('.hero-call'))
            out.innerHTML = '';
    } finally {
        poBusy = false;
        if (btn) btn.disabled = false;
    }
}

// ── Wiring ───────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Keep the context bar in step with the Hammer Time scenario.
    ['ov-leader', 'ov-chaser', 'ov-track', 'ov-year', 'ov-lap', 'ov-gap',
     'ov-ers-batt'].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('change', poUpdateContext);
        el.addEventListener('input', poUpdateContext);
    });
    ['po-batt-in', 'po-reserve', 'po-threat-batt'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', poUpdateContext);
    });
    poUpdateContext();
});

// Toggle the leader-posture lever (PASSIVE vs DEFENSIVE BOOST).  Buttons
// are radio-like within #po-posture; the context chip mirrors the choice.
function poSetPosture(btn) {
    document.querySelectorAll('#po-posture .po-posture-btn')
        .forEach(b => b.classList.toggle('active', b === btn));
    poUpdateContext();
}

// Switch seat: CHASER (we attack) vs LEADER (we defend).  Flips the input
// set (leader seat adds the threat-battery input, hides the posture lever),
// the battery label's meaning, and the hero copy.
function poSetPerspective(btn) {
    document.querySelectorAll('#po-persp .po-persp-btn')
        .forEach(b => b.classList.toggle('active', b === btn));
    const leader = btn.getAttribute('data-perspective') === 'leader';
    const lx = document.getElementById('po-leader-extras');
    const cx = document.getElementById('po-chaser-extras');
    if (lx) lx.style.display = leader ? '' : 'none';
    if (cx) cx.style.display = leader ? 'none' : '';
    const lbl = document.getElementById('po-batt-label');
    if (lbl) lbl.textContent = leader
        ? 'OUR battery as leader (% — blank = ~62% band)'
        : 'Chaser battery now (% — blank = ~62% band)';
    const line = document.getElementById('po-hero-line');
    if (line) line.textContent = leader
        ? 'The car behind is attacking — five defences are scored on how long the position holds and what holding costs.'
        : 'Five competing policies are scored against the same race state — push the button and let them fight it out.';
    // A rendered card from the other seat is stale the moment the seat flips.
    const out = document.getElementById('po-out');
    if (out && out.querySelector('.hero-call')) out.innerHTML =
        '<div class="chart-note">Perspective changed — re-run the engine for this seat.</div>';
    poUpdateContext();
}
