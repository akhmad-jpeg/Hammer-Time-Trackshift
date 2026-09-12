// THE CALL — one merged decision from BOTH engines (unified UI).
//
// This controller drives the merged "WHAT'S THE CALL?" module in the Hammer
// Time tab.  The module is a live view over /api/strategy/call: the scenario
// (leader / chaser / track / tyres / gap / lap / battery) is set right there
// in the same panel, and the backend runs BOTH decision engines coupled —
// the leader engine names the defence it would run, that defence sets the
// posture the attack policies are scored against, and the seat switch only
// chooses whose call is surfaced as THE CALL.  Every policy from both seats
// (ten rows) is rendered below the call, so "why not the alternative?" —
// across BOTH sides of the battle — is answered without a second request.
//
// AUTO-REFRESH: any scenario input change (typed, dragged or programmatically
// filled via htUpdateContext) schedules a debounced re-run, so the surfaced
// decision can never go stale.  Changes that land while a request is in
// flight queue exactly one trailing re-run — the last state is always the
// one rendered.  The engine cache on the server makes repeated re-runs of an
// unchanged state ~free; a genuinely changed state walks fresh.

// ── Seat plumbing ────────────────────────────────────────────────────────

// The active seat ('chaser' = we attack, 'leader' = we defend).  Both
// engines always run; this only picks the surfaced call.
function ucSeat() {
    const btn = document.querySelector('#uc-persp .po-persp-btn.active');
    return (btn && btn.getAttribute('data-perspective')) || 'chaser';
}

// Switch seat: CHASER (we attack) vs LEADER (we defend).  Flips the threat
// input's visibility, the battery label's meaning and the hero copy.  The
// scenario inputs (drivers / track / lap / gap / tyres / battery) are shared.
function ucSetPerspective(btn) {
    document.querySelectorAll('#uc-persp .po-persp-btn')
        .forEach(b => b.classList.toggle('active', b === btn));
    const leader = btn.getAttribute('data-perspective') === 'leader';
    const tw = document.getElementById('uc-threat-wrap');
    if (tw) tw.style.display = leader ? '' : 'none';
    const lbl = document.getElementById('uc-batt-label');
    if (lbl) lbl.textContent = leader
        ? 'Our battery now — the defender (MJ of the 4.0 MJ store)'
        : 'Our battery now — the chaser (MJ of the 4.0 MJ store)';
    const shapeLbl = document.getElementById('uc-shape-label');
    if (shapeLbl) shapeLbl.textContent = leader
        ? 'Our ERS sector shape — the leader\'s defence (MJ/lap deploy delta per sector)'
        : 'Our ERS sector shape — the chaser\'s attack (MJ/lap deploy delta per sector)';
    const line = document.getElementById('ht-hero-line');
    if (line) line.textContent = leader
        ? 'The car behind is attacking — the defence engine answers from this seat.'
        : 'The attack engine answers from this seat — scored against the defence the opponent\'s engine actually recommends.';
    // (The scenario narrative reclaims this line whenever a race-state input
    // changes — see htUpdateContext in whatif.js.)
    // The slider bank follows the seat: reset it on the flip so a drag
    // made for one car can never silently ride the other car's call.
    [1, 2, 3].forEach(k => {
        const el = document.getElementById('uc-' + UC_SHAPE_PREFIX + 'd' + k);
        if (el) el.value = '0';
    });
    ucSectorInput();
    // A rendered card from the other seat is stale the moment the seat flips:
    // schedule the auto-refresh instead of demanding a manual re-run.
    const out = document.getElementById('uc-out');
    if (out && out.querySelector('.hero-call')) out.innerHTML =
        '<div class="chart-note">Seat changed — rescoring both engines for this seat…</div>';
    ucScheduleAuto();
}

// MJ of the 4.0 MJ store -> percent, clamped to [0, 100].
function ucMjToPct(raw) {
    if (raw === '' || raw == null) return null;
    const mj = parseFloat(raw);
    if (!Number.isFinite(mj)) return null;
    return Math.max(0, Math.min(100, 100 * mj / 4.0));
}

// ── ERS sector slider bank (Energy Sandbox vocabulary) ──────────────────

// ONE slider bank, bound to the selected seat: the chaser's attack when we
// sit chaser, the leader's defence when we sit leader.  There is no second
// bank — you always steer the car whose seat is selected.

// The active bank's field prefix ('s' — ids uc-sd1..3).
const UC_SHAPE_PREFIX = 's';

// Read the slider bank (ids uc-sd1..3) as [s1, s2, s3] MJ/lap, or null
// when the whole bank sits at zero (backend collapses that to no shape).
function ucShapeVec() {
    const v = [1, 2, 3].map(k => parseFloat(htVal('uc-' + UC_SHAPE_PREFIX + 'd' + k)) || 0);
    return v.some(x => Math.abs(x) > 1e-9) ? v : null;
}

// Live value/sum readout for the bank — the sandbox's sbSliderInput,
// mirrored so a slider drag shows its energy meaning before any run.
function ucSectorInput() {
    const tail = ucSeat() === 'leader'
        ? 'every defence expresses its lever through this shape (zero-sum reallocates pace, net deploy draws the store)'
        : "every attack policy expresses its lever through this shape (zero-sum reallocates pace, net deploy draws the store)";
    const d = [1, 2, 3].map(k => parseFloat(htVal('uc-' + UC_SHAPE_PREFIX + 'd' + k)) || 0);
    d.forEach((v, k) => {
        const el = document.getElementById('uc-' + UC_SHAPE_PREFIX + 'v' + (k + 1));
        if (el) el.textContent = (v >= 0 ? '+' : '') + v.toFixed(2);
    });
    const sum = d[0] + d[1] + d[2];
    const sumEl = document.getElementById('uc-' + UC_SHAPE_PREFIX + '-sum');
    if (sumEl) {
        sumEl.textContent = 'Σ Δ = ' + (sum >= 0 ? '+' : '') + sum.toFixed(2) +
            ' MJ/lap · ' + (Math.abs(sum) < 0.005
                ? 'pure reallocation — store-neutral, pace shape only'
                : (sum > 0 ? "net deploy from the car's 4.0 MJ store (30% floor)" : 'net bank to the car\'s 4.0 MJ store (full = stop lifting)')) +
            ' · ' + tail;
    }
}

// ── Payload assembly (the panel's own inputs are the single source) ──────

function ucState() {
    const leader = htVal('ov-leader'), chaser = htVal('ov-chaser');
    const track = htVal('ov-track');
    if (!leader || !chaser || !track) {
        return { error: 'Pick both drivers and the race first — fastest: Calendar tab → click a round.' };
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
        // The seat chooses whose call is surfaced; both engines always run
        // and the leader engine's recommendation conditions the posture.
        perspective: ucSeat(),
    };
    // OUR battery: the seat's own car (MJ of the 4.0 MJ store).
    const batt = ucMjToPct(htVal('ov-ers-batt'));
    if (batt != null) body.battery_pct = batt;
    // Per-sector ERS slider bank (the Energy Sandbox's exact vocabulary:
    // MJ/lap deploy delta per sector) — ONE bank, bound to the selected
    // seat: from the chaser seat it expresses every attack policy's lever;
    // from the leader seat it expresses every defence's lever.  An
    // all-zero bank sends no field — it collapses to the no-shape walk.
    const shape = ucShapeVec();
    if (shape) {
        if (body.perspective === 'leader') body.leader_ers_shape = shape;
        else body.chaser_ers_shape = shape;
    }
    // Reserve target (% of store -> MJ): policies below it are rejected.
    const reservePct = parseFloat(htVal('uc-reserve'));
    if (Number.isFinite(reservePct) && reservePct > 0) {
        body.reserve_target_mj = Math.round(reservePct / 100.0 * 4.0 * 100) / 100;
    }
    // Threat battery (leader seat only): the attacking car's store.
    if (body.perspective === 'leader') {
        const threat = ucMjToPct(htVal('uc-threat-batt'));
        if (threat != null) body.threat_battery_pct = threat;
    }
    return { body };
}

// ── Rendering ────────────────────────────────────────────────────────────

function ucPct(p) { return Math.round(100 * Math.min(1, Math.max(0, p || 0))); }

function ucRiskClass(card) {
    if (card.feasible === false) return { t: 'INFEASIBLE', c: '#ff6b6b' };
    if (card.battery_margin_pct != null && card.battery_margin_pct < 10)
        return { t: '⚠ THIN BATTERY MARGIN', c: '#ffb300' };
    if (card.confidence >= 0.5) return { t: '● DECISIVE', c: '#00c853' };
    return { t: 'MARGINAL CALL', c: '#ffd700' };
}

// THE CALL hero: the surfaced seat's action card, the opponent engine's
// answer, and the race projection behind the manoeuvre.
function ucRenderCall(data) {
    const fc = data.final_call;
    const seat = data.perspective;
    const own = (seat === 'leader' ? data.leader : data.chaser).recommendation;
    const card = own.action_card;
    const risk = ucRiskClass(card);
    const st = data.state;
    const leaderView = seat === 'leader';
    const stat = (l, v, col) =>
        `<div class="hc-stat"><span class="hcs-l">${l}</span><span class="hcs-v"${col ? ` style="color:${col}"` : ''}>${v}</span></div>`;

    // Seat-aware headline numbers.
    const threatP = leaderView ? card.threat_probability : card.overtake_probability;
    const threatLabel = leaderView ? 'ATTACK THREAT P' : 'OVERTAKE PROJECTION';
    const threatCol = !leaderView ? undefined
        : (threatP >= 0.75 ? '#ff6b6b' : (threatP >= 0.4 ? '#ffb300' : '#00c853'));
    const verdictTxt = fc.projection && fc.projection.verdict
        ? String(fc.projection.verdict).toUpperCase() : '—';

    // Coupling disclosure — the merged engine's whole point, in one line.
    const coup = fc.engine_coupling || {};
    const opp = fc.opponent || {};

    return `
    <div class="hero-call">
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
            <div class="hc-mode" style="color:var(--gold)">${fc.action}</div>
            <span class="po-risk" style="color:${leaderView ? '#ffd700' : '#ff6b6b'};border-color:${leaderView ? '#ffd700' : '#ff6b6b'}">${leaderView ? 'SEAT: LEADER — DEFENDING' : 'SEAT: CHASER — ATTACKING'}</span>
        </div>
        <div class="hc-tag">${fc.reason}</div>
        <div class="hc-stats">
            ${stat(threatLabel, ucPct(threatP) + '%', threatCol)}
            ${leaderView
                ? stat('HOLDS POSITION UNTIL', card.converted !== false && card.pass_lap
                    ? 'L' + card.pass_lap + ' (' + (card.laps_held || 0) + ' laps held)'
                    : 'FLAG — position survives', threatP < 0.4 ? '#00c853' : undefined)
                : stat('PASS / DEPLOY LAP', 'L' + (card.deploy_lap || fc.deploy_lap || st.start_lap))}
            ${stat('RACE PROJECTION', verdictTxt + (fc.projection && fc.projection.pass_lap ? ' · PASS L' + fc.projection.pass_lap : ''),
                fc.projection && fc.projection.pass_lap ? '#ff6b6b' : undefined)}
            ${stat('ENERGY COST', (card.energy_cost_mj || 0).toFixed(2) + ' MJ' +
                (card.energy_pct ? ' (' + card.energy_pct + '%)' : ''))}
            ${stat('EXPECTED FINISH Δ', (card.expected_finish_delta_s >= 0 ? '+' : '') +
                (card.expected_finish_delta_s || 0).toFixed(2) + 's',
                card.expected_finish_delta_s < 0 ? '#00c853' : '#ff6b6b')}
            ${stat('BATTERY MARGIN', card.battery_margin_pct != null
                ? '+' + card.battery_margin_pct.toFixed(0) + '%'
                + (card.battery_margin_worst_pct != null
                    ? ' <span style="color:#889;font-size:0.75em">(worst ±band: '
                      + (card.battery_margin_worst_pct >= 0 ? '+' : '')
                      + card.battery_margin_worst_pct.toFixed(0) + '%)</span>'
                    : '')
                : '—')}
            ${stat('CONFIDENCE', Math.round(100 * (card.confidence || 0)) + '%')}
        </div>
        <div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap;align-items:center">
            <span class="po-risk" style="color:${risk.c};border-color:${risk.c}">${risk.t}</span>
            ${leaderView && own.no_hope_disclosure
                ? `<span class="po-risk" style="color:#ff6b6b;border-color:#ff6b6b">⚠ ATTACK LIKELY UNDER EVERY DEFENCE</span>` : ''}
            ${own.runner_up ? `<span style="font-family:'Share Tech Mono',monospace;font-size:0.62em;letter-spacing:1px;color:#889">
                vs runner-up ${own.runner_up.policy} (margin ${(own.decision_margin_s || 0).toFixed(2)}s)</span>` : ''}
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.62em;letter-spacing:1px;color:#667">
                ${data.latency_ms.toFixed(0)} ms${(data.engine_cache && (data.engine_cache.leader_reused || data.engine_cache.chaser_reused)) ? ' · re-used stored engine evaluation (same state)' : ' · fresh walk simulation'} — deterministic, same state = same call</span>
        </div>
        <div class="chart-note" style="margin-top:10px">
            <b>BOTH engines ran.</b> The opponent's (${opp.seat}) engine calls
            <b style="color:#ffd700">${opp.action}</b>
            <span title="${(opp.reason || '').replace(/"/g, '&quot;')}" style="color:#889">— hover for its reasoning</span>.
            ${coup.note || ''}
        </div>
        <div class="chart-note" style="margin-top:6px">
            State: ${st.leader} vs ${st.chaser} · ${st.track} · lap ${st.start_lap}/${st.race_length} ·
            gap ${st.gap_before_s}s · ${leaderView
                ? `our (leader) battery ${data.leader_state.battery_pct}%${data.leader_state.battery_band_pct != null ? ' ±' + data.leader_state.battery_band_pct + '% (synthetic)' : ''}`
                : `chaser battery ${st.battery_pct}%${st.battery_band_pct != null ? ' ±' + st.battery_band_pct + '% (synthetic)' : ''}`} ·
            reserve target ${st.reserve_target_pct}% of store. Score prices the
            battery at the <b>worst case</b> of its band; confidence degrades as the band widens.
        </div>
    </div>`;
}

// ONE table, both seats: every policy from the attack engine AND the defence
// engine, ranked by score, with the surfaced seat's winner starred.
function ucRenderPolicies(data) {
    const fc = data.final_call;
    const rows = [...data.policies].sort((a, b) => a.score_s - b.score_s);
    const head = `<div class="tbl-row tbl-head" style="grid-template-columns:.62fr 1.3fr .55fr .6fr .5fr .5fr 1.55fr">
        <span>SEAT</span><span>POLICY</span><span>P(PASS)</span><span>PASS LAP</span><span>E MJ</span>
        <span>SCORE</span><span>WHY / REJECTION</span></div>`;
    const body = rows.map(p => {
        const winner = (p.policy === fc.action && p.seat === fc.seat);
        const isLeader = p.seat === 'leader';
        const seatTag = isLeader
            ? `<span style="color:#ffd700">DEFEND</span>` : `<span style="color:#ff6b6b">ATTACK</span>`;
        const verdict = p.feasible
            ? (isLeader
                ? (p.converted ? `lands L${p.pass_lap}` : 'holds to flag')
                : (p.pass_lap ? `L${p.pass_lap}` : 'no pass'))
            : 'REJECTED';
        const verdictCol = !p.feasible ? '#ff6b6b' : ((p.pass_lap && !(isLeader && !p.converted)) ? '#ff6b6b' : (isLeader ? '#00c853' : '#889'));
        const why = p.feasible ? (p.why || '') : (p.infeasible_reason || 'infeasible');
        return `<div class="tbl-row po-row${winner ? ' po-winner' : ' po-loser'}" style="grid-template-columns:.62fr 1.3fr .55fr .6fr .5fr .5fr 1.55fr" title="${why}">
            <span>${seatTag}</span>
            <span style="font-family:'Barlow Condensed',sans-serif;font-weight:700;letter-spacing:1px">${p.policy}${winner ? ' ★' : ''}</span>
            <span style="color:${p.overtake_probability >= 0.5 ? '#ff6b6b' : '#cdd'}">${ucPct(p.overtake_probability)}%</span>
            <span style="color:${verdictCol}">${verdict}</span>
            <span>${(p.energy_cost_mj || 0).toFixed(2)}</span>
            <span style="color:${winner ? 'var(--gold)' : '#9ab'}">${p.score_s.toFixed(2)}</span>
            <span style="color:${p.feasible ? '#667' : '#ff9f43'}">${why}</span>
        </div>`;
    }).join('');

    // Score ledgers for both engines' winners — where the scores went.
    const cw = (data.chaser.recommendation.action_card || {});
    const chaserWin = rows.find(p => p.seat === 'chaser' && p.policy === cw.action);
    const leaderWin = rows.find(p => p.seat === 'leader'
        && p.policy === (data.leader.recommendation.action_card || {}).action);
    let ledgers = '';
    if (chaserWin && chaserWin.score_components && chaserWin.score_components.pass_gain_s !== undefined) {
        const sc = chaserWin.score_components;
        ledgers += `<div class="chart-note" style="margin-top:10px">
            Attack ledger — <b>${chaserWin.policy}</b>: pass +${sc.pass_gain_s.toFixed(2)}s ·
            battery −${sc.battery_cost_s.toFixed(2)}s · wear −${sc.wear_cost_s.toFixed(2)}s ·
            cliff −${sc.cliff_cost_s.toFixed(2)}s · risk −${sc.risk_cost_s.toFixed(2)}s ·
            deferral −${sc.latency_cost_s.toFixed(2)}s = <b>${chaserWin.score_s.toFixed(2)}</b>
            (lower wins).</div>`;
    }
    if (leaderWin && leaderWin.score_components && leaderWin.score_components.position_risk_s !== undefined) {
        const sc = leaderWin.score_components;
        ledgers += `<div class="chart-note" style="margin-top:6px">
            Defence ledger — <b>${leaderWin.policy}</b>: position risk +${sc.position_risk_s.toFixed(2)}s ·
            hold credit −${sc.hold_credit_s.toFixed(2)}s · battery −${sc.battery_cost_s.toFixed(2)}s ·
            wear −${sc.wear_cost_s.toFixed(2)}s · cliff −${(sc.cliff_cost_s || 0).toFixed(2)}s
            = <b>${leaderWin.score_s.toFixed(2)}</b>
            (each lap the defence buys earns credit).</div>`;
    }
    const infeasChaser = data.chaser.recommendation.infeasible_policies || [];
    const infeasLeader = data.leader.recommendation.infeasible_policies || [];
    let infeas = '';
    if (infeasChaser.length || infeasLeader.length) {
        infeas = `<div class="chart-note" style="color:#ff9f43;margin-top:8px">
            ⚠ Reserve-constraint pruning: ${infeasChaser.length
                ? `attack — ${infeasChaser.join(', ')}` : ''}${(infeasChaser.length && infeasLeader.length) ? ' · ' : ''}
            ${infeasLeader.length ? `defence — ${infeasLeader.join(', ')}` : ''}.
            That pruning IS the decision — the engine refuses to buy a pass (or a defence)
            with battery the car will not have.</div>`;
    }
    return `<div class="alt-head">ALL POLICIES — BOTH ENGINES, RANKED BY SCORE (★ = the surfaced seat's winner)</div>
        <div class="tbl-wrap">${head}${body}</div>${ledgers}${infeas}`;
}

function ucRender(data) {
    const out = document.getElementById('uc-out');
    if (!out) return;
    out.innerHTML = ucRenderCall(data) + ucRenderPolicies(data);
}

// ── The call ─────────────────────────────────────────────────────────────

let ucBusy = false;
let ucAutoTimer = null;
let ucSeq = 0;              // request token: only the latest may render
let ucRerunQueued = false;  // a change landed while a request was in flight

// Debounce window: long enough that a slider drag / number spin collapses
// into one request, short enough that the card feels live.
const UC_AUTO_DEBOUNCE_MS = 700;

function ucScheduleAuto() {
    if (ucAutoTimer) clearTimeout(ucAutoTimer);
    ucAutoTimer = setTimeout(() => { ucAutoTimer = null; runUnifiedCall(true); },
                             UC_AUTO_DEBOUNCE_MS);
}

async function runUnifiedCall(fromAuto) {
    if (ucBusy) {
        // A debounced fire or a manual press while a request is in flight:
        // honour it with exactly one trailing re-run after the current one
        // lands — the last state is always the one rendered.
        ucRerunQueued = true;
        return;
    }
    const err = document.getElementById('uc-err');
    if (err) err.style.display = 'none';
    const st = ucState();
    if (st.error) {
        // Auto path: an incomplete scenario is the user mid-editing — never
        // error-spam, but a stale card must not survive an invalid state.
        const out = document.getElementById('uc-out');
        if (out && (out.querySelector('.hero-call') || fromAuto)) {
            out.innerHTML = '<div class="chart-note">Scenario incomplete — ' +
                st.error.charAt(0).toLowerCase() + st.error.slice(1) +
                ' The call re-runs automatically once the state is usable.</div>';
        }
        return;
    }
    ucBusy = true;
    const seq = ++ucSeq;
    const btn = document.getElementById('uc-run-btn');
    if (btn) btn.disabled = true;
    const out = document.getElementById('uc-out');
    if (out) {
        if (out.querySelector('.hero-call')) {
            // Re-scoring over a rendered card: keep it visible, mark it stale.
            let note = document.getElementById('uc-live-note');
            if (!note) {
                note = document.createElement('div');
                note.id = 'uc-live-note';
                note.className = 'chart-note';
                note.style.cssText = 'color:#ffd700;border:1px solid #ffd70055;border-radius:4px;padding:4px 10px;margin-bottom:8px';
                out.prepend(note);
            }
            note.textContent = '⟳ Inputs changed — rescoring both engines…';
        } else {
            out.innerHTML = '<div class="chart-note">Running BOTH engines — five attack policies, five defences, coupled… (first call loads the pace models)</div>';
        }
    }
    try {
        const res = await fetch('/api/strategy/call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(st.body)
        });
        const data = await res.json();
        if (seq !== ucSeq) return;          // a newer run owns the panel
        if (data.error) throw new Error(data.error);
        ucRender(data);
    } catch (e) {
        if (seq !== ucSeq) return;
        if (err) { err.textContent = 'Call error: ' + e.message; err.style.display = 'block'; }
        const note = document.getElementById('uc-live-note');
        if (note) note.remove();
        if (out && out.querySelector('.chart-note') && !out.querySelector('.hero-call'))
            out.innerHTML = '';
    } finally {
        ucBusy = false;
        if (btn) btn.disabled = false;
        if (ucRerunQueued) {
            ucRerunQueued = false;
            runUnifiedCall(true);           // trailing guarantee: last state wins
        }
    }
}

// ── Auto-refresh wiring ─────────────────────────────────────────────────────

// Every input that defines the surfaced decision.  Manual edits are caught
// by delegated input/change listeners; programmatic fills (live sync,
// calendar round click, preset gap) end in htUpdateContext, which calls
// ucScheduleAuto from whatif.js — both roads lead to the same scheduler.
const UC_AUTO_IDS = new Set([
    'ov-leader', 'ov-chaser', 'ov-track', 'ov-year',
    'ov-lap', 'ov-racelaps', 'ov-gap',
    'ov-ltyre', 'ov-ctyre', 'ov-lage', 'ov-cage',
    'ov-ers-batt', 'uc-reserve', 'uc-threat-batt',
]);

// The slider bank carries its own oninput (ucSectorInput for the live Σ);
// the range 'input' events still reach the delegated listener below, so
// map them onto the auto-refresh set by id prefix.
document.addEventListener('input', e => {
    if (e.target instanceof Element && /^uc-sd[123]$/.test(e.target.id)) ucScheduleAuto();
});

document.addEventListener('input', e => {
    if (e.target instanceof Element && UC_AUTO_IDS.has(e.target.id)) ucScheduleAuto();
});
document.addEventListener('change', e => {
    if (e.target instanceof Element && UC_AUTO_IDS.has(e.target.id)) ucScheduleAuto();
});
