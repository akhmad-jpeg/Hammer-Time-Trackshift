const TYRE_COLORS = {
    Hypersoft:'#ff00ff', Ultrasoft:'#9400d3', Supersoft:'#ff4444',
    Soft:'#ffdd00', Medium:'#dddddd', Hard:'#4fc3f7',
    Intermediate:'#00c853', Wet:'#0091ea'
};
const PIT_COLOR = '#ffa726';

let lapChart = null, tyreChart = null, energyChart = null;

// Energy-card simulator control state: the session whose trace is displayed,
// the mode last simulated for it (highlighted on the SIMULATE bar), and a
// guard so rapid clicks queue one run at a time.
let energySessionId = null;
let energyActiveMode = null;
let simBusy = false;
const ENERGY_MODE_LABELS = { balanced: 'Balanced', push: 'Push', liftcoast: 'Lift & Coast' };

function setEnergyBarState() {
    const bar = document.getElementById('energySimBar');
    if (bar) bar.style.display = energySessionId != null ? 'flex' : 'none';
    [...document.querySelectorAll('#energySimBar .es-btn')].forEach(b => {
        b.disabled = energySessionId == null;
        b.classList.toggle('active', energySessionId != null && b.dataset.mode === energyActiveMode);
    });
}

function energyMsg(text, isErr) {
    const msg = document.getElementById('energySimMsg');
    if (!msg) return;
    msg.textContent = text;
    msg.className = 'es-msg' + (isErr ? ' err' : '');
}

// Re-run the synthetic ERS simulator for the displayed session under one
// strategy and redraw the single tracking line (no page reload).  The API
// replaces the session's race_state rows and returns the fresh trace points
// in the same shape the GET endpoint serves.
async function simEnergy(mode) {
    if (energySessionId == null || simBusy) return;
    simBusy = true;
    setEnergyBarState();
    energyMsg('RUNNING ' + (ENERGY_MODE_LABELS[mode] || mode).toUpperCase() + '...');
    try {
        const res = await fetch(`/api/session/${energySessionId}/energy-simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: mode })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        energyActiveMode = mode;
        setEnergyBarState();
        paintEnergyChart(data.trace, ENERGY_MODE_LABELS[mode]);
        energyMsg((ENERGY_MODE_LABELS[mode] || mode).toUpperCase() + ' \u2192 ' + data.laps +
                  ' laps simulated (' + data.written + ' rows rewritten) in ' + data.elapsed_ms.toFixed(0) + ' ms');
        // The P1 full-race panel listens for this and re-runs when its race
        // uses the session that was just re-simulated (no page reload).
        window.dispatchEvent(new CustomEvent('energy-simulated',
            { detail: { session_id: energySessionId } }));
    } catch (e) {
        energyMsg('Error: ' + e.message, true);
    } finally {
        simBusy = false;
        setEnergyBarState();
    }
}
let currentSessionId = null;
window.addEventListener('energy-simulated', e => {
    if (!lastRaceSim) return;
    const m = lastRaceSim.meta || {};
    const sids = [m.leader && m.leader.session_id, m.chaser && m.chaser.session_id];
    if (e.detail && sids.indexOf(e.detail.session_id) !== -1) {
        runFullRaceSim();
    }
});

// INIT
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadOptions();
    loadSessionsWithCache();
    loadDashboardDrivers();
    loadComparisonYears();
    updateStats();
    setInterval(updateStats, 2000);
    setInterval(updateClocks, 1000);
    updateClocks();
    document.querySelectorAll('#cal-year-row .calendar-toggle-btn').forEach(b =>
        b.classList.toggle('active', parseInt(b.dataset.year, 10) === calYear));
    loadCalendars();   // renders 2020-2026 calendars annotated with DB coverage
    loadOvertakeOptions();
    // Hammer Time context bar: keep the situation summary live.
    ['ov-leader', 'ov-chaser', 'ov-track', 'ov-year', 'ov-lap', 'ov-racelaps', 'ov-gap',
     'ov-ltyre', 'ov-ctyre', 'ov-lage', 'ov-cage', 'ov-ers-batt'].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('change', htUpdateContext);
        el.addEventListener('input', htUpdateContext);
    });
    htUpdateContext();
});
