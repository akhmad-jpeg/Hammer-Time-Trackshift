// DRIVER COMPARISON: RACE DATA
const RC_COLORS = ['#00e5ff', '#ffdd00', '#ff6b6b', '#00c853', '#c792ea', '#ffa726', '#4fc3f7', '#ff8a65'];
let rcChart = null;

function renderRaceLegend(data, codes) {
    const el = document.getElementById('rc-legend');
    if (!el) return;
    el.innerHTML = '<span class="rc-legend-title">DRIVER</span>' +
        codes.map((code, i) => `<span class="rc-leg-item"><i class="rc-leg-line" style="border-top-color:${RC_COLORS[i % RC_COLORS.length]}"></i>${badgeifyDriver(code, data.drivers[code].name)}</span>`).join('') +
        '<span class="rc-leg-item"><i class="rc-leg-dot" style="background:' + PIT_COLOR + '"></i>pit in/out</span>';
}

async function loadComparisonYears() {
    const sel = document.getElementById('rc-year');
    try {
        const res = await fetch('/api/comparison/years');
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        if (!data.years || !data.years.length) { sel.innerHTML = '<option value="">No race data imported yet</option>'; return; }
        sel.innerHTML = '<option value="">Pick a season...</option>' + data.years.map(y => `<option value="${y}">${y}</option>`).join('');
    } catch(e) { sel.innerHTML = `<option value="">Error: ${e.message}</option>`; }
}

async function loadComparisonTracks() {
    const year = document.getElementById('rc-year').value;
    const sel = document.getElementById('rc-track');
    const box = document.getElementById('rc-drivers');
    box.innerHTML = '<div style="color:var(--muted);font-size:0.72em;padding:8px">Pick a track to load drivers...</div>';
    if (!year) { sel.innerHTML = '<option value="">Pick a season first</option>'; return; }
    try {
        const res = await fetch(`/api/comparison/tracks?year=${encodeURIComponent(year)}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        if (!data.tracks || !data.tracks.length) { sel.innerHTML = '<option value="">No races that season</option>'; return; }
        sel.innerHTML = '<option value="">Pick a track...</option>' + data.tracks.map(t => `<option value="${t}">${t}</option>`).join('');
    } catch(e) { sel.innerHTML = `<option value="">Error: ${e.message}</option>`; }
}

async function loadComparisonDrivers() {
    const year = document.getElementById('rc-year').value;
    const track = document.getElementById('rc-track').value;
    const box = document.getElementById('rc-drivers');
    if (!year || !track) { box.innerHTML = '<div style="color:var(--muted);font-size:0.72em;padding:8px">Pick a track to load drivers...</div>'; return; }
    try {
        const res = await fetch(`/api/comparison/drivers?year=${encodeURIComponent(year)}&track=${encodeURIComponent(track)}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        if (!data.drivers || !data.drivers.length) { box.innerHTML = '<div style="color:var(--muted);font-size:0.72em;padding:8px">No drivers found for that race</div>'; return; }
        box.innerHTML = data.drivers.map(d =>
            `<label class="rc-driver" id="rc-drv-${d.code}"><input type="checkbox" value="${d.code}" onchange="toggleRcDriver('${d.code}')"><span>${badgeifyDriver(d.code, d.name)}<br><small>${d.date || '?'} / ${d.laps} laps</small></span></label>`
        ).join('');
    } catch(e) { box.innerHTML = `<div style="color:var(--muted);font-size:0.72em;padding:8px">Error: ${e.message}</div>`; }
}

function toggleRcDriver(code) {
    const lbl = document.getElementById('rc-drv-' + code);
    if (lbl) lbl.classList.toggle('selected', lbl.querySelector('input').checked);
}

async function runRaceCompare() {
    const year = document.getElementById('rc-year').value;
    const track = document.getElementById('rc-track').value;
    const checked = [...document.querySelectorAll('#rc-drivers input:checked')].map(i => i.value);
    const err = document.getElementById('rc-err');
    err.style.display = 'none';
    if (!year || !track) { err.textContent = 'Pick a season and a track first.'; err.style.display = 'block'; return; }
    if (checked.length < 2) { err.textContent = 'Select at least two drivers.'; err.style.display = 'block'; return; }
    try {
        const res = await fetch(`/api/comparison/race?year=${encodeURIComponent(year)}&track=${encodeURIComponent(track)}&drivers=${checked.join(',')}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderRaceCompare(data);
    } catch(e) { err.textContent = 'Error: ' + e.message; err.style.display = 'block'; }
}

function renderRaceCompare(data) {
    const codes = Object.keys(data.drivers);
    const ctx = document.getElementById('rcChart').getContext('2d');
    if (rcChart) rcChart.destroy();
    const datasets = codes.map((code, i) => {
        const color = RC_COLORS[i % RC_COLORS.length];
        const d = data.drivers[code];
        const laps = d.laps;
        return {
            label: `${d.name} (${code})`,
            data: laps.map(l => ({ x: l.lap_number, y: l.lap_time })),
            borderColor: color, backgroundColor: color, borderWidth: 2,
            pointRadius: laps.map(l => l.has_pit_stop ? 6 : 3),
            pointStyle: laps.map(l => l.has_pit_stop ? 'triangle' : 'circle'),
            pointBackgroundColor: laps.map(l => l.has_pit_stop ? PIT_COLOR : color),
            pointBorderColor: laps.map(l => l.has_pit_stop ? PIT_COLOR : color),
            pointBorderWidth: 1, pointHitRadius: 10, pointHoverRadius: 8, spanGaps: false,
        };
    });
    rcChart = new Chart(ctx, {
        type: 'line', data: { datasets },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: items => `Lap ${items[0].parsed.x}`,
                        label: item => {
                            const code = item.dataset.label.split(' (')[1].replace(')', '');
                            const d = data.drivers[code];
                            const lap = d.laps.find(l => l.lap_number === item.parsed.x);
                            if (!lap) return `  ${item.dataset.label}: ${item.parsed.y.toFixed(3)}s`;
                            const pit = lap.has_pit_stop ? ' PIT' : '';
                            const inv = lap.is_valid ? '' : ' (INVALID)';
                            const lines = [`  ${item.dataset.label}: ${item.parsed.y.toFixed(3)}s | ${lap.tyre_compound} age ${lap.tyre_age}${pit}${inv}`];
                            if (lap.telemetry && lap.telemetry.avg_speed != null) {
                                lines.push(`  avg ${lap.telemetry.avg_speed} km/h top ${lap.telemetry.top_speed ?? '---'} gear ${lap.telemetry.avg_gear ?? '---'} ${lap.telemetry.avg_rpm ?? '---'} rpm`);
                            }
                            return lines;
                        }
                    }
                }
            },
            scales: {
                x: { type: 'linear', ...CHART_SCALE, ticks: { ...CHART_SCALE.ticks, stepSize: 5 }, title: { display: true, text: 'RACE LAP', color: '#888', font: { family: "'Share Tech Mono',monospace", size: 10 } } },
                y: { ...CHART_SCALE, beginAtZero: false, title: { display: true, text: 'SECONDS', color: '#888', font: { family: "'Share Tech Mono',monospace", size: 10 } } }
            }
        }
    });

    renderRaceLegend(data, codes);

    const rows = codes.map(code => {
        const d = data.drivers[code];
        const valid = d.laps.filter(l => l.is_valid);
        const fastest = valid.length ? Math.min(...valid.map(l => l.lap_time)) : null;
        const avg = valid.length ? valid.reduce((s, l) => s + l.lap_time, 0) / valid.length : null;
        const tyres = [...new Set(d.laps.map(l => l.tyre_compound).filter(Boolean))].join(', ');
        return { code, name: d.name, laps: d.laps.length, fastest, avg, tyres };
    });
    const bestAvg = Math.min(...rows.filter(r => r.avg).map(r => r.avg));
    const tbody = document.querySelector('#rc-table tbody');
    tbody.innerHTML = rows.map(r =>
        `<tr><td>${badgeifyDriver(r.code, r.name)}</td><td>${r.laps}</td><td>${r.fastest ? fmtTime(r.fastest) : '---'}</td><td>${r.avg ? fmtTime(r.avg) : '---'}</td><td>${r.tyres || '---'}</td></tr>`
    ).join('');
    document.getElementById('rc-table').style.display = 'table';

    const fastestName = rows.find(r => r.avg === bestAvg);
    if (!fastestName) {
        document.getElementById('rc-summary').innerHTML = '<span class="dc-big">No valid laps</span> --- nothing to compare.';
    } else {
        const gapNote = rows.filter(r => r.avg && r !== fastestName).map(r => `${r.code} +${(r.avg - bestAvg).toFixed(3)}s`).join(', ');
        document.getElementById('rc-summary').innerHTML = `<span class="dc-big">${fastestName.name} (${fastestName.code})</span> fastest average valid pace (${fmtTime(bestAvg)}) --- ${gapNote || 'only driver with valid laps'}.`;
    }
    if (data.missing && data.missing.length) {
        document.getElementById('rc-note').textContent = `No race data for: ${data.missing.join(', ')}`;
    }
}

// DRIVER COMPARISON: MODELS
let dcChart = null;
let driverModels = [];

async function loadDriverOptions() {
    const selA = document.getElementById('dc-driver-a');
    const selB = document.getElementById('dc-driver-b');
    const yearSel = document.getElementById('dc-year');
    const prevA = selA.value;
    const prevB = selB.value;
    const prevYear = yearSel.value;
    try {
        const res = await fetch('/api/drivers');
        const drivers = await res.json();
        if (drivers.error || !drivers.length) {
            const msg = drivers.error || 'No per-driver models.';
            selA.innerHTML = `<option value="">${msg}</option>`;
            selB.innerHTML = `<option value="">${msg}</option>`;
            return;
        }
        driverModels = drivers;
        const opts = drivers.map(d => `<option value="${d.code}">${d.name} (${d.code}) --- ${d.tracks.length} tracks, ${d.laps} laps${d.years && d.years.length ? ` / ${d.years.join(', ')} seasons` : ''}</option>`).join('');
        selA.innerHTML = '<option value="">Select driver A...</option>' + opts;
        selB.innerHTML = '<option value="">Select driver B...</option>' + opts;
        const codes = drivers.map(d => d.code);
        if (codes.includes(prevA)) selA.value = prevA;
        if (codes.includes(prevB)) selB.value = prevB;
        updateDcYearOptions();
        if (prevYear && [...yearSel.options].some(o => o.value === prevYear)) yearSel.value = prevYear;
    } catch(e) {
        const msg = 'Error loading drivers: ' + e.message;
        selA.innerHTML = `<option value="">${msg}</option>`;
        selB.innerHTML = `<option value="">${msg}</option>`;
    }
}

function dcYearsFor(code) {
    const d = driverModels.find(m => m.code === code);
    return (d && d.years) ? d.years : [];
}

function updateDcYearOptions() {
    const sel = document.getElementById('dc-year');
    const a = document.getElementById('dc-driver-a').value;
    const b = document.getElementById('dc-driver-b').value;
    const base = '<option value="">All seasons (aggregate)</option>';
    if (!a || !b) { sel.innerHTML = base; return; }
    const common = dcYearsFor(a).filter(y => dcYearsFor(b).includes(y));
    sel.innerHTML = base + common.map(y => `<option value="${y}">${y} season</option>`).join('');
}

function fmtDelta(d, aCode, bCode) {
    const who = d < 0 ? aCode + ' faster' : (d > 0 ? bCode + ' faster' : 'even');
    return `${d >= 0 ? '+' : ''}${d.toFixed(3)}s (${who})`;
}

async function runDriverCompare() {
    const a = document.getElementById('dc-driver-a').value;
    const b = document.getElementById('dc-driver-b').value;
    const year = document.getElementById('dc-year').value;
    const err = document.getElementById('dc-err');
    err.style.display = 'none';
    if (!a || !b) { err.textContent = 'Select both drivers.'; err.style.display = 'block'; return; }
    if (a === b) { err.textContent = 'Pick two different drivers.'; err.style.display = 'block'; return; }
    try {
        const url = `/api/drivers/compare?driver_a=${encodeURIComponent(a)}&driver_b=${encodeURIComponent(b)}` + (year ? `&year=${encodeURIComponent(year)}` : '');
        const res = await fetch(url);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        const s = data.summary;
        const modelsLabel = s.year_used ? `${s.year_used} same-season models` : (year ? `aggregate fallback (no ${year} model for one driver)` : 'aggregate all-seasons models');
        let summaryHtml;
        if (s.avg_delta == null || s.shared_tracks === 0) {
            summaryHtml = `<span class="dc-big">No shared tracks</span> --- ${data.driver_a.name} and ${data.driver_b.name} have no (track, tyre) pair in common.`;
        } else {
            summaryHtml = `<span class="dc-big">${s.faster_name} (${s.faster_code})</span> is on average <b>${Math.abs(s.avg_delta).toFixed(3)}s/lap</b> faster across ${s.shared_tracks} shared track(s) --- using <b>${modelsLabel}</b>. Delta = ${data.driver_a.code} - ${data.driver_b.code}.`;
        }
        const sig = s.significance;
        if (sig && sig.label && s.avg_delta != null) {
            const pTxt = sig.p_value != null ? `p=${sig.p_value.toFixed(3)}` : 'n/a';
            const nTxt = `n=${sig.n} track${sig.n === 1 ? '' : 's'}`;
            if (sig.label === 'significant') summaryHtml += `<div class="dc-sig dc-sig-good">Statistically significant (${pTxt}, ${nTxt}) --- ${sig.note}</div>`;
            else if (sig.label === 'suggestive') summaryHtml += `<div class="dc-sig dc-sig-warn">Suggestive (${pTxt}, ${nTxt}) --- ${sig.note}</div>`;
            else if (sig.label === 'inconclusive') summaryHtml += `<div class="dc-sig dc-sig-bad">Within model noise (${pTxt}, ${nTxt}) --- ${sig.note}</div>`;
            else summaryHtml += `<div class="dc-sig dc-sig-mute">--- ${sig.note}</div>`;
        }
        document.getElementById('dc-summary').innerHTML = summaryHtml;
        document.getElementById('dc-note').textContent = 'Green bar = ' + data.driver_a.code + ' faster / Red = ' + data.driver_b.code + ' faster.';

        const tracks = data.per_track.map(t => t.track);
        const deltas = data.per_track.map(t => t.avg_delta);
        const colors = deltas.map(d => d <= 0 ? '#2ecc71' : '#ff6b6b');
        const ctx = document.getElementById('dcChart').getContext('2d');
        if (dcChart) dcChart.destroy();
        dcChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: tracks, datasets: [{ label: 'Avg Delta (s)', data: deltas, backgroundColor: colors }] },
            options: {
                indexAxis: 'y', responsive: true, maintainAspectRatio: false, animation: false,
                plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => fmtDelta(c.parsed.x, data.driver_a.code, data.driver_b.code) } } },
                scales: {
                    x: { ...CHART_SCALE, title: { display: true, text: 'SECONDS', color: '#888', font: { family: "'Share Tech Mono',monospace", size: 10 } } },
                    y: { ...CHART_SCALE }
                }
            }
        });

        const tbody = document.querySelector('#dc-table tbody');
        tbody.innerHTML = data.per_track.map(t => {
            const who = t.avg_delta < 0 ? data.driver_a.code : (t.avg_delta > 0 ? data.driver_b.code : '---');
            return `<tr><td>${t.track}</td><td>${t.avg_delta >= 0 ? '+' : ''}${t.avg_delta.toFixed(3)}</td><td>${who}</td></tr>`;
        }).join('');
        document.getElementById('dc-table').style.display = 'table';
    } catch(e) { err.textContent = 'Error: ' + e.message; err.style.display = 'block'; }
}

