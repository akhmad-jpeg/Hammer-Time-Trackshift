function fmtTime(s) {
    if (!s || isNaN(s) || s === Infinity) return '---';
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(3).padStart(6,'0');
    return `${m}:${sec}`;
}

const CHART_SCALE = {
    ticks:{ color:'#555', font:{ family:"'Share Tech Mono', monospace", size:10 } },
    grid:{ color:'rgba(255,255,255,0.04)' }
};

let _currentLapsByNum = {};
let _currentFastestTime = null;

function initCharts() {
    const lapCtx = document.getElementById('lapChart').getContext('2d');
    lapChart = new Chart(lapCtx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: {
                legend: { display: true, labels: { color: '#f0f0f0', font: { family: "'Share Tech Mono',monospace", size: 11 } } },
                tooltip: {
                    callbacks: {
                        title: items => items.length ? `Race Lap ${items[0].label}` : '',
                        label: ctx => {
                            if (ctx.parsed.y === null || ctx.parsed.y === undefined) return '';
                            const lapNum = ctx.label;
                            const lap = _currentLapsByNum[lapNum];
                            const timeStr = fmtTime(ctx.parsed.y);
                            const isFastest = (typeof _currentFastestTime === 'number') && Math.abs(ctx.parsed.y - _currentFastestTime) < 0.0001;
                            const tag = isFastest ? ' FASTEST' : '';
                            if (lap && lap.tyre_compound) {
                                const ageStr = (lap.tyre_age !== undefined && lap.tyre_age !== null) ? ` (Age: ${lap.tyre_age})` : '';
                                return `  ${ctx.dataset.label}: ${timeStr}${tag} | ${lap.tyre_compound}${ageStr}`;
                            }
                            return `  ${ctx.dataset.label}: ${timeStr}${tag}`;
                        }
                    }
                }
            },
            scales: {
                x: { ...CHART_SCALE, title: { display: true, text: 'RACE LAP', color: '#888', font: { family: "'Share Tech Mono',monospace", size: 10 } } },
                y: { ...CHART_SCALE, beginAtZero: false, title: { display: true, text: 'SECONDS', color: '#888', font: { family: "'Share Tech Mono',monospace", size: 10 } } }
            }
        }
    });

    const tyreCtx = document.getElementById('tyreChart').getContext('2d');
    // Gold pit-window band: teams pit with ~40-50% of a set's life left
    // (1-2 laps before the performance cliff), so the band marks where a
    // stint should end on the health axis.  Drawn as a custom plugin — the
    // CDN build has no annotation plugin loaded.
    const pitWindowBand = {
        id: 'pitWindowBand',
        beforeDatasetsDraw(chart, args, opts) {
            const y = chart.scales.y;
            const area = chart.chartArea;
            if (!y || !area) return;
            const yTop = y.getPixelForValue(50);
            const yBot = y.getPixelForValue(40);
            const ctx = chart.ctx;
            ctx.save();
            ctx.fillStyle = 'rgba(255, 215, 0, 0.07)';
            ctx.fillRect(area.left, yTop, area.width, yBot - yTop);
            ctx.fillStyle = 'rgba(255, 215, 0, 0.55)';
            ctx.font = "9px 'Share Tech Mono', monospace";
            ctx.textAlign = 'left';
            ctx.fillText('PIT WINDOW', area.left + 4, yBot - 3);
            ctx.restore();
        }
    };
    tyreChart = new Chart(tyreCtx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        plugins: [pitWindowBand],
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: {
                legend: { display: true, labels: { color: '#f0f0f0', font: { family: "'Share Tech Mono',monospace", size: 11 } } },
                tooltip: {
                    callbacks: {
                        title: items => items.length ? `Race Lap ${items[0].label}` : '',
                        label: ctx => {
                            if (ctx.parsed.y === null || ctx.parsed.y === undefined) return '';
                            const lap = _currentLapsByNum[ctx.label];
                            const health = ctx.parsed.y;
                            let comp = '', ageStr = '';
                            if (lap) {
                                if (lap.tyre_compound) comp = ` ${lap.tyre_compound}`;
                                if (lap.tyre_age !== undefined && lap.tyre_age !== null) {
                                    ageStr = ` · age ${lap.tyre_age} lap${lap.tyre_age === 1 ? '' : 's'}`;
                                }
                            }
                            const scrub = (lap && lap.is_valid !== undefined && lap.is_valid != 1) ? ' scrubbed' : '';
                            const pitTag = (lap && lap.is_pit_lap) ? ((lap.has_pit_stop == 1) ? ' · pit in' : ' · pit out') : '';
                            return `  Tyre Health ${health.toFixed(0)}%${comp}${ageStr}${scrub}${pitTag}`;
                        }
                    }
                }
            },
            scales: {
                x: { ...CHART_SCALE, title: { display: true, text: 'RACE LAP', color: '#888', font: { family: "'Share Tech Mono',monospace", size: 10 } } },
                y: { ...CHART_SCALE, beginAtZero: true, title: { display: true, text: 'TYRE HEALTH (%)', color: '#888', font: { family: "'Share Tech Mono',monospace", size: 10 } } }
            }
        }
    });
}

let _sessionsCache = [];

function getTrackName(sessionId) {
    const cached = _sessionsCache.find(s => String(s.session_id) === String(sessionId));
    return cached ? cached.track_name : '---';
}

function showSessionInfo({ track, fastest, lastLap, tyre }, sessionId) {
    const trackName = track || getTrackName(sessionId);
    setStatVal('info-track', trackName);
    if (fastest) setStatVal('info-fastest', fastest);
    if (lastLap) setStatVal('info-lastlap', lastLap);
    if (tyre) setStatVal('info-tyre', tyre);
}

async function loadSession(id) {
    currentSessionId = id;
    await Promise.all([loadLapChart(id), loadTyreChart(id), loadEnergyChart(id)]);
    updateCircuitBanner(id);
    const cached = _sessionsCache.find(s => String(s.session_id) === String(id));
    if (cached) highlightCalendarRow(cached.track_name);
}

function paintEnergyChart(rows, modeLabel) {
    const noData = document.getElementById('energyNoData');
    const note = document.getElementById('energyChartNote');
    const canvas = document.getElementById('energyChart');
    if (!rows || !rows.length) {
        noData.textContent = 'NO ENERGY TRACE — pick a strategy on the SIMULATE bar above to generate one';
        noData.style.display = 'flex';
        canvas.style.display = 'none';
        if (note) note.style.display = 'none';
        return;
    }
    noData.style.display = 'none';
    canvas.style.display = 'block';
    if (note) note.style.display = 'block';

    if (!energyChart) {
            energyChart = new Chart(canvas.getContext('2d'), {
                type: 'line',
                data: { datasets: [] },
                options: {
                    responsive: true, maintainAspectRatio: false, animation: false,
                    spanGaps: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                title: () => '',
                                label: ctx => {
                                    if (ctx.dataset.label && ctx.dataset.label.startsWith('uncertainty')) return null;
                                    const r = rows[ctx.dataIndex];
                                    const b = r && r.band_pct != null ? ' ±' + r.band_pct.toFixed(1) + '%' : '';
                                    return 'LAP ' + Math.floor(ctx.parsed.x) + ' · battery ' + ctx.parsed.y.toFixed(1) + '%' + b + ' (synthetic)';
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear', min: 0.5,
                            grid: { color: 'rgba(255,255,255,0.04)' },
                            ticks: { color: '#555', font: { family: "'Share Tech Mono', monospace", size: 10 }, maxTicksLimit: 24, callback: v => (Number.isInteger(v) && v >= 1) ? v : '' },
                            title: { display: true, text: 'RACE LAP', color: '#888', font: { family: "'Share Tech Mono',monospace", size: 10 } }
                        },
                        y: { ...CHART_SCALE, min: 0, max: 100, title: { display: true, text: 'BATTERY %  ·  100% = 4 MJ (1% = 0.04 MJ)', color: '#ffd700', font: { family: "'Share Tech Mono',monospace", size: 10 } } }
                    }
                }
            });
        }

        // Single battery-% tracking line at intra-lap (telemetry-sample)
        // resolution: falls as energy is deployed, rises as it is harvested.
        // Dashed guides at 30% (1.2 MJ) and 80% (3.2 MJ) mark a *soft* SOC
        // band: steady laps mostly cycle inside it, but the line can leak
        // out -- faster-than-average laps drain it below 30%, slower laps
        // bank it above 80%. The battery starts the race full (100%).
        const xs = rows.map(r => r.x);
        const xMin = Math.max(0.5, Math.min(...xs) - 0.5);
        const xMax = Math.max(...xs) + 0.5;
        // SOC uncertainty envelope: the battery is a SYNTHESIZED estimate,
        // so each point carries a ± band (floor ±2% at the simulator write,
        // +0.5%/lap of drift, capped ±8% — energy_simulator.
        // battery_uncertainty_band).  Drawn as two shaded boundaries around
        // the tracking line: the estimate is honest about what it does not
        // know, growing dimmer as it drifts from the last anchor.
        const hasBand = rows.some(r => r.band_pct != null);
        const envelope = hasBand ? [
            {
                label: 'uncertainty +band',
                data: rows.map(r => ({ x: r.x, y: Math.min(100, r.battery_pct + (r.band_pct || 0)) })),
                borderColor: 'rgba(255,215,0,0.18)', backgroundColor: 'rgba(255,215,0,0.05)',
                borderWidth: 1, pointRadius: 0, tension: 0.15,
                fill: '-1', spanGaps: true
            },
            {
                label: 'uncertainty −band',
                data: rows.map(r => ({ x: r.x, y: Math.max(0, r.battery_pct - (r.band_pct || 0)) })),
                borderColor: 'rgba(255,215,0,0.18)', backgroundColor: 'rgba(255,215,0,0.10)',
                borderWidth: 1, pointRadius: 0, tension: 0.15,
                fill: false, spanGaps: true
            }
        ] : [];
        energyChart.data.datasets = [
            ...envelope,
            {
                label: (modeLabel ? modeLabel + ' · Battery %' : 'Battery %'),
                data: rows.map(r => ({ x: r.x, y: r.battery_pct })),
                borderColor: '#ffd700', backgroundColor: 'rgba(255,215,0,0.08)',
                borderWidth: 1.5, pointRadius: 0, tension: 0.15, fill: hasBand ? false : true
            },
            {
                label: 'soft band floor 30% (1.2 MJ)',
                data: [{ x: xMin, y: 30 }, { x: xMax, y: 30 }],
                borderColor: 'rgba(0,210,190,0.55)', borderDash: [6, 4],
                borderWidth: 1, pointRadius: 0, fill: false
            },
            {
                label: 'soft band ceiling 80% (3.2 MJ)',
                data: [{ x: xMin, y: 80 }, { x: xMax, y: 80 }],
                borderColor: 'rgba(255,107,107,0.55)', borderDash: [6, 4],
                borderWidth: 1, pointRadius: 0, fill: false
            }
        ];
        energyChart.resize();
        energyChart.update();
}

async function loadEnergyChart(id) {
    energySessionId = id;
    setEnergyBarState();
    try {
        const res = await fetch(`/api/session/${id}/energy`);
        const rows = await res.json();
        if (rows.error) throw new Error(rows.error);
        // The stored race_state may have been simulated under any strategy,
        // so no SIMULATE button is highlighted until the user re-runs one.
        energyActiveMode = null;
        paintEnergyChart(rows, null);
    } catch (e) {
        const noData = document.getElementById('energyNoData');
        const canvas = document.getElementById('energyChart');
        const note = document.getElementById('energyChartNote');
        if (noData) { noData.textContent = 'ERROR: ' + e.message; noData.style.display = 'flex'; }
        if (canvas) canvas.style.display = 'none';
        if (note) note.style.display = 'none';
    }
    setEnergyBarState();
}

async function loadLapChart(id) {
    const noData = document.getElementById('lapNoData');
    const canvas = document.getElementById('lapChart');
    try {
        const res = await fetch(`/api/session/${id}/laps`);
        const laps = await res.json();
        if (laps.error) throw new Error(laps.error);

        if (!laps || !laps.length) {
            noData.textContent = 'NO LAP DATA';
            noData.style.display = 'flex';
            canvas.style.display = 'none';
            showSessionInfo({ track: getTrackName(id), fastest: '---', lastLap: '---', tyre: '---' }, id);
            return;
        }

        const validLaps = laps.filter(l => (l.is_valid == 1 || l.is_valid === true) && parseFloat(l.lap_time) > 0);
        if (!validLaps.length) {
            noData.textContent = 'NO VALID LAPS';
            noData.style.display = 'flex';
            canvas.style.display = 'none';
            showSessionInfo({ track: getTrackName(id), fastest: '---', lastLap: '---', tyre: '---' }, id);
            return;
        }

        noData.style.display = 'none';
        canvas.style.display = 'block';

        const validTimes = validLaps.map(l => parseFloat(l.lap_time));
        const fastest = Math.min(...validTimes);
        _currentFastestTime = fastest;
        const lastLap = validLaps[validLaps.length - 1];
        const lastTime = parseFloat(lastLap.lap_time);

        const tyresUsed = [...new Set(validLaps.map(l => l.tyre_compound).filter(Boolean))];
        const tyreStr = tyresUsed.join(' / ') || '---';

        showSessionInfo({ track: getTrackName(id), fastest: fmtTime(fastest), lastLap: fmtTime(lastTime), tyre: tyreStr }, id);

        laps.forEach(l => { _currentLapsByNum[l.lap_number] = l; });

        const lapNumbers = laps.map(l => parseInt(l.lap_number)).filter(n => !isNaN(n));
        const minLap = Math.min(...lapNumbers);
        const maxLap = Math.max(...lapNumbers);
        const allLapNumbers = [];
        for (let i = minLap; i <= maxLap; i++) allLapNumbers.push(i);

        const numericData = allLapNumbers.map(num => {
            const l = _currentLapsByNum[num];
            if (l && (l.is_valid == 1 || l.is_valid === true) && parseFloat(l.lap_time) > 0) return parseFloat(l.lap_time);
            return null;
        });

        const pointColors = numericData.map(val => (val !== null && Math.abs(val - fastest) < 0.0001) ? '#ffd700' : '#00d2be');
        const pointRadii  = numericData.map(val => (val !== null && Math.abs(val - fastest) < 0.0001) ? 7 : 4);

        lapChart.data.labels = allLapNumbers;
        lapChart.data.datasets = [{
            label: 'Lap Time', data: numericData, borderColor: '#00d2be', backgroundColor: 'rgba(0,210,190,0.05)',
            borderWidth: 2, pointRadius: pointRadii, pointBackgroundColor: pointColors, pointBorderColor: pointColors,
            pointHitRadius: 10, pointHoverRadius: 8, tension: 0.3, fill: true, spanGaps: true
        }];
        lapChart.resize();
        lapChart.update();
    } catch(e) {
        noData.textContent = `ERROR: ${e.message}`;
        noData.style.display = 'flex';
        canvas.style.display = 'none';
    }
}

async function loadTyreChart(id) {
    const noData = document.getElementById('tyreNoData');
    const canvas = document.getElementById('tyreChart');
    try {
        const res = await fetch(`/api/session/${id}/tyre-degradation`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        if (!data || !data.length) {
            noData.textContent = 'NO TYRE DATA';
            noData.style.display = 'flex';
            canvas.style.display = 'none';
            document.getElementById('tyreChartNote').style.display = 'none';
            return;
        }

        noData.style.display = 'none';
        canvas.style.display = 'block';
        const legendEl = document.getElementById('tyreLegend');
        if (legendEl) legendEl.style.display = 'flex';
        document.getElementById('tyreChartNote').style.display = 'block';

        const laps = data.slice().sort((a, b) => (parseInt(a.lap_number) || 0) - (parseInt(b.lap_number) || 0));
        laps.forEach(l => { _currentLapsByNum[l.lap_number] = l; });

        const lapNumbers = laps.map(l => parseInt(l.lap_number)).filter(n => !isNaN(n));
        const minLap = Math.min(...lapNumbers);
        const maxLap = Math.max(...lapNumbers);
        const allLapNumbers = [];
        for (let i = minLap; i <= maxLap; i++) allLapNumbers.push(i);

        const stints = [];
        let currentStint = null;
        let stintCount = 0;

        for (let i = 0; i < laps.length; i++) {
            const lap = laps[i];
            const prevLap = i > 0 ? laps[i - 1] : null;
            const isNewStint = !currentStint || (prevLap && (
                prevLap.has_pit_stop == 1 ||
                (lap.tyre_compound && prevLap.tyre_compound && lap.tyre_compound !== prevLap.tyre_compound) ||
                (lap.tyre_age !== null && prevLap.tyre_age !== null && lap.tyre_age < prevLap.tyre_age)
            ));
            if (isNewStint) {
                stintCount++;
                currentStint = { stintNumber: stintCount, compound: lap.tyre_compound || 'Unknown', lapMap: {} };
                stints.push(currentStint);
            }
            currentStint.lapMap[lap.lap_number] = lap;
        }

        // Build one continuous dataset with per-point color/style
        // so the line connects across stint boundaries (gaps only at
        // null laps, not between datasets).
        const allData = [];
        const allStyles = [];
        const allRadii = [];
        const allBg = [];
        const allBorder = [];
        let lapColor = '#00d2be';
        allLapNumbers.forEach(num => {
            let lap = null;
            for (const stint of stints) { if (stint.lapMap[num]) { lap = stint.lapMap[num]; lapColor = TYRE_COLORS[stint.compound] || '#00d2be'; break; } }
            if (lap) {
                const health = parseFloat(lap.tyre_health_pct);
                allData.push(isNaN(health) ? null : health);
                allStyles.push(lap.is_pit_lap ? 'rectRot' : 'circle');
                allRadii.push(lap.is_pit_lap ? 8 : 4);
                allBg.push(lap.is_pit_lap ? PIT_COLOR : lapColor);
            } else {
                allData.push(null);
                allStyles.push('circle');
                allRadii.push(4);
                allBg.push(lapColor);
            }
        });
        const datasets = [{
            label: 'Tyre Health', borderColor: '#00d2be', backgroundColor: 'transparent',
            segment: { borderColor: (ctx) => { const n = allLapNumbers[ctx.p0DataIndex]; let c = '#00d2be'; for (const s of stints) { if (s.lapMap[n]) { c = TYRE_COLORS[s.compound] || '#00d2be'; break; } } return c; } },
            // spanGaps:false so laps with no tyre data (health null) render
            // as TRUE gaps instead of a bridged line — a hole in the tyre
            // record must not look like rubber that stopped degrading (the
            // fake flat lines from missing compound/age rows).
            borderWidth: 2, pointStyle: allStyles, pointRadius: allRadii, pointBackgroundColor: allBg,
            pointBorderColor: allBg, pointHitRadius: 10, pointHoverRadius: 8, tension: 0.3, spanGaps: false, data: allData
        }];

        tyreChart.data.labels = allLapNumbers;
        tyreChart.data.datasets = datasets;

        // Fixed 0-100 health scale — health is bounded by construction, so no
        // per-session clamping is needed; pit diamonds sit at their true
        // health like every other lap (a stop resets the curve to ~100%).
        tyreChart.options.scales.y.min = 0;
        tyreChart.options.scales.y.max = 100;

        tyreChart.resize();
        tyreChart.update();
    } catch(e) {
        noData.textContent = `ERROR: ${e.message}`;
        noData.style.display = 'flex';
        canvas.style.display = 'none';
        document.getElementById('tyreChartNote').style.display = 'none';
    }
}

// LIVE STATS POLLING
async function updateStats() {
    try {
        const url = selectedDashboardDriver
            ? `/api/latest-lap?driver=${encodeURIComponent(selectedDashboardDriver)}`
            : '/api/latest-lap';
        const res = await fetch(url);
        const lap = await res.json();
        if (lap && lap.live && lap.lap_time) {
            const t = parseFloat(lap.lap_time);
            setStatVal('stat-laptime', fmtTime(t));
            setStatVal('stat-track', lap.track_name || '---');
            setStatVal('stat-tyre', lap.tyre_compound || '---');
            setStatVal('stat-lap', lap.lap_number || '---');
            if (t < sessionFastestMs) { sessionFastestMs = t; setStatVal('stat-fastest', fmtTime(t)); }
            if (!currentSessionId) {
                setStatVal('info-track', lap.track_name || '---');
                setStatVal('info-lastlap', fmtTime(t));
                if (sessionFastestMs < Infinity) setStatVal('info-fastest', fmtTime(sessionFastestMs));
                if (lap.tyre_compound) setStatVal('info-tyre', lap.tyre_compound);
            }
            setLiveBadge(true);
        } else {
            setStatVal('stat-laptime', '---');
            setStatVal('stat-fastest', '---');
            setStatVal('stat-track', '---');
            setStatVal('stat-tyre', '---');
            setStatVal('stat-lap', '---');
            sessionFastestMs = Infinity;
            if (!currentSessionId) {
                setStatVal('info-track', '---');
                setStatVal('info-lastlap', '---');
                setStatVal('info-fastest', '---');
                setStatVal('info-tyre', '---');
            }
            setLiveBadge(false);
        }
    } catch(e) {}
}

function setLiveBadge(live) {
    const badge = document.querySelector('.f1-live-badge');
    if (!badge) return;
    const text = badge.querySelector('.f1-live-text');
    if (text) text.textContent = live ? 'LIVE TELEMETRY' : 'STANDBY';
    badge.classList.toggle('standby', !live);
}

function setStatVal(id, val) {
    const el = document.getElementById(id);
    if (el && el.textContent !== String(val)) {
        el.textContent = val;
        el.classList.remove('flash');
        void el.offsetWidth;
        el.classList.add('flash');
    }
}

// MODEL OPTIONS
async function loadOptions() {
    try {
        const res = await fetch('/api/predict/options');
        if (!res.ok) throw new Error('Model not loaded');
        const opts = await res.json();
        if (opts.error) throw new Error(opts.error);
        buildTyreGrid('p-tyre-grid', opts.tyres, 'predict');
        buildTrackGrid('p-track-grid', opts.tracks, 'predict');
        buildTyreGrid('s-tyre-grid', opts.tyres, 'strategy');
        buildTrackGrid('s-track-grid', opts.tracks, 'strategy');
    } catch(e) {
        ['p-tyre-grid','p-track-grid','s-tyre-grid','s-track-grid'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = `<div style="color:#ff6666;font-family:'Share Tech Mono',monospace;font-size:0.7em;padding:8px">${e.message}</div>`;
        });
    }
}

function buildTyreGrid(gridId, tyres, scope) {
    const grid = document.getElementById(gridId);
    if (!grid) return;
    grid.innerHTML = '';
    tyres.forEach(t => {
        const btn = document.createElement('button');
        btn.className = 'opt-btn';
        const color = TYRE_COLORS[t.name] || '#aaa';
        btn.innerHTML = `<span class="tyre-dot" style="background:${color}"></span>${t.name}<span class="opt-tag">${t.type}</span>`;
        btn.onclick = () => { grid.querySelectorAll('.opt-btn').forEach(b => b.classList.remove('sel')); btn.classList.add('sel'); if (scope === 'predict') selectedPTyre = t.name; else selectedSTyre = t.name; };
        grid.appendChild(btn);
    });
}

function buildTrackGrid(gridId, tracks, scope) {
    const grid = document.getElementById(gridId);
    if (!grid) return;
    grid.innerHTML = '';
    tracks.forEach(t => {
        const btn = document.createElement('button');
        btn.className = 'opt-btn';
        btn.textContent = t;
        btn.onclick = () => { grid.querySelectorAll('.opt-btn').forEach(b => b.classList.remove('sel')); btn.classList.add('sel'); if (scope === 'predict') selectedPTrack = t; else selectedSTrack = t; };
        grid.appendChild(btn);
    });
}

// LAP PREDICTOR
async function runPredict() {
    const btn = document.getElementById('predict-btn');
    const err = document.getElementById('predict-err');
    err.style.display = 'none';
    const age = document.getElementById('p-age').value;
    const lapNumber = document.getElementById('p-lap-number').value;
    if (!age || !lapNumber || !selectedPTyre || !selectedPTrack) {
        err.textContent = 'Fill all fields and select a tyre + track.';
        err.style.display = 'block'; return;
    }
    btn.disabled = true; btn.textContent = 'COMPUTING...';
    try {
        const res = await fetch('/api/predict', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ tyre_age:parseFloat(age), lap_number:parseInt(lapNumber, 10), tyre_compound:selectedPTyre, track_name:selectedPTrack })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        document.getElementById('result-empty').style.display = 'none';
        document.getElementById('result-content').classList.add('show');
        const color = TYRE_COLORS[data.tyre_compound] || '#00d2be';
        const rt = document.getElementById('result-time');
        rt.textContent = data.formatted; rt.style.color = color; rt.style.textShadow = `0 0 28px ${color}55`;
        document.getElementById('result-secs').textContent = data.predicted_time.toFixed(3) + ' seconds';
        document.getElementById('res-track').textContent = data.track;
        document.getElementById('res-tyre').innerHTML = `<span class="tyre-dot-lg" style="background:${color}"></span>${data.tyre_compound}`;
        document.getElementById('res-age').textContent = data.tyre_age + ' laps';
        document.getElementById('res-fuel').textContent = data.fuel_load.toFixed(1) + ' kg';
    } catch(e) {
        err.textContent = 'Error: ' + e.message; err.style.display = 'block';
    }
    btn.disabled = false; btn.textContent = 'PREDICT LAP TIME';
}

// STRATEGY ADVISOR
function selectEvent(ev) {
    document.querySelectorAll('.event-btn').forEach(b => b.classList.remove('sel'));
    document.getElementById('ev-' + ev)?.classList.add('sel');
    selectedEvent = ev;
}

async function runStrategy() {
    const err = document.getElementById('strategy-err');
    err.style.display = 'none';
    const curLap = document.getElementById('s-curlap').value;
    const totLaps = document.getElementById('s-totlaps').value;
    const age = document.getElementById('s-age').value;
    if (!curLap || !totLaps || !age || !selectedSTyre || !selectedSTrack) {
        err.textContent = 'Fill all fields, select tyre + track.'; err.style.display = 'block'; return;
    }
    if (!selectedEvent) {
        err.textContent = 'Select a race event (VSC, Safety Car, Rain, Crash).'; err.style.display = 'block'; return;
    }
    const sessionId = document.getElementById('s-session').value;
    const driver = document.getElementById('s-driver').value.trim();
    const gap = document.getElementById('s-gap').value;
    const traffic = document.getElementById('s-traffic').value;
    const year = document.getElementById('s-year').value;
    const body = { current_lap:parseInt(curLap), total_laps:parseInt(totLaps), current_tyre:selectedSTyre, current_age:parseInt(age), track:selectedSTrack, event_type:selectedEvent, traffic };
    if (sessionId) body.session_id = parseInt(sessionId, 10);
    if (driver) body.driver = driver;
    if (gap !== '') body.gap_to_ahead = parseFloat(gap);
    if (year) body.year = parseInt(year, 10);
    try {
        const res = await fetch('/api/strategy/analyze', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        showStrategyResult(data);
    } catch(e) {
        err.textContent = 'Error: ' + e.message; err.style.display = 'block';
    }
}

function showStrategyResult(data) {
    document.getElementById('strat-empty').style.display = 'none';
    document.getElementById('strat-result').classList.add('show');
    // Fresh-set inventory chips (from the driver's real stint history when a
    // session + driver were supplied, else the full weekend allocation).
    const inv = data.inventory || {};
    const invChips = Object.keys(inv).map(c => {
        const v = inv[c];
        const col = TYRE_COLORS[c] || '#aaa';
        const gone = v.fresh <= 0;
        return `<span style="display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;border:1px solid ${gone ? '#553' : '#2a4a44'};border-radius:4px;font-family:'Share Tech Mono',monospace;font-size:0.68em;color:${gone ? '#998' : '#7fffe0'};background:rgba(0,0,0,0.3)"><span class="tyre-dot" style="background:${col}"></span>${c} ×${v.fresh}<span style="opacity:0.5"> / ${v.allocation}</span></span>`;
    }).join('');
    // Track-position / undercut / overcut note
    const pos = data.position || {};
    const posHtml = pos.note
        ? `<div class="strat-desc" style="margin-top:8px;padding:8px 10px;border-left:3px solid ${pos.undercut_open ? '#ffd700' : pos.overcut_open ? '#00d2be' : '#556'};">${pos.note}</div>`
        : '';
    document.getElementById('strat-options').innerHTML =
        (invChips ? `<div style="font-family:'Share Tech Mono',monospace;font-size:0.68em;letter-spacing:1px;color:#9aa;margin-bottom:6px">FRESH SETS LEFT ${data.current_situation && data.current_situation.traffic ? '· rejoin ' + data.current_situation.traffic + ' traffic' : ''}</div><div style="margin-bottom:10px">${invChips}</div>` : '') +
        data.strategies.map((s, i) => {
        const mins = Math.floor(s.total_time / 60);
        const secs = (s.total_time % 60).toFixed(1);
        const riskColor = s.risk === 'Low' ? '#00ff88' : s.risk === 'High' ? '#ff4444' : '#ffdd00';
        return `<div class="strat-option${i === 0 ? ' best' : ''}"><div class="strat-badge">RECOMMENDED</div><div class="strat-opt-title">${s.option}</div><div class="strat-desc">${s.description}</div><div class="strat-stats"><div>TIME <span>${mins}m ${secs}s</span></div><div>STOPS <span>${s.pit_stops}</span></div><div>RISK <span style="color:${riskColor}">${s.risk}</span></div></div></div>`;
    }).join('') + posHtml;
    document.getElementById('strat-rec').innerHTML = `<div class="strat-rec-title">Recommendation</div><div class="strat-rec-action">${data.recommendation.action}</div><div class="strat-rec-reason">${data.recommendation.reason}</div>`;
}

// ENERGY MODES
async function runEnergyCompare() {
    const err = document.getElementById('energy-err');
    err.style.display = 'none';
    const sid = parseInt(document.getElementById('e-session').value, 10);
    const lap = parseInt(document.getElementById('e-lap').value, 10);
    const battRaw = document.getElementById('e-batt').value;
    if (isNaN(sid) || isNaN(lap) || sid <= 0 || lap <= 0) {
        err.textContent = 'Enter a session id and a current lap.';
        err.style.display = 'block'; return;
    }
    const body = { session_id: sid, current_lap: lap };
    if (battRaw && !isNaN(parseFloat(battRaw))) body.battery_mj = parseFloat(battRaw);
    const btn = [...document.querySelectorAll('button')].find(b => /ANALYZE ENERGY MODES/.test(b.innerText || ''));
    if (btn) { btn.disabled = true; btn.textContent = 'PROJECTING...'; }
    try {
        const res = await fetch('/api/strategy/energy-analyze', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        const out = document.getElementById('energy-whatif-out');
        const cur = data.current;
        const cfg = data.config;
        const ordered = [...data.modes].sort((a, b) => (b.feasible - a.feasible) || (a.total_time_s - b.total_time_s));
        const fastestT = ordered.length ? ordered[0].total_time_s : null;
        const riskTxt = m => !m.feasible
            ? '<span style="color:#ff6b6b">HIGH</span>'
            : (m.limited_laps > 0 ? '<span style="color:#ffd700">MEDIUM</span>' : '<span style="color:#00c853">LOW</span>');
        const ECOLS = 'grid-template-columns:1.2fr 1.1fr 1fr 0.6fr;';
        const modeLabel = k => ({ push: 'PUSH', balanced: 'BALANCED', liftcoast: 'LIFT & COAST' }[k] || k.toUpperCase());
        const rows = ordered.map(m => {
            const color = m.mode === 'push' ? '#ff6b6b' : (m.mode === 'balanced' ? '#ffd700' : '#00c853');
            const dt = (fastestT != null && m.total_time_s > fastestT + 1e-9)
                ? '+' + (m.total_time_s - fastestT).toFixed(1) + 's'
                : 'Fastest';
            return `<div class="tbl-row" style="${ECOLS}">` +
                `<span style="color:${color}"><b>${modeLabel(m.mode)}</b>${m.mode === data.recommendation.mode ? ' <span class="pass-chip">RECOMMENDED</span>' : ''}</span>` +
                `<span>${dt} <span style="color:#667">(${fmtTime(m.total_time_s)})</span></span>` +
                `<span${m.final_battery_pct < 30 ? ' style="color:#ff6b6b;font-weight:700"' : ''}>${m.final_battery_pct.toFixed(1)}%${m.feasible ? '' : ' ⚠ BATTERY RISK'}</span>` +
                `<span>${riskTxt(m)}</span></div>`;
        }).join('');
        out.innerHTML =
            `<div class="chart-note" style="margin:14px 0 4px">${data.session.track} · ${data.session.date} — lap ${cur.lap}, ${cur.tyre} age ${cur.tyre_age}, battery ${cur.battery_mj.toFixed(1)} MJ (${cur.battery_pct.toFixed(1)}%), ${data.laps_remaining} laps remaining</div>` +
            `<div class="tbl-wrap" style="margin-top:8px">` +
            `<div class="tbl-row tbl-head" style="${ECOLS}"><span>STRATEGY</span><span>RACE TIME (LEFT)</span><span>BATTERY AT FLAG</span><span>RISK</span></div>` +
            rows +
            `</div>` +
            `<div class="strat-recommend" style="margin-top:14px;display:block"><div class="strat-rec-title">RECOMMENDATION</div><div class="strat-rec-action">${modeLabel(data.recommendation.mode)}${data.recommendation.feasible ? '' : ' (survivor)'}</div><div class="strat-rec-reason">${data.recommendation.reason}</div></div>` +
            `<details class="ht-hint" style="margin-top:12px"><summary>ENGINEERING DETAILS — projection assumptions &amp; per-mode totals</summary>` +
            `<div class="chart-note" style="margin-top:8px">Crude ${cfg.era} projection: per-lap regen from this session's telemetry speed drops; regs cap recharge at ${cfg.per_lap_recharge_mj} MJ/lap; deployment ${cfg.era.indexOf('2026') >= 0 ? 'is store-limited bursts — the 2026 PU has no fixed per-lap deploy quota (ceiling ' + cfg.deploy_allowance_mj + ' MJ/lap)' : 'is capped at ' + cfg.deploy_allowance_mj + ' MJ/lap'} (deploy / recover power 120 kW = ${cfg.rate_mj_s} MJ/s; store = ${cfg.capacity_mj} MJ usable, so 1% = 0.04 MJ); time effect ${cfg.pace_s_per_mj}s per MJ deployed — measured per track from this circuit's full-throttle share (${cfg.limited_effectiveness}x on energy-limited laps). Energy values are synthetic, not measured.</div>` +
            `<div style="margin-top:8px">` + ordered.map(m =>
                `<div class="mp-row"><span>${modeLabel(m.mode)} — deployed ${m.deployed_total_mj.toFixed(1)} MJ · energy-limited laps ${m.limited_laps}</span><b>${m.final_battery_mj.toFixed(2)} MJ</b></div>` +
                `<div class="mp-row"><span>${cfg.mode_notes[m.mode]}</span><b></b></div>`).join('') +
            `</div></details>`;
    } catch(e) {
        err.textContent = 'Error: ' + e.message;
        err.style.display = 'block';
    }
    if (btn) { btn.disabled = false; btn.textContent = 'ANALYZE ENERGY MODES'; }
}

// ENERGY SANDBOX (PPT module 02): per-sector deploy deltas within the mode's per-lap deploy envelope
let sbSeq = 0;
let sbTimer = null;

function sbSliderInput() {
    const d = [0, 1, 2].map(k => parseFloat(document.getElementById('sb-d' + (k + 1)).value));
    d.forEach((v, k) => document.getElementById('sb-v' + (k + 1)).textContent = (v >= 0 ? '+' : '') + v.toFixed(2));
    const sum = d[0] + d[1] + d[2];
    document.getElementById('sb-sum').innerHTML =
        '&Sigma; &Delta; = ' + (sum >= 0 ? '+' : '') + sum.toFixed(2) +
        ' MJ &middot; lap budget fixed &mdash; deltas reallocate within it (live recalc)';
    clearTimeout(sbTimer);
    sbTimer = setTimeout(() => runSandbox(true), 250);   // PPT target: interactive backend recalc < 300 ms
}

async function runSandbox(auto) {
    const err = document.getElementById('sb-err');
    err.style.display = 'none';
    const seq = ++sbSeq;
    const sid = parseInt(document.getElementById('sb-session').value, 10);
    const lap = parseInt(document.getElementById('sb-lap').value, 10);
    const mode = document.getElementById('sb-mode').value;
    const battRaw = document.getElementById('sb-batt').value;
    if (isNaN(sid) || isNaN(lap) || sid <= 0 || lap <= 0) {
        err.textContent = 'Enter a session id and a lap.'; err.style.display = 'block'; return;
    }
    const body = {
        session_id: sid, current_lap: lap, mode: mode,
        deltas_mj: {
            s1: parseFloat(document.getElementById('sb-d1').value),
            s2: parseFloat(document.getElementById('sb-d2').value),
            s3: parseFloat(document.getElementById('sb-d3').value)
        }
    };
    if (battRaw && !isNaN(parseFloat(battRaw))) body.battery_mj = parseFloat(battRaw);
    const btn = [...document.querySelectorAll('button')].find(b => /SIMULATE SECTOR SHIFT/.test(b.innerText || ''));
    if (btn && !auto) { btn.disabled = true; btn.textContent = 'SIMULATING...'; }
    try {
        const res = await fetch('/api/strategy/energy-sandbox', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (seq !== sbSeq) return;                 // a newer run superseded this one
        if (data.error) throw new Error(data.error);
        renderSandbox(data);
    } catch(e) {
        if (seq === sbSeq) { err.textContent = 'Error: ' + e.message; err.style.display = 'block'; }
    }
    if (btn && !auto) { btn.disabled = false; btn.textContent = 'SIMULATE SECTOR SHIFT'; }
}

function sbBar(label, v, maxV, color) {
    const w = maxV > 0 ? Math.max(2, v / maxV * 100) : 0;
    return '<div class="sb-bar-lbl"><span>' + label + '</span><span>' + v.toFixed(2) + ' MJ</span></div>' +
           '<div class="sb-bar"><i style="width:' + w.toFixed(1) + '%;background:' + color + '"></i></div>';
}

function sbSectorCard(s) {
    const maxV = Math.max(s.deploy_mj, s.harvest_mj, 1e-6);
    const net = s.deploy_mj - s.harvest_mj;
    const netCol = net > 0.005 ? '#ff6b6b' : (net < -0.005 ? '#00c853' : '#999');
    const pace = (s.pace_s_per_mj != null) ?
        ' &middot; <span style="color:#8ee6a8" title="measured per-track value of deployment in this sector">' + s.pace_s_per_mj.toFixed(2) + ' s/MJ</span>' : '';
    const credit = (s.time_credit_s != null && Math.abs(s.time_credit_s) >= 0.005) ?
        ' &middot; <span style="color:#ffd97a">' + (s.time_credit_s >= 0 ? '+' : '') + s.time_credit_s.toFixed(2) + 's</span>' : '';
    return '<div class="sb-sec">' +
        '<div class="sb-sec-t">S' + s.sector + ' &middot; ' + s.time_s + 's &middot; ' + Math.round(s.share * 100) + '% of lap' + pace + credit + '</div>' +
        sbBar('DEPLOY', s.deploy_mj, maxV, '#ffd700') +
        sbBar('HARVEST', s.harvest_mj, maxV, '#00d2be') +
        '<div class="sb-soc">NET <span style="color:' + netCol + '">' + (net >= 0 ? '+' : '') + net.toFixed(2) + ' MJ</span> &middot; SOC ' +
        s.start_soc_pct.toFixed(0) + '&rarr;' + s.min_soc_pct.toFixed(0) + '&rarr;' + s.end_soc_pct.toFixed(0) + '%</div>' +
        '</div>';
}

function sbWarns(list) {
    if (!list || !list.length) return '';
    const colors = { warn: '#ff6b6b', info: '#00d2be', ok: '#00c853' };
    return '<div class="sb-warns">' + list.map(w =>
        '<div class="sb-warn ' + w.level + '"><span style="color:' + (colors[w.level] || '#aaa') + '">[' + w.code.toUpperCase() + ']</span> ' + w.message + '</div>'
    ).join('') + '</div>';
}

function sbSrcLabel(src) {
    if (src === 'measured_lap') return 'this lap\'s measured FastF1 sectors';
    if (src === 'measured_fastf1') return 'measured FastF1 sector split for this track';
    return 'equal thirds (no measured sectors for this track yet)';
}

function renderSandbox(data) {
    const cfg = data.config, B = data.baseline, R = data.result, bd = data.budget;
    const el = document.getElementById('sb-out');
    const socChg = R.end_soc_pct - B.end_soc_pct;
    const srcNote = data.lap.sector_time_s ? ' sectors ' + data.lap.sector_time_s.map((t, i) => 'S' + (i + 1) + '=' + t.toFixed(1) + 's').join('/') + ' (' + sbSrcLabel(cfg.sector_time_source) + ')' : '';
    el.innerHTML =
        '<div class="sb-hdr">' + data.session.track + ' &middot; ' + data.session.date + ' &mdash; lap ' + data.lap.lap_number +
        ' &middot; ' + data.lap.mode.toUpperCase() + ' mode &middot; battery in ' + cfg.battery_start_pct.toFixed(1) + '% (' + cfg.battery_start_mj.toFixed(2) + ' MJ) &middot; backend ' + data.elapsed_ms.toFixed(0) + ' ms</div>' +
        '<div class="sb-hdr" style="opacity:.75;font-size:.68em;letter-spacing:.5px">' + srcNote + '</div>' +
        '<div class="sb-sec-grid">' + R.sectors.map(sbSectorCard).join('') + '</div>' +
        '<div class="sb-chips">' +
            '<span class="sb-chip">LAP BUDGET (fixed) <b>' + bd.lap_budget_mj.toFixed(2) + ' MJ</b></span>' +
            '<span class="sb-chip">DELIVERED <b>' + R.deploy_mj.toFixed(2) + ' MJ</b></span>' +
            (R.unspent_mj > 0.005 ? '<span class="sb-chip" style="border-color:#ff6b6b;color:#ff6b6b">UNSPENT (floor) <b>' + R.unspent_mj.toFixed(2) + ' MJ</b></span>' : '') +
            '<span class="sb-chip">END SOC <b>' + B.end_soc_pct.toFixed(1) + '&rarr;' + R.end_soc_pct.toFixed(1) + '%</b></span>' +
            (Math.abs(socChg) > 0.05 ? '<span class="sb-chip">SOC &Delta; <b>' + (socChg >= 0 ? '+' : '') + socChg.toFixed(1) + '%</b></span>' : '') +
            '<span class="sb-chip">MIN SOC <b>' + R.min_soc_pct.toFixed(1) + '%</b></span>' +
        '</div>' +
        sbWarns(R.warnings) +
        '<details class="ht-hint" style="margin-top:10px"><summary>ENGINEERING DETAILS — sandbox model</summary><div class="chart-note" style="margin-top:8px">Sandbox = crude ' + cfg.era + ' sector model. Sector durations come from measured sector times (per-lap FastF1 or the track profile trained into the lap model); deploy vs harvest placement within the lap still uses the ~' + data.lap.telemetry_samples +
        ' telemetry samples (no sector markers in the feed, so energy flows are thirds of the sample stream with shares smoothed for the coarse sampling). Deployment\'s time value is measured per track: ' + cfg.pace_s_per_mj.toFixed(3) + ' s/MJ overall here (per-sector ' + cfg.sector_pace_s_per_mj.map((p, i) => 'S' + (i + 1) + ' ' + p.toFixed(2)).join(' / ') + '), from this track\'s full-throttle share. Deltas reallocate within the per-lap deploy envelope (2026: no fixed quota — store-limited bursts); deploy/recover flow is capped at 120 kW = ' + cfg.deploy_rate_mj_s +
        ' MJ/s per sector; store ' + cfg.capacity_mj.toFixed(1) + ' MJ usable (reserve ' + cfg.reserve_mj.toFixed(2) + ' MJ). Energy values are synthetic, not measured.</div></details>';
}

// DASHBOARD DRIVER FILTER
let selectedDashboardDriver = '';

async function loadDashboardDrivers() {
    const sel = document.getElementById('dash-driver');
    try {
        const res = await fetch('/api/drivers/list');
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        const drivers = data.drivers || [];
        sel.innerHTML = '<option value="">All drivers</option>' + drivers.map(d => `<option value="${d.code}">${d.name} (${d.code}) --- ${d.sessions} session${d.sessions === 1 ? '' : 's'}</option>`).join('');
    } catch(e) { sel.innerHTML = '<option value="">All drivers</option>'; }
}

function onDashboardDriverChange() {
    selectedDashboardDriver = document.getElementById('dash-driver').value;
    sessionFastestMs = Infinity;
    setStatVal('stat-fastest', '--:--.---');
    loadSessionsWithCache();
    updateStats();
}

// SESSIONS SIDEBAR
function renderSessionsList(sessions) {
    const list = document.getElementById('sessions-list');
    if (!list) return;
    _sessionsCache = Array.isArray(sessions) ? sessions : [];
    if (!_sessionsCache.length) {
        list.innerHTML = '<div class="no-data" style="height:80px">NO SESSIONS YET</div>';
        return;
    }
    list.innerHTML = '';
    // Auto-select the first session that actually has laps.  Sessions with
    // 0 laps are genuine DNS / early-DNF records (driver present, no timed
    // lap) and stay listed, but should never become the silently-loaded
    // default - that would open every chart empty.
    const firstWithLaps = _sessionsCache.findIndex(s => (s.total_laps || 0) > 0);
    _sessionsCache.forEach((s, i) => {
        const div = document.createElement('div');
        div.className = 'session-item' + (i === firstWithLaps ? ' active' : '');
        div.innerHTML = `<div class="sesh-track">${s.track_name}</div><div class="sesh-meta">${s.driver_code ? badgeifyDriver(s.driver_code, '') + ' / ' : ''}${s.date || ''} / ${s.total_laps} laps<span class="sesh-id"> · ID ${s.session_id}</span></div><div class="sesh-best">${s.fastest_lap ? fmtTime(s.fastest_lap) : 'N/A'}</div>`;
        div.onclick = () => { document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active')); div.classList.add('active'); loadSession(s.session_id); };
        list.appendChild(div);
        if (i === firstWithLaps) loadSession(s.session_id);
    });
}

async function loadSessionsWithCache() {
    const list = document.getElementById('sessions-list');
    try {
        const url = selectedDashboardDriver ? `/api/sessions?driver=${encodeURIComponent(selectedDashboardDriver)}` : '/api/sessions';
        const res = await fetch(url);
        const sessions = await res.json();
        if (sessions.error) throw new Error(sessions.error);
        setRecentSessionsMode(null);
        renderSessionsList(sessions);
    } catch(e) { list.innerHTML = `<div class="no-data" style="height:80px;color:#ff6666">ERROR: ${e.message}</div>`; }
}

// The sidebar can show a calendar round's sessions instead of the plain
// "most recent" list; the little red label in the header jumps back.
function setRecentSessionsMode(label) {
    const el = document.getElementById('sessions-mode');
    if (!el) return;
    if (label) {
        el.textContent = '◀ all · ' + label;
        el.style.display = 'inline';
    } else {
        el.style.display = 'none';
    }
}

function resetRecentSessions() {
    selectedDashboardDriver = document.getElementById('dash-driver').value;
    loadSessionsWithCache();
}

// ── CALENDAR ROUND CLICK ────────────────────────────────────────────
// Loads one round's race sessions into the Recent Sessions panel (so the
// sidebar becomes that race), then preselects the P0 overtake what-if
// (year / track / leader / chaser) so the Hammer Time tab is ready to run
// that race without re-typing anything.
async function loadRoundSessions(year, idx) {
    const races = (RACE_CALENDARS && RACE_CALENDARS[year]) || [];
    const race = races[idx];
    if (!race) return;
    if (calYear !== year) selectCalendarYear(year);
    const list = document.getElementById('sessions-list');
    if (!race.covered) {
        selectedDashboardDriver = '';
        const dd = document.getElementById('dash-driver');
        if (dd) dd.value = '';
        setRecentSessionsMode(null);
        if (list) list.innerHTML = `<div class="no-data" style="height:80px">NO DB SESSIONS FOR ${(race.country || '').toUpperCase()} ${year}</div>`;
        return;
    }
    if (list) list.innerHTML = '<div class="no-data" style="height:80px">LOADING...</div>';
    try {
        const res = await fetch(`/api/sessions?track=${encodeURIComponent(race.track || race.country || '')}&year=${year}&limit=500`);
        const sessions = await res.json();
        if (sessions.error) throw new Error(sessions.error);
        selectedDashboardDriver = '';
        const dd = document.getElementById('dash-driver');
        if (dd) dd.value = '';
        renderSessionsList(sessions);
        // Re-highlight the clicked round (renderSessionsList may have opened a
        // session, which highlights by track name anyway).
        const hi = (race.db_tracks && race.db_tracks.length) ? race.db_tracks[0] : race.track;
        if (hi) highlightCalendarRow(hi);
        setRecentSessionsMode(`${year} R${String(race.round).padStart(2, '0')} · ${race.country}`);
        preselectP0ForRace(year, race, sessions);
    } catch (e) {
        if (list) list.innerHTML = `<div class="no-data" style="height:80px;color:#ff6666">ERROR: ${e.message}</div>`;
    }
}

function preselectP0ForRace(year, race, sessions) {
    const yearEl = document.getElementById('ov-year');
    if (yearEl) yearEl.value = year;
    const trackVal = pickP0Track(race);
    if (trackVal) setSelectValue('ov-track', trackVal);
    const pair = pickP0DriverPair(race, sessions);
    if (pair) {
        setSelectValue('ov-leader', pair[0]);
        setSelectValue('ov-chaser', pair[1]);
    }
    htUpdateContext();
}

// Best ov-track option for a round.  The exact DB circuit name wins (the
// P1 session resolver matches it against s.track_name), appended to the
// dropdown when the overtake model only knows the title-cased variant.
function pickP0Track(race) {
    const sel = document.getElementById('ov-track');
    if (!sel) return null;
    // Exact option values only — the P1 session resolver matches the chosen
    // value against the DB's stored track_name, so a merely title-cased
    // variant ('...Nazionale Di Monza') is not a substitute for the real DB
    // name ('...Nazionale di Monza').  Append the exact name when missing.
    const existing = [...sel.options].map(o => o.value);
    const dbTracks = (race.db_tracks || []).map(t => String(t)).filter(Boolean);
    if (dbTracks.length) {
        const track = dbTracks[0];
        if (existing.indexOf(track) === -1) {
            const opt = document.createElement('option');
            opt.value = track;
            opt.textContent = track;
            sel.appendChild(opt);
        }
        return track;
    }
    // No DB session for the round: fall back to a model-covered venue that
    // the calendar track key points at (rare, best-effort preselect).
    const parts = (race.track || '').toLowerCase().split('|').filter(Boolean);
    const opt = [...sel.options].find(o => {
        const l = o.value.toLowerCase();
        return parts.some(p => p && (l.includes(p) || p.includes(l)));
    });
    return opt ? opt.value : null;
}

// Default leader/chaser for a race: the drivers present that have pace
// models.  Prefer a headline duel (HAM/VER, else LEC/RUS) when both raced
// that round, otherwise the two fastest by best lap — the pair the P1
// full-race sim can resolve immediately.
function pickP0DriverPair(race, sessions) {
    if (!Array.isArray(sessions)) return null;
    const modelCodes = new Set((ovDrivers || []).map(d => d.code));
    const fastest = {};
    sessions.forEach(s => {
        const code = s.driver_code;
        if (!code || !(s.total_laps > 0) || !s.fastest_lap) return;
        if (fastest[code] === undefined || s.fastest_lap < fastest[code]) fastest[code] = s.fastest_lap;
    });
    const codes = Object.keys(fastest).sort((a, b) => fastest[a] - fastest[b]);
    const candidates = codes.filter(c => modelCodes.has(c));
    const pool = candidates.length >= 2 ? candidates : codes;
    if (pool.length < 2) return null;
    for (const pair of [['HAM', 'VER'], ['LEC', 'RUS']]) {
        if (pool.indexOf(pair[0]) !== -1 && pool.indexOf(pair[1]) !== -1) return pair;
    }
    return [pool[0], pool[1]];
}

function setSelectValue(id, value) {
    const sel = document.getElementById(id);
    if (!sel || !value) return false;
    const ok = [...sel.options].some(o => o.value === value);
    if (ok) sel.value = value;
    return ok;
}

// F1 INFO BAR: Clocks + Circuit Banner

const CIRCUIT_FLAGS = {
    'Albert Park':'au','Melbourne':'au','Bahrain':'bh','Sakhir':'bh',
    'Jeddah':'sa','Saudi Arabia':'sa','Miami':'us','Imola':'it',
    'Emilia-Romagna':'it','Monaco':'mc','Monte Carlo':'mc','Spain':'es',
    'Montmelo':'es','Barcelona':'es','Montreal':'ca','Canada':'ca',
    'Austria':'at','Spielberg':'at','Red Bull Ring':'at','Silverstone':'gb','Britain':'gb',
    'Great Britain':'gb','Hungary':'hu','Hungaroring':'hu','Belgium':'be',
    'Spa':'be','Spa-Francorchamps':'be','Netherlands':'nl','Zandvoort':'nl',
    'Italy':'it','Monza':'it','Singapore':'sg','Japan':'jp',
    'Suzuka':'jp','Qatar':'qa','Lusail':'qa','USA':'us','United States':'us','Austin':'us',
    'COTA':'us','Mexico':'mx','Hermanos Rodriguez':'mx','Brazil':'br','Sao Paulo':'br','São Paulo':'br',
    'Interlagos':'br','Las Vegas':'us','Abu Dhabi':'ae','Yas Marina':'ae','Yas Island':'ae',
    'China':'cn','Shanghai':'cn','Baku':'az','Azerbaijan':'az',
    'Portugal':'pt','Algarve':'pt','Istanbul':'tr','Turkey':'tr',
    'Sochi':'ru','Russia':'ru',
    'Mugello':'it','Enzo e Dino Ferrari':'it','Nurburgring':'de','Hockenheim':'de','Germany':'de',
    'Paul Ricard':'fr','Le Castellet':'fr','France':'fr',
};

function getFlagUrl(trackName) {
    if (!trackName) return null;
    const code = CIRCUIT_FLAGS[trackName];
    if (code) return `/static/flags/${code}.png`;
    // Strip accents for fuzzy matching (São Paulo -> Sao Paulo)
    const norm = trackName.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    const normCode = CIRCUIT_FLAGS[norm];
    if (normCode) return `/static/flags/${normCode}.png`;
    for (const [key, c] of Object.entries(CIRCUIT_FLAGS)) {
        const keyNorm = key.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
        if (trackName.includes(key) || key.includes(trackName) ||
            norm.includes(keyNorm) || keyNorm.includes(norm)) {
            return `/static/flags/${c}.png`;
        }
    }
    return null;
}

function getFlagEmoji(trackName) {
    if (!trackName) return '';
    const map = {
        'Albert Park':'au','Melbourne':'au','Bahrain':'bh','Sakhir':'bh',
        'Jeddah':'sa','Saudi Arabia':'sa','Miami':'us','Imola':'it',
        'Monaco':'mc','Spain':'es','Montreal':'ca','Canada':'ca',
        'Austria':'at','Silverstone':'gb','Britain':'gb','Hungary':'hu',
        'Belgium':'be','Netherlands':'nl','Italy':'it','Singapore':'sg',
        'Japan':'jp','Qatar':'qa','USA':'us','Austin':'us','Mexico':'mx',
        'Brazil':'br','Las Vegas':'us','Abu Dhabi':'ae','China':'cn',
        'Baku':'az','Azerbaijan':'az',
    };
    if (map[trackName]) return map[trackName];
    for (const [k,v] of Object.entries(map)) {
        if (trackName.includes(k) || k.includes(trackName)) return v;
    }
    return '';
}

function updateClocks() {
    const now = new Date();
    const myTime = document.getElementById('f1-my-time');
    if (myTime) myTime.textContent = formatTime(now);
}

function updateCircuitBanner(sessionId) {
    const cached = _sessionsCache.find(s => String(s.session_id) === String(sessionId));
    if (!cached) return;
    const flagEl = document.getElementById('f1-flag');
    const nameEl = document.getElementById('f1-circuit-name');
    const roundEl = document.getElementById('f1-round-num');
    const dateEl = document.getElementById('f1-date-range');
    if (flagEl) {
        const url = getFlagUrl(cached.track_name);
        if (url) {
            flagEl.innerHTML = `<img src="${url}" alt="" style="width:22px;height:16px;border-radius:2px;object-fit:cover;border:1px solid rgba(255,255,255,0.1);vertical-align:middle;">`;
        } else {
            flagEl.textContent = '';
        }
    }
    if (nameEl) nameEl.textContent = cached.track_name || '---';
    if (roundEl && _sessionsCache.length) {
        const idx = _sessionsCache.findIndex(s => String(s.session_id) === String(sessionId));
        if (idx >= 0) roundEl.textContent = String(_sessionsCache.length - idx).padStart(2, '0');
    }
    if (dateEl && cached.date) {
        try {
            const d = new Date(cached.date + 'T00:00:00');
            const months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
            dateEl.textContent = `${d.getDate()} ${months[d.getMonth()]}`;
        } catch(e) { dateEl.textContent = cached.date; }
    }
}

function formatTime(date, tz) {
    try {
        const opts = { hour:'2-digit', minute:'2-digit', hour12:false };
        if (tz) opts.timeZone = tz;
        return date.toLocaleTimeString('en-GB', opts);
    } catch(e) { return '--:--'; }
}

// LIGHT / DARK MODE
function toggleTheme() {
    document.body.classList.toggle('light-mode');
    const btn = document.getElementById('theme-btn');
    if (btn) btn.textContent = document.body.classList.contains('light-mode') ? 'NIGHT' : 'SUN';
    try { localStorage.setItem('f1dp-theme', document.body.classList.contains('light-mode') ? 'light' : 'dark'); } catch(e) {}
}
(function restoreTheme() {
    try {
        if (localStorage.getItem('f1dp-theme') === 'light') {
            document.body.classList.add('light-mode');
            const btn = document.getElementById('theme-btn');
            if (btn) btn.textContent = 'NIGHT';
        }
    } catch(e) {}
})();

// RACE CALENDAR
// Official FIA race calendars 2020-2026 are served by /api/calendar —
// single source of truth in scripts/race_calendar.py — with every round
// annotated against the DB (race sessions, the DB circuits it matches,
// drivers present, best lap) so the calendar doubles as a data-coverage
// map for the overtake tooling.  Clicking a round loads that race's
// sessions into the Recent Sessions panel and preselects the P0 what-if.
// Rounds with no DB session render dimmed.
let RACE_CALENDARS = null;   // {year: [rounds...]} once /api/calendar loads

async function loadCalendars() {
    const grid = document.getElementById('cal-grid');
    try {
        const res = await fetch('/api/calendar');
        const d = await res.json();
        if (d.error) throw new Error(d.error);
        RACE_CALENDARS = d.calendars || {};
        buildCalendar(calYear);
    } catch (e) {
        if (grid) grid.innerHTML = '<div class="chart-note" style="padding:10px">Calendar load error: ' + e.message + '</div>';
    }
}

let calYear = 2026;

function selectCalendarYear(year) {
    calYear = year;
    document.querySelectorAll('#cal-year-row .calendar-toggle-btn').forEach(b =>
        b.classList.toggle('active', parseInt(b.dataset.year, 10) === year));
    buildCalendar(year);
    const cal = document.getElementById('race-calendar');
    if (cal && !cal.classList.contains('show')) cal.classList.add('show');
}

function buildCalendar(year) {
    const grid = document.getElementById('cal-grid');
    if (!grid) return;
    const label = document.getElementById('cal-year-label');
    if (label) label.textContent = year;
    if (!RACE_CALENDARS) {
        grid.innerHTML = '<div class="chart-note" style="padding:10px">Loading calendar from /api/calendar...</div>';
        return;
    }
    const races = RACE_CALENDARS[year] || [];
    grid.innerHTML = races.map((r, i) => {
        const flagUrl = `/static/flags/${r.code}.png`;
        // Coverage line: drivers present + best lap when the DB has a
        // session for this round; dim the item when it has none.
        const cov = r.covered
            ? `<div class="cal-db">${(r.drivers || []).join(' · ')}${r.best_lap_s ? ' · best ' + fmtTime(r.best_lap_s) : ''}</div>`
            : `<div class="cal-db">no DB data</div>`;
        const dim = r.covered ? '' : ' nodata';
        return `<div class="cal-item${dim}" data-track="${(r.track || r.country).toLowerCase()}" data-year="${year}" onclick="loadRoundSessions(${year}, ${i})" title="${r.country} ${year} — click to load this round's sessions into Recent Sessions and preselect the P0 what-if"><span class="cal-round">R${String(r.round).padStart(2,'0')}</span><img class="cal-flag" src="${flagUrl}" alt="${r.country}"><div class="cal-info"><div class="cal-country">${r.country}</div><div class="cal-dates">${r.dates}</div>${cov}</div></div>`;
    }).join('');
}

function highlightCalendarRow(trackName) {
    document.querySelectorAll('.cal-item').forEach(el => el.classList.remove('current'));
    if (!trackName) return;
    const t = trackName.toLowerCase();
    document.querySelectorAll('.cal-item').forEach(el => {
        const parts = (el.dataset.track || '').split('|');
        if (parts.some(p => p && (t.includes(p) || p.includes(t)))) el.classList.add('current');
    });
}

// DRIVER TEAM COLORS
const DRIVER_TEAMS = {
    'VER':'#3671C6','PER':'#3671C6','HAM':'#27F4D2','RUS':'#27F4D2',
    'LEC':'#E8002D','SAI':'#E8002D','NOR':'#FF8000','PIA':'#FF8000',
    'ALO':'#229971','STR':'#229971','GAS':'#0093CC','OCO':'#0093CC',
    'ALB':'#64C4FF','SAR':'#64C4FF','TSU':'#6692FF','LAW':'#6692FF',
    'HUL':'#B6BABD','RIC':'#B6BABD','MAG':'#B6BABD','BEA':'#B6BABD',
    'BOT':'#C92D4B','ZHO':'#C92D4B','DEV':'#6692FF','COL':'#FF8000',
};

function badgeifyDriver(code, name) {
    const color = DRIVER_TEAMS[code] || '#888';
    const label = name ? `${name} (${code})` : code;
    return `<span class="driver-badge"><span class="driver-badge-dot" style="background:${color}"></span>${label}</span>`;
}

// ── P0 DUAL-AGENT OVERTAKE (Hammer Time tab) ───────────────────
