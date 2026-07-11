/* ═══════════════════════════════════════════════════════
   ABOUT — TERMINAL INTRO + PAC-MAN PLAYGROUND
   ─────────────────────────────────────────────────────────
   1. Terminal boot animation gates the experience
   2. Native page scroll drives Pac-Man along a Catmull-Rom path
   3. A continuous trail of pellets marks the journey
   4. Ghosts wander naturally between the bento walls
   ═══════════════════════════════════════════════════════ */

(function () {
  'use strict';
  if (!document.querySelector('.about-body')) return;

  /* ── DOM refs ── */
  const body          = document.body;
  const intro         = document.getElementById('terminal-intro');
  const introLines    = document.getElementById('intro-lines');
  const startBtn      = document.getElementById('start-btn');
  const startRow      = document.getElementById('start-row');
  const viewport      = document.getElementById('playground-viewport');
  const world         = document.getElementById('playground-world');
  const canvas        = document.getElementById('pac-canvas');
  const ctx           = canvas ? canvas.getContext('2d') : null;
  const miniCv        = document.getElementById('minimap-canvas');
  const miniCtx       = miniCv ? miniCv.getContext('2d') : null;
  const hudScore      = document.getElementById('hud-score');
  const hudCount      = document.getElementById('hud-collected');
  const hudChap       = document.getElementById('hud-chapter');
  const toast         = document.getElementById('milestone-toast');
  const toastTitle    = document.getElementById('toast-title');

  if (!viewport || !world || !canvas || !ctx) return;

  /* ── Config ── */
  const PAC_R          = 14;
  const PELLET_SPACING = 26;   // px along the path
  const BIG_EVERY      = 22;   // every Nth pellet is a big milestone
  const NUM_GHOSTS     = 3;

  /* ── State ── */
  let walls = [];
  let pathPoints = [];
  let totalPathLen = 0;
  let pellets = [];
  let pelletsEaten = 0;
  let totalPellets = 0;
  let score = 0;
  let pacProgress = 0;
  let pacTargetProgress = 0;
  let pac = { x: 0, y: 0, dir: 0, mouth: 0, mouthDir: 1, sx: 0, sy: 0 };
  let ghosts = [];
  let camX = 0, camY = 0;
  let gameStarted = false;
  let dpr = 1;
  let frameCount = 0;
  let chapterMilestones = []; // { progress, title }
  let colXs = {}; // col index → x in world coords
  let rowYs = {}; // row index → y in world coords

  /* ═════════════════════════════════════════════════════
     TERMINAL INTRO
  ═════════════════════════════════════════════════════ */
  const INTRO_SEQUENCE = [
    { t: 'akhrot@about ~ $ ./launch --immersive',          d: 180, c: '' },
    { t: '',                                                d: 60,  c: '' },
    { t: '> Loading About Page...',                         d: 220, c: '' },
    { t: '> Initializing gamified immersive experience...', d: 240, c: '' },
    { t: '> Generating interactive world...',               d: 220, c: '' },
    { t: '> Spawning bento walls .......... [OK]',          d: 160, c: 'success' },
    { t: '> Calibrating story trail ........ [OK]',         d: 160, c: 'success' },
    { t: '> Loading Pac-Man module ......... [OK]',         d: 160, c: 'success' },
    { t: '> Booting ghost AI ............... [OK]',         d: 160, c: 'success' },
    { t: '> Mounting cursor system ......... [OK]',         d: 140, c: 'success' },
    { t: '> Compiling motion engine ........ [OK]',         d: 220, c: 'success' },
    { t: '',                                                d: 80,  c: '' },
    { t: '> All systems green. Ready to play.',             d: 260, c: 'muted' },
    { t: '',                                                d: 0,   c: '' }
  ];

  // Scales the whole boot sequence (typing + inter-line pauses). <1 = faster.
  const INTRO_SPEED = 0.55;

  function runIntro() {
    if (!intro || !introLines) { enterGame(); return; }

    let idx = 0;
    function nextLine() {
      if (idx >= INTRO_SEQUENCE.length) {
        if (startRow) startRow.classList.add('ready');
        return;
      }
      const line = INTRO_SEQUENCE[idx];
      const el = document.createElement('div');
      el.className = 'intro-line' + (line.c ? ' ' + line.c : '');
      introLines.appendChild(el);

      if (line.t === '') {
        idx++;
        setTimeout(nextLine, (line.d || 100) * INTRO_SPEED);
        return;
      }

      let i = 0;
      const cursor = document.createElement('span');
      cursor.className = 'cursor-blink';
      el.appendChild(cursor);

      function typeChar() {
        if (i <= line.t.length) {
          el.textContent = line.t.slice(0, i);
          el.appendChild(cursor);
          i++;
          // Slightly variable typing speed for organic feel
          setTimeout(typeChar, (9 + Math.random() * 14) * INTRO_SPEED);
        } else {
          // Done with this line — remove cursor, advance
          cursor.remove();
          idx++;
          setTimeout(nextLine, (line.d || 160) * INTRO_SPEED);
        }
      }
      typeChar();
    }

    nextLine();

    // Wire up Start button + keyboard
    if (startBtn) startBtn.addEventListener('click', enterGame);
    function onIntroKey(e) {
      if (!startRow || !startRow.classList.contains('ready')) {
        // Allow skipping intro mid-stream with Esc
        if (e.key === 'Escape') {
          while (idx < INTRO_SEQUENCE.length) {
            const line = INTRO_SEQUENCE[idx];
            if (line.t) {
              const el = document.createElement('div');
              el.className = 'intro-line' + (line.c ? ' ' + line.c : '');
              el.textContent = line.t;
              introLines.appendChild(el);
            }
            idx++;
          }
          if (startRow) startRow.classList.add('ready');
        }
        return;
      }
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        enterGame();
      }
    }
    document.addEventListener('keydown', onIntroKey);
  }

  function enterGame() {
    if (gameStarted) return;
    gameStarted = true;
    if (intro) intro.classList.add('done');
    body.classList.remove('intro-active');
    body.classList.add('game-active');

    // Start game after intro fade
    setTimeout(() => {
      if (intro) intro.style.display = 'none';
    }, 900);

    // Boot game (give layout one frame to settle now that scroll is enabled)
    requestAnimationFrame(() => requestAnimationFrame(init));
  }

  /* ═════════════════════════════════════════════════════
     GAME INIT
  ═════════════════════════════════════════════════════ */
  function init() {
    resizeCanvas();
    insertGridAnchors();
    measureWorld();
    measureGrid();
    buildPath();
    placePellets();
    initGhosts();
    setupInput();

    // Start at path beginning
    pacProgress = 0;
    pacTargetProgress = 0;
    const p0 = pathPoints[0] || { x: 0, y: 0, dir: 0 };
    pac.x = p0.x; pac.y = p0.y; pac.dir = p0.dir || 0;

    camX = window.innerWidth  / 2 - pac.x;
    camY = window.innerHeight / 2 - pac.y;
    world.style.transform = `translate3d(${camX}px, ${camY}px, 0)`;

    loop();
  }

  /* ═════════════════════════════════════════════════════
     CANVAS / WORLD MEASUREMENT
  ═════════════════════════════════════════════════════ */
  function resizeCanvas() {
    dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    canvas.width  = window.innerWidth  * dpr;
    canvas.height = window.innerHeight * dpr;
    canvas.style.width  = window.innerWidth  + 'px';
    canvas.style.height = window.innerHeight + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function measureWorld() {
    walls = [];
    world.querySelectorAll('.bento-wall').forEach(el => {
      walls.push({
        x: el.offsetLeft,
        y: el.offsetTop,
        w: el.offsetWidth,
        h: el.offsetHeight,
        el,
        key: el.dataset.bento || 'ornament'
      });
    });
  }

  /* ── Grid anchors: invisible 1×1 cells used to read row/col pixel positions
     after CSS Grid lays the world out (rows may auto-size to content) ── */
  function insertGridAnchors() {
    if (world.querySelector('.row-anchor')) return; // already inserted
    // 13 row anchors at col 12 (no card occupies col 12)
    for (let r = 1; r <= 13; r++) {
      const el = document.createElement('div');
      el.className = 'grid-anchor row-anchor';
      el.dataset.row = r;
      el.style.gridColumn = '12 / span 1';
      el.style.gridRow = r + ' / span 1';
      world.appendChild(el);
    }
    // 12 col anchors at row 13 (no card occupies row 13)
    for (let c = 1; c <= 12; c++) {
      const el = document.createElement('div');
      el.className = 'grid-anchor col-anchor';
      el.dataset.col = c;
      el.style.gridColumn = c + ' / span 1';
      el.style.gridRow = '13 / span 1';
      world.appendChild(el);
    }
  }

  function measureGrid() {
    colXs = {}; rowYs = {};
    world.querySelectorAll('.row-anchor').forEach(el => {
      rowYs[el.dataset.row] = el.offsetTop + el.offsetHeight / 2;
    });
    world.querySelectorAll('.col-anchor').forEach(el => {
      colXs[el.dataset.col] = el.offsetLeft + el.offsetWidth / 2;
    });
  }

  function pointInAnyWall(x, y, pad) {
    pad = pad || 0;
    for (const w of walls) {
      if (x > w.x - pad && x < w.x + w.w + pad &&
          y > w.y - pad && y < w.y + w.h + pad) return true;
    }
    return false;
  }

  /* ═════════════════════════════════════════════════════
     PATH BUILDING — STRAIGHT-LINE CORNERS ONLY
     Every consecutive pair of corners differs in exactly ONE of (col, row).
     Pac-Man moves along pure horizontal or vertical segments.
     Corner (col, row) coords are resolved at runtime via grid anchors,
     so the path automatically follows content-sized rows.
  ═════════════════════════════════════════════════════ */
  const PATH_CORNERS = [
    { col: 6,  row: 1,  chapter: 'Origin'   },  // start — between hero & identity
    { col: 6,  row: 3                       },  // ↓
    { col: 11, row: 3                       },  // →
    { col: 11, row: 1,  chapter: 'Identity' },  // ↑ to corner near identity
    { col: 11, row: 7,  chapter: 'Beliefs'  },  // ↓ along right edge of beliefs
    { col: 6,  row: 7,  chapter: 'Culture'  },  // ← below culture
    { col: 4,  row: 7,  chapter: 'Genesis'  },  // ← right of genesis
    { col: 11, row: 7                       },  // → return along row 7
    { col: 11, row: 10, chapter: 'Evolving' },  // ↓ right of evolving
    { col: 7,  row: 10, chapter: 'Vision'   },  // ← below vision
    { col: 7,  row: 13                      },  // ↓ down to bottom edge
    { col: 9,  row: 13, chapter: 'Join Us'  }   // → below CTA
  ];

  function buildPath() {
    if (Object.keys(colXs).length === 0 || Object.keys(rowYs).length === 0) {
      pathPoints = [];
      totalPathLen = 0;
      return;
    }

    // Resolve each logical corner to pixel coords using the measured grid
    const corners = PATH_CORNERS.map(c => ({
      x: colXs[c.col],
      y: rowYs[c.row],
      chapter: c.chapter || null
    }));

    pathPoints = [];
    chapterMilestones = [];
    let cumLen = 0;

    for (let i = 0; i < corners.length - 1; i++) {
      const a = corners[i];
      const b = corners[i + 1];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const segLen = Math.hypot(dx, dy);
      // Snap direction to pure horizontal or vertical
      const dir = Math.abs(dx) > Math.abs(dy)
        ? (dx >= 0 ? 0 : Math.PI)
        : (dy >= 0 ? Math.PI / 2 : -Math.PI / 2);
      const steps = Math.max(2, Math.ceil(segLen / 5));

      a.lenAt = cumLen;

      for (let s = 0; s < steps; s++) {
        const u = s / steps;
        pathPoints.push({
          x: a.x + dx * u,
          y: a.y + dy * u,
          dir,
          len: cumLen + segLen * u
        });
      }
      cumLen += segLen;
    }
    // Final corner
    const last = corners[corners.length - 1];
    last.lenAt = cumLen;
    pathPoints.push({
      x: last.x,
      y: last.y,
      dir: pathPoints.length ? pathPoints[pathPoints.length - 1].dir : 0,
      len: cumLen
    });

    totalPathLen = cumLen;

    // Chapter milestones — fire when pac progress reaches each labeled corner
    corners.forEach(c => {
      if (!c.chapter) return;
      chapterMilestones.push({
        title: c.chapter,
        progress: totalPathLen > 0 ? c.lenAt / totalPathLen : 0
      });
    });
  }

  function getPointAtProgress(prog) {
    if (pathPoints.length === 0) return { x: 0, y: 0, dir: 0 };
    if (prog <= 0) return pathPoints[0];
    if (prog >= 1) return pathPoints[pathPoints.length - 1];
    const target = prog * totalPathLen;
    let lo = 0, hi = pathPoints.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (pathPoints[mid].len < target) lo = mid + 1;
      else hi = mid;
    }
    const idx = Math.max(1, lo);
    const a = pathPoints[idx - 1];
    const b = pathPoints[idx];
    const segLen = b.len - a.len;
    const u = segLen > 0 ? (target - a.len) / segLen : 0;
    return {
      x: a.x + (b.x - a.x) * u,
      y: a.y + (b.y - a.y) * u,
      dir: b.dir
    };
  }

  /* ═════════════════════════════════════════════════════
     PELLETS — continuous trail along the path
  ═════════════════════════════════════════════════════ */
  function placePellets() {
    pellets = [];
    if (totalPathLen === 0) return;

    const count = Math.floor(totalPathLen / PELLET_SPACING);
    for (let i = 1; i < count; i++) {
      const prog = i / count;
      const pt = getPointAtProgress(prog);
      const isBig = (i % BIG_EVERY === 0);
      pellets.push({
        x: pt.x,
        y: pt.y,
        big: isBig,
        eaten: false,
        progress: prog
      });
    }
    totalPellets = pellets.length;
    pelletsEaten = 0;
    score = 0;
    if (hudCount) hudCount.textContent = `0 / ${totalPellets}`;
    if (hudScore) hudScore.textContent = '0';
  }

  /* ═════════════════════════════════════════════════════
     GHOSTS — wandering AI
  ═════════════════════════════════════════════════════ */
  const GHOST_PALETTE = [
    { name: 'Blinky', color: '#e94560' },
    { name: 'Pinky',  color: '#f4a4c0' },
    { name: 'Inky',   color: '#65d6ff' },
    { name: 'Clyde',  color: '#f4a261' }
  ];

  function initGhosts() {
    ghosts = [];
    const W = world.offsetWidth;
    const H = world.offsetHeight;
    const seeds = [
      { rx: 0.86, ry: 0.42 },
      { rx: 0.32, ry: 0.62 },
      { rx: 0.62, ry: 0.84 },
      { rx: 0.18, ry: 0.18 }
    ];
    for (let i = 0; i < NUM_GHOSTS; i++) {
      const s = seeds[i];
      const x = s.rx * W;
      const y = s.ry * H;
      if (pointInAnyWall(x, y, 22)) continue;
      ghosts.push({
        x, y,
        tx: x, ty: y,
        vx: 0, vy: 0,
        speed: 0.55 + Math.random() * 0.4,
        wobble: Math.random() * Math.PI * 2,
        wobbleSpeed: 0.04 + Math.random() * 0.03,
        color: GHOST_PALETTE[i].color,
        name:  GHOST_PALETTE[i].name,
        retargetCooldown: 0
      });
    }
  }

  function pickGhostTarget(g) {
    let tries = 14;
    while (tries-- > 0) {
      const angle = Math.random() * Math.PI * 2;
      const dist = 180 + Math.random() * 260;
      const nx = g.x + Math.cos(angle) * dist;
      const ny = g.y + Math.sin(angle) * dist;
      const cx = Math.max(40, Math.min(world.offsetWidth  - 40, nx));
      const cy = Math.max(40, Math.min(world.offsetHeight - 40, ny));
      if (!pointInAnyWall(cx, cy, 24)) {
        g.tx = cx; g.ty = cy;
        return;
      }
    }
    // Fallback: nudge slightly
    g.tx = g.x + (Math.random() - 0.5) * 60;
    g.ty = g.y + (Math.random() - 0.5) * 60;
  }

  function updateGhosts() {
    ghosts.forEach(g => {
      const dx = g.tx - g.x;
      const dy = g.ty - g.y;
      const dist = Math.hypot(dx, dy);
      g.retargetCooldown--;
      if (dist < 24 || g.retargetCooldown <= 0) {
        pickGhostTarget(g);
        g.retargetCooldown = 90 + Math.floor(Math.random() * 60);
      }
      // Move toward target with collision check
      const sp = g.speed;
      if (dist > 0.01) {
        const nvx = (dx / dist) * sp;
        const nvy = (dy / dist) * sp;
        const newX = g.x + nvx;
        const newY = g.y + nvy;
        if (!pointInAnyWall(newX, g.y, 18)) g.x = newX;
        else { pickGhostTarget(g); g.retargetCooldown = 40; }
        if (!pointInAnyWall(g.x, newY, 18)) g.y = newY;
        else { pickGhostTarget(g); g.retargetCooldown = 40; }
      }
      g.wobble += g.wobbleSpeed;
    });
  }

  /* ═════════════════════════════════════════════════════
     PAC-MAN UPDATE — driven by scroll progress
  ═════════════════════════════════════════════════════ */
  function updatePac() {
    // Smoothly lerp current progress toward target
    pacProgress += (pacTargetProgress - pacProgress) * 0.08;
    if (Math.abs(pacTargetProgress - pacProgress) < 0.0002) pacProgress = pacTargetProgress;

    const pt = getPointAtProgress(pacProgress);
    pac.x = pt.x;
    pac.y = pt.y;

    // Smooth turn (shortest angular path)
    let dd = (pt.dir || 0) - pac.dir;
    while (dd > Math.PI)  dd -= Math.PI * 2;
    while (dd < -Math.PI) dd += Math.PI * 2;
    pac.dir += dd * 0.18;

    // Mouth animation
    pac.mouth += 0.07 * pac.mouthDir;
    if (pac.mouth > 0.32 || pac.mouth < 0.02) pac.mouthDir *= -1;
  }

  /* ═════════════════════════════════════════════════════
     EAT pellets in front of pac as it advances
  ═════════════════════════════════════════════════════ */
  function eatPellets() {
    for (let i = 0; i < pellets.length; i++) {
      const p = pellets[i];
      if (p.eaten) continue;
      if (p.progress <= pacProgress) {
        p.eaten = true;
        pelletsEaten++;
        score += p.big ? 250 : 50;
        if (hudScore) {
          hudScore.textContent = score;
          hudScore.classList.add('pop');
          setTimeout(() => hudScore && hudScore.classList.remove('pop'), 280);
        }
        if (hudCount) hudCount.textContent = `${pelletsEaten} / ${totalPellets}`;
        sfxCollect(p.big);
      }
    }
  }

  /* ═════════════════════════════════════════════════════
     CHAPTERS / TOAST
  ═════════════════════════════════════════════════════ */
  let firedMilestones = new Set();
  function checkChapters() {
    for (const m of chapterMilestones) {
      if (firedMilestones.has(m.title)) continue;
      if (pacProgress >= m.progress) {
        firedMilestones.add(m.title);
        if (hudChap) hudChap.textContent = m.title;
        showToast(m.title);
      }
    }
    // Mark nearest non-ornament card as visited
    for (const w of walls) {
      if (w.key === 'ornament') continue;
      const cx = Math.max(w.x, Math.min(pac.x, w.x + w.w));
      const cy = Math.max(w.y, Math.min(pac.y, w.y + w.h));
      const dx = pac.x - cx, dy = pac.y - cy;
      if (dx * dx + dy * dy < 110 * 110) w.el.classList.add('visited');
    }
  }

  let toastTimer = null;
  function showToast(title) {
    if (!toast || !toastTitle) return;
    toastTitle.textContent = title;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 2200);
  }

  /* ═════════════════════════════════════════════════════
     CAMERA — keeps Pac centered, world pans
  ═════════════════════════════════════════════════════ */
  function updateCamera() {
    const vw = window.innerWidth, vh = window.innerHeight;
    const tx = vw / 2 - pac.x;
    const ty = vh / 2 - pac.y;
    camX += (tx - camX) * 0.12;
    camY += (ty - camY) * 0.12;
    world.style.transform = `translate3d(${camX}px, ${camY}px, 0)`;

    // Reveal bento walls when they enter the viewport
    walls.forEach(w => {
      const sx = w.x + camX, sy = w.y + camY;
      if (sx + w.w > -80 && sx < vw + 80 && sy + w.h > -80 && sy < vh + 80) {
        w.el.classList.add('in-view');
      }
    });
  }

  /* ═════════════════════════════════════════════════════
     CANVAS RENDER
  ═════════════════════════════════════════════════════ */
  function render() {
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    renderTrail();
    renderPellets();
    renderGhosts();
    renderPac();
  }

  /* Faint glowing path line behind the pellets */
  function renderTrail() {
    if (pathPoints.length < 2) return;
    ctx.save();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = 'rgba(200, 163, 104, 0.08)';
    ctx.beginPath();
    for (let i = 0; i < pathPoints.length; i++) {
      const p = pathPoints[i];
      const sx = p.x + camX;
      const sy = p.y + camY;
      if (i === 0) ctx.moveTo(sx, sy);
      else         ctx.lineTo(sx, sy);
    }
    ctx.stroke();

    // Active progress segment (brighter, ahead of pac)
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(200, 163, 104, 0.22)';
    ctx.beginPath();
    let moved = false;
    for (let i = 0; i < pathPoints.length; i++) {
      const p = pathPoints[i];
      const prog = p.len / totalPathLen;
      if (prog < pacProgress) continue;
      if (prog > pacProgress + 0.18) break;
      const sx = p.x + camX;
      const sy = p.y + camY;
      if (!moved) { ctx.moveTo(sx, sy); moved = true; }
      else        { ctx.lineTo(sx, sy); }
    }
    ctx.stroke();
    ctx.restore();
  }

  /* Pellets drawn directly on canvas (perf-friendly for hundreds) */
  function renderPellets() {
    const vw = window.innerWidth, vh = window.innerHeight;
    const pulse = 0.85 + Math.sin(frameCount * 0.06) * 0.15;

    for (const p of pellets) {
      const sx = p.x + camX;
      const sy = p.y + camY;
      // Off-screen cull
      if (sx < -20 || sx > vw + 20 || sy < -20 || sy > vh + 20) continue;
      if (p.eaten) continue;

      const r = p.big ? 5.5 : 2.4;
      // Glow
      ctx.beginPath();
      ctx.arc(sx, sy, r * 3.5, 0, Math.PI * 2);
      ctx.fillStyle = p.big
        ? `rgba(200, 163, 104, ${0.10 * pulse})`
        : `rgba(200, 163, 104, ${0.05 * pulse})`;
      ctx.fill();
      // Core
      ctx.beginPath();
      ctx.arc(sx, sy, r * (p.big ? pulse : 1), 0, Math.PI * 2);
      ctx.fillStyle = p.big ? '#f0c878' : 'rgba(232, 200, 121, 0.9)';
      ctx.fill();
      if (p.big) {
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(255, 230, 180, 0.5)';
        ctx.stroke();
      }
    }
  }

  /* Ghost render */
  function renderGhosts() {
    for (const g of ghosts) {
      const sx = g.x + camX;
      const sy = g.y + camY + Math.sin(g.wobble) * 2;

      ctx.save();
      ctx.translate(sx, sy);

      // Body (dome + bumpy skirt)
      ctx.beginPath();
      ctx.arc(0, -2, 13, Math.PI, Math.PI * 2);
      // Right side down to skirt
      ctx.lineTo(13, 11);
      // Bumps (3 zigzags)
      const bumpY = 11;
      const baseY = 14;
      for (let i = 0; i < 3; i++) {
        const x1 = 13 - (i * 9) - 4;
        const x2 = 13 - (i * 9) - 9;
        ctx.lineTo(x1, baseY);
        ctx.lineTo(x2, bumpY);
      }
      ctx.lineTo(-13, baseY);
      ctx.lineTo(-13, -2);
      ctx.closePath();

      // Fill with subtle radial gradient for depth
      const grad = ctx.createRadialGradient(0, -4, 2, 0, 4, 14);
      grad.addColorStop(0, g.color);
      grad.addColorStop(1, shadeColor(g.color, -28));
      ctx.fillStyle = grad;
      ctx.shadowBlur = 16;
      ctx.shadowColor = g.color;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Eyes (whites)
      ctx.fillStyle = '#fbf6ec';
      ctx.beginPath();
      ctx.ellipse(-4.5, -4, 3.2, 4, 0, 0, Math.PI * 2);
      ctx.ellipse( 4.5, -4, 3.2, 4, 0, 0, Math.PI * 2);
      ctx.fill();

      // Pupils — look toward pac
      const lookX = pac.x - g.x;
      const lookY = pac.y - g.y;
      const ld = Math.hypot(lookX, lookY) || 1;
      const px = (lookX / ld) * 1.3;
      const py = (lookY / ld) * 1.6;
      ctx.fillStyle = '#0a0908';
      ctx.beginPath();
      ctx.arc(-4.5 + px, -4 + py, 1.6, 0, Math.PI * 2);
      ctx.arc( 4.5 + px, -4 + py, 1.6, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();
    }
  }

  function shadeColor(hex, percent) {
    // Lightens (positive) or darkens (negative) a hex color
    const f = parseInt(hex.slice(1), 16);
    const t = percent < 0 ? 0 : 255;
    const p = Math.abs(percent) / 100;
    const R = f >> 16;
    const G = (f >> 8) & 0x00FF;
    const B = f & 0x0000FF;
    const r = Math.round((t - R) * p) + R;
    const g = Math.round((t - G) * p) + G;
    const b = Math.round((t - B) * p) + B;
    return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
  }

  /* Pac render */
  function renderPac() {
    const sx = pac.x + camX;
    const sy = pac.y + camY;

    // Outer glow
    const glow = ctx.createRadialGradient(sx, sy, 0, sx, sy, PAC_R * 3.6);
    glow.addColorStop(0, 'rgba(232, 200, 121, 0.22)');
    glow.addColorStop(1, 'rgba(232, 200, 121, 0)');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(sx, sy, PAC_R * 3.6, 0, Math.PI * 2);
    ctx.fill();

    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(pac.dir);

    // Body
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, PAC_R, pac.mouth * Math.PI, (2 - pac.mouth) * Math.PI);
    ctx.closePath();
    const bodyGrad = ctx.createLinearGradient(-PAC_R, -PAC_R, PAC_R, PAC_R);
    bodyGrad.addColorStop(0, '#f0d290');
    bodyGrad.addColorStop(1, '#a87f44');
    ctx.fillStyle = bodyGrad;
    ctx.fill();

    // Shell ridge
    ctx.strokeStyle = 'rgba(90, 64, 32, 0.55)';
    ctx.lineWidth = 1.4;
    ctx.stroke();

    // Highlight
    ctx.beginPath();
    ctx.arc(-PAC_R * 0.2, -PAC_R * 0.3, PAC_R * 0.32, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 240, 200, 0.22)';
    ctx.fill();

    ctx.restore();

    pac.sx = sx; pac.sy = sy;
  }

  /* ═════════════════════════════════════════════════════
     MINIMAP
  ═════════════════════════════════════════════════════ */
  function drawMinimap() {
    if (!miniCtx || !miniCv) return;
    const mw = miniCv.width, mh = miniCv.height;
    miniCtx.clearRect(0, 0, mw, mh);

    const W = world.offsetWidth, H = world.offsetHeight;
    if (W === 0 || H === 0) return;
    const sx = mw / W, sy = mh / H;

    // Walls
    for (const w of walls) {
      miniCtx.fillStyle = w.el.classList.contains('visited')
        ? 'rgba(200, 163, 104, 0.5)'
        : 'rgba(156, 138, 118, 0.18)';
      miniCtx.fillRect(w.x * sx, w.y * sy, w.w * sx, w.h * sy);
    }

    // Path
    miniCtx.strokeStyle = 'rgba(200, 163, 104, 0.3)';
    miniCtx.lineWidth = 1;
    miniCtx.beginPath();
    for (let i = 0; i < pathPoints.length; i += 4) {
      const p = pathPoints[i];
      const xx = p.x * sx, yy = p.y * sy;
      if (i === 0) miniCtx.moveTo(xx, yy);
      else         miniCtx.lineTo(xx, yy);
    }
    miniCtx.stroke();

    // Ghosts
    for (const g of ghosts) {
      miniCtx.fillStyle = g.color;
      miniCtx.beginPath();
      miniCtx.arc(g.x * sx, g.y * sy, 2.4, 0, Math.PI * 2);
      miniCtx.fill();
    }

    // Pac
    miniCtx.fillStyle = '#f0d290';
    miniCtx.beginPath();
    miniCtx.arc(pac.x * sx, pac.y * sy, 3.2, 0, Math.PI * 2);
    miniCtx.fill();
    miniCtx.strokeStyle = 'rgba(255, 240, 200, 0.7)';
    miniCtx.lineWidth = 0.6;
    miniCtx.stroke();
  }

  /* ═════════════════════════════════════════════════════
     AUDIO (subtle)
  ═════════════════════════════════════════════════════ */
  let audioCtx = null;
  function ac() {
    if (audioCtx) return audioCtx;
    try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) {}
    return audioCtx;
  }
  let lastSfx = 0;
  function sfxCollect(big) {
    const now = performance.now();
    if (!big && now - lastSfx < 60) return; // throttle small pellet sfx
    lastSfx = now;
    const a = ac(); if (!a) return;
    const freqs = big ? [523, 659, 784, 1047] : [780];
    freqs.forEach((f, i) => {
      setTimeout(() => {
        const o = a.createOscillator(), g = a.createGain();
        o.connect(g); g.connect(a.destination);
        o.type = 'sine'; o.frequency.value = f;
        const vol = big ? 0.05 : 0.018;
        g.gain.setValueAtTime(vol, a.currentTime);
        g.gain.exponentialRampToValueAtTime(0.001, a.currentTime + (big ? 0.2 : 0.07));
        o.start(); o.stop(a.currentTime + (big ? 0.2 : 0.07));
      }, i * 55);
    });
  }

  /* ═════════════════════════════════════════════════════
     INPUT — native scroll + keyboard
  ═════════════════════════════════════════════════════ */
  function syncScrollProgress() {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    if (max <= 0) return;
    pacTargetProgress = Math.min(1, Math.max(0, window.scrollY / max));
  }

  function setupInput() {
    window.addEventListener('scroll', syncScrollProgress, { passive: true });
    // Initial sync (in case page was already scrolled)
    syncScrollProgress();

    document.addEventListener('keydown', e => {
      if (!gameStarted) return;
      const step = window.innerHeight * 0.4;
      if (e.key === 'ArrowDown' || e.key === 'ArrowRight' ||
          e.key === 's' || e.key === 'S' || e.key === 'd' || e.key === 'D' ||
          e.key === 'PageDown' || e.key === ' ') {
        e.preventDefault();
        window.scrollBy({ top: step, behavior: 'smooth' });
      } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft' ||
                 e.key === 'w' || e.key === 'W' || e.key === 'a' || e.key === 'A' ||
                 e.key === 'PageUp') {
        e.preventDefault();
        window.scrollBy({ top: -step, behavior: 'smooth' });
      } else if (e.key === 'Home') {
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      } else if (e.key === 'End') {
        e.preventDefault();
        window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' });
      }
    });
  }

  /* ═════════════════════════════════════════════════════
     RESIZE
  ═════════════════════════════════════════════════════ */
  let rtid;
  window.addEventListener('resize', () => {
    clearTimeout(rtid);
    rtid = setTimeout(() => {
      resizeCanvas();
      measureWorld();
      measureGrid();
      buildPath();
      // Re-place pellets and re-mark eaten for any behind current progress
      placePellets();
      pellets.forEach(p => {
        if (p.progress < pacProgress) { p.eaten = true; pelletsEaten++; }
      });
      if (hudCount) hudCount.textContent = pelletsEaten + ' / ' + totalPellets;
      initGhosts();
    }, 200);
  });

  /* ═════════════════════════════════════════════════════
     MAIN LOOP
  ═════════════════════════════════════════════════════ */
  function loop() {
    frameCount++;
    updatePac();
    updateGhosts();
    eatPellets();
    checkChapters();
    updateCamera();
    render();
    if (frameCount % 4 === 0) drawMinimap();
    requestAnimationFrame(loop);
  }

  /* ═════════════════════════════════════════════════════
     BOOT
  ═════════════════════════════════════════════════════ */
  function boot() {
    runIntro();
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
