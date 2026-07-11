/* ═══ AKHROT PAC-MAN — Scroll-Driven Walnut Theme ═══ */
(function () {
  'use strict';

  const CELL = 18;
  const GHOST_SPD = 1.2;
  const POWER_DUR = 280;

  /* ── Walnut color palette ── */
  const C = {
    pac:       '#C8A96E',
    pacShell:  '#8B6340',
    pellet:    'rgba(200,169,110,0.75)',
    power:     '#D4A84B',
    ghosts:    ['#8B6340','#6B4226','#A0785A','#5C3A1E'],  // walnut shell tones
    scared:    '#4A6B8A',
    nutMeat:   '#D4B896',
  };

  let canvas, ctx;
  let mazeW = 0, mazeH = 0, maze = [];
  let pellets = [], powerPellets = [];
  let pac, ghosts = [];
  let score = 0, lives = 3, hiScore = 0;
  let scared = false, scaredTimer = 0;
  let state = 'idle';
  let mouthA = 0.18, mouthDir = 1;
  let frameCount = 0;
  let lastScrollY = 0;
  let scrollVelocity = 0;
  let audioCtx = null;

  /* ── Audio ── */
  function ac() { return audioCtx || (audioCtx = new (window.AudioContext || window.webkitAudioContext || function(){})()); }
  function tone(f, d, v) {
    try {
      const o = ac().createOscillator(), g = ac().createGain();
      o.connect(g); g.connect(ac().destination);
      o.type = 'sine'; o.frequency.value = f;
      g.gain.setValueAtTime(v || 0.08, ac().currentTime);
      g.gain.exponentialRampToValueAtTime(0.001, ac().currentTime + d);
      o.start(); o.stop(ac().currentTime + d);
    } catch(e) {}
  }
  function sfxEat()   { tone(523, 0.05, 0.06); }
  function sfxPower() { [220,330,440].forEach((f,i) => setTimeout(() => tone(f, 0.15, 0.12), i*80)); }
  function sfxDeath() { [440,330,220,165,110].forEach((f,i) => setTimeout(() => tone(f, 0.18, 0.13), i*100)); }
  function sfxGhost() { tone(660, 0.08, 0.1); }

  /* ── Build maze from the bento grid gaps ── */
  function buildMaze() {
    const grid = document.getElementById('pac-bento-grid');
    const wrapper = document.getElementById('pacman-game-wrapper');
    if (!grid || !wrapper) return false;

    const wRect = wrapper.getBoundingClientRect();
    if (wRect.width < 100 || wRect.height < 80) return false;

    canvas.width  = wRect.width;
    canvas.height = wRect.height;

    mazeW = Math.ceil(wRect.width  / CELL);
    mazeH = Math.ceil(wRect.height / CELL);

    maze = Array.from({ length: mazeH }, () => new Array(mazeW).fill(0));

    const cards = grid.querySelectorAll('.pac-card');
    cards.forEach(card => {
      const r = card.getBoundingClientRect();
      const pad = 4;
      const c0 = Math.floor((r.left - wRect.left + pad) / CELL);
      const r0 = Math.floor((r.top  - wRect.top  + pad) / CELL);
      const c1 = Math.ceil( (r.left - wRect.left + r.width  - pad) / CELL);
      const r1 = Math.ceil( (r.top  - wRect.top  + r.height - pad) / CELL);
      for (let row = Math.max(0, r0); row < Math.min(mazeH, r1); row++)
        for (let col = Math.max(0, c0); col < Math.min(mazeW, c1); col++)
          maze[row][col] = 1;
    });

    pellets = []; powerPellets = [];
    for (let row = 0; row < mazeH; row++) {
      for (let col = 0; col < mazeW; col++) {
        if (maze[row][col] !== 0) continue;
        const isPwr = (col % 14 === 3 && row % 10 === 3);
        if (isPwr) powerPellets.push({ col, row, eaten: false });
        else       pellets.push({ col, row, eaten: false });
      }
    }
    return pellets.length > 5;
  }

  function freePt() {
    for (let tries = 0; tries < 2000; tries++) {
      const c = Math.floor(Math.random() * mazeW);
      const r = Math.floor(Math.random() * mazeH);
      if (maze[r] && maze[r][c] === 0) return { x: c * CELL + CELL/2, y: r * CELL + CELL/2 };
    }
    return { x: CELL * 1.5, y: CELL * 1.5 };
  }

  function initEntities() {
    const sp = freePt();
    pac = { x: sp.x, y: sp.y, dx: 0, dy: 0, ndx: 0, ndy: 0, r: 8, scrollDir: 'right' };
    ghosts = C.ghosts.map((color, i) => {
      const gp = freePt();
      return { x: gp.x, y: gp.y, dx: (i%2?1:-1)*GHOST_SPD, dy: (i<2?1:-1)*GHOST_SPD, color, eaten: false };
    });
  }

  function isWall(x, y) {
    const c = Math.floor(x / CELL), r = Math.floor(y / CELL);
    if (c < 0 || r < 0 || c >= mazeW || r >= mazeH) return true;
    return maze[r][c] === 1;
  }
  function wallAt(x, y, margin) {
    return isWall(x - margin, y) || isWall(x + margin, y) ||
           isWall(x, y - margin) || isWall(x, y + margin);
  }

  /* ── Scroll-driven movement ── */
  function handleScroll() {
    if (state !== 'playing') return;
    const currentScrollY = window.scrollY;
    scrollVelocity = currentScrollY - lastScrollY;
    lastScrollY = currentScrollY;

    const speed = Math.min(Math.abs(scrollVelocity) * 0.8, 6);
    if (speed < 0.3) return;

    // Scroll down = move right, scroll up = move left
    if (scrollVelocity > 0) {
      pac.ndx = speed; pac.ndy = 0; pac.scrollDir = 'right';
    } else {
      pac.ndx = -speed; pac.ndy = 0; pac.scrollDir = 'left';
    }
  }

  function movePac() {
    const m = pac.r - 2;
    const nx = pac.x + pac.ndx, ny = pac.y + pac.ndy;
    if (!wallAt(nx, ny, m)) {
      pac.dx = pac.ndx; pac.dy = pac.ndy;
    } else {
      // Try vertical alternatives when blocked horizontally
      const tryDirs = pac.ndx !== 0 ?
        [[0, Math.abs(pac.ndx)], [0, -Math.abs(pac.ndx)]] :
        [[Math.abs(pac.ndy), 0], [-Math.abs(pac.ndy), 0]];
      for (const [tdx, tdy] of tryDirs) {
        if (!wallAt(pac.x + tdx, pac.y + tdy, m)) {
          pac.dx = tdx; pac.dy = tdy;
          break;
        }
      }
    }
    const mx = pac.x + pac.dx, my = pac.y + pac.dy;
    if (!wallAt(mx, my, m)) { pac.x = mx; pac.y = my; }

    // Dampen movement
    pac.ndx *= 0.92;
    pac.ndy *= 0.92;
    pac.dx *= 0.95;
    pac.dy *= 0.95;

    // Wrap edges
    if (pac.x < 0) pac.x = canvas.width;
    if (pac.x > canvas.width) pac.x = 0;
    if (pac.y < 0) pac.y = canvas.height;
    if (pac.y > canvas.height) pac.y = 0;
  }

  function moveGhost(g) {
    if (g.eaten) return;
    const spd = scared ? GHOST_SPD * 0.5 : GHOST_SPD;
    const m = 6;
    const nx = g.x + g.dx * (spd / GHOST_SPD);
    const ny = g.y + g.dy * (spd / GHOST_SPD);
    if (!wallAt(nx, ny, m)) { g.x = nx; g.y = ny; return; }

    const dirs = [[1,0],[-1,0],[0,1],[0,-1]].map(([a,b]) => [a*spd, b*spd]);
    const valid = dirs.filter(([ddx, ddy]) => !wallAt(g.x + ddx*2, g.y + ddy*2, m));
    if (!valid.length) return;
    let chosen;
    if (!scared && Math.random() > 0.35 && frameCount % 3 === 0) {
      chosen = valid.reduce((best, d) => {
        const dist  = Math.hypot(g.x + d[0] - pac.x, g.y + d[1] - pac.y);
        const bdist = Math.hypot(g.x + best[0] - pac.x, g.y + best[1] - pac.y);
        return dist < bdist ? d : best;
      });
    } else {
      chosen = valid[Math.floor(Math.random() * valid.length)];
    }
    [g.dx, g.dy] = chosen;
    g.x += g.dx; g.y += g.dy;
    if (g.x < 0) g.x = canvas.width;
    if (g.x > canvas.width) g.x = 0;
    if (g.y < 0) g.y = canvas.height;
    if (g.y > canvas.height) g.y = 0;
  }

  function checkPellets() {
    const SNAP = CELL * 0.6;
    pellets.forEach(p => {
      if (p.eaten) return;
      if (Math.hypot(pac.x - (p.col*CELL+CELL/2), pac.y - (p.row*CELL+CELL/2)) < SNAP) {
        p.eaten = true; score += 10; sfxEat(); updateHUD();
      }
    });
    powerPellets.forEach(p => {
      if (p.eaten) return;
      if (Math.hypot(pac.x - (p.col*CELL+CELL/2), pac.y - (p.row*CELL+CELL/2)) < SNAP+3) {
        p.eaten = true; score += 50; scared = true; scaredTimer = POWER_DUR; sfxPower(); updateHUD();
      }
    });
    if (pellets.every(p=>p.eaten) && powerPellets.every(p=>p.eaten)) {
      state = 'win'; score += 500; updateHUD();
    }
  }

  function checkGhosts() {
    ghosts.forEach(g => {
      if (g.eaten) return;
      if (Math.hypot(pac.x - g.x, pac.y - g.y) < 12) {
        if (scared) {
          g.eaten = true; score += 200; sfxGhost(); updateHUD();
          setTimeout(() => { g.eaten = false; const p = freePt(); g.x=p.x; g.y=p.y; }, 3000);
        } else {
          lives--; sfxDeath(); updateHUD();
          if (lives <= 0) { state = 'gameover'; return; }
          const sp = freePt(); pac.x=sp.x; pac.y=sp.y;
        }
      }
    });
  }

  /* ── Draw walnut pac-man (with nut texture) ── */
  function drawWalnut(x, y, r, mouthOpen, angle) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);

    // Outer shell
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, r, mouthOpen * Math.PI, (2 - mouthOpen) * Math.PI);
    ctx.closePath();
    ctx.fillStyle = C.pac;
    ctx.shadowColor = C.pac;
    ctx.shadowBlur = 14;
    ctx.fill();

    // Shell ridge
    ctx.strokeStyle = C.pacShell;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Walnut brain texture lines
    if (mouthOpen < 0.25) {
      ctx.beginPath();
      ctx.arc(-r*0.15, -r*0.2, r*0.35, -0.6*Math.PI, 0.2*Math.PI);
      ctx.strokeStyle = 'rgba(139,99,64,0.5)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(r*0.15, r*0.1, r*0.3, -Math.PI*0.9, -0.1*Math.PI);
      ctx.stroke();
    }
    ctx.restore();
  }

  /* ── Draw walnut-shell ghost ── */
  function drawGhost(g) {
    if (g.eaten) return;
    const color = scared ? C.scared : g.color;
    const r = 9;
    ctx.save();

    // Walnut shell shape (more oval, textured)
    ctx.beginPath();
    ctx.ellipse(g.x, g.y - 2, r, r * 1.1, 0, Math.PI, 0);
    // Cracked bottom edge
    for (let i = 0; i <= 5; i++) {
      const sx = (g.x - r) + (i * r * 2 / 5);
      const sy = g.y + r * 0.8;
      const cp = g.y + (i % 2 === 0 ? r * 0.4 : r * 0.1);
      if (i === 0) ctx.lineTo(sx, sy);
      else ctx.quadraticCurveTo(sx - r*0.2, cp, sx, sy);
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = scared ? 4 : 10;
    ctx.fill();

    // Shell texture lines
    if (!scared) {
      ctx.shadowBlur = 0;
      ctx.strokeStyle = 'rgba(0,0,0,0.15)';
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      ctx.arc(g.x, g.y - 3, r * 0.5, -0.8 * Math.PI, 0.3 * Math.PI);
      ctx.stroke();

      // Eyes (small dark dots on the shell)
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(g.x-3, g.y-3, 2.5, 0, Math.PI*2); ctx.fill();
      ctx.beginPath(); ctx.arc(g.x+3, g.y-3, 2.5, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = '#3d2f22';
      ctx.beginPath(); ctx.arc(g.x-2.5, g.y-2.5, 1.2, 0, Math.PI*2); ctx.fill();
      ctx.beginPath(); ctx.arc(g.x+3.5, g.y-2.5, 1.2, 0, Math.PI*2); ctx.fill();
    } else {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(g.x-5,g.y-4); ctx.lineTo(g.x-2,g.y-1); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(g.x-2,g.y-4); ctx.lineTo(g.x-5,g.y-1); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(g.x+2,g.y-4); ctx.lineTo(g.x+5,g.y-1); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(g.x+5,g.y-4); ctx.lineTo(g.x+2,g.y-1); ctx.stroke();
    }
    ctx.restore();
  }

  /* ── Draw pellets as tiny walnut pieces ── */
  function drawNutPellet(x, y, big) {
    ctx.save();
    if (big) {
      // Power pellet = whole walnut
      const pulse = 0.5 + 0.5 * Math.sin(frameCount / 14);
      const sz = 4 + pulse * 2;
      ctx.beginPath();
      ctx.ellipse(x, y, sz, sz * 0.85, 0, 0, Math.PI * 2);
      ctx.fillStyle = C.power;
      ctx.shadowColor = C.power; ctx.shadowBlur = 10 + pulse * 6;
      ctx.fill();
      // Walnut ridge line
      ctx.strokeStyle = 'rgba(139,99,64,0.6)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, y - sz * 0.7);
      ctx.lineTo(x, y + sz * 0.7);
      ctx.stroke();
    } else {
      // Small pellet = nut crumb
      ctx.beginPath();
      ctx.ellipse(x, y, 2.2, 1.8, frameCount * 0.001, 0, Math.PI * 2);
      ctx.fillStyle = C.pellet;
      ctx.shadowColor = C.pellet; ctx.shadowBlur = 3;
      ctx.fill();
    }
    ctx.shadowBlur = 0;
    ctx.restore();
  }

  /* ── Main draw ── */
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Subtle path tint
    ctx.fillStyle = 'rgba(158,143,127,0.015)';
    for (let row = 0; row < mazeH; row++)
      for (let col = 0; col < mazeW; col++)
        if (maze[row][col] === 0)
          ctx.fillRect(col*CELL, row*CELL, CELL, CELL);

    // Pellets
    pellets.forEach(p => {
      if (p.eaten) return;
      drawNutPellet(p.col*CELL+CELL/2, p.row*CELL+CELL/2, false);
    });

    // Power pellets
    powerPellets.forEach(p => {
      if (p.eaten) return;
      drawNutPellet(p.col*CELL+CELL/2, p.row*CELL+CELL/2, true);
    });

    // Ghosts
    ghosts.forEach(drawGhost);

    // Pac-Man
    if (state !== 'gameover') {
      mouthA += 0.07 * mouthDir;
      if (mouthA > 0.35 || mouthA < 0.02) mouthDir *= -1;
      const angle = Math.atan2(pac.dy, pac.dx || (pac.scrollDir === 'right' ? 1 : -1));
      drawWalnut(pac.x, pac.y, pac.r, mouthA, angle);
    }

    if (state !== 'playing') drawOverlay();
  }

  function drawOverlay() {
    const W = canvas.width, H = canvas.height;
    ctx.save();
    ctx.fillStyle = 'rgba(6,5,4,0.78)';
    ctx.fillRect(0, 0, W, H);
    ctx.textAlign = 'center';

    if (state === 'idle') {
      ctx.fillStyle = 'rgba(200,169,110,0.95)';
      ctx.font = 'bold 20px "Clash Display",sans-serif';
      ctx.fillText('🌰  SCROLL TO PLAY', W/2, H/2 - 18);
      ctx.fillStyle = 'rgba(240,233,223,0.45)';
      ctx.font = '12px Satoshi,sans-serif';
      ctx.fillText('Scroll down to move · Arrow keys also work · Space to pause', W/2, H/2 + 12);
    } else if (state === 'gameover') {
      ctx.fillStyle = '#E8A598';
      ctx.font = 'bold 26px "Clash Display",sans-serif';
      ctx.fillText('SHELL CRACKED!', W/2, H/2 - 22);
      ctx.fillStyle = 'rgba(240,233,223,0.7)';
      ctx.font = '13px Satoshi,sans-serif';
      ctx.fillText('Score: ' + score + '   Best: ' + hiScore, W/2, H/2 + 6);
      ctx.fillText('Press SPACE or scroll to restart', W/2, H/2 + 28);
    } else if (state === 'win') {
      ctx.fillStyle = C.power;
      ctx.font = 'bold 26px "Clash Display",sans-serif';
      ctx.fillText('YOU CRACKED THE WALNUT! 🌰', W/2, H/2 - 22);
      ctx.fillStyle = 'rgba(240,233,223,0.7)';
      ctx.font = '13px Satoshi,sans-serif';
      ctx.fillText('Score: ' + score, W/2, H/2 + 6);
      ctx.fillText('Press SPACE or scroll to play again', W/2, H/2 + 28);
    } else if (state === 'paused') {
      ctx.fillStyle = 'rgba(200,169,110,0.9)';
      ctx.font = 'bold 22px "Clash Display",sans-serif';
      ctx.fillText('PAUSED', W/2, H/2);
    }
    ctx.restore();
  }

  function updateHUD() {
    if (score > hiScore) hiScore = score;
    const s = document.getElementById('pac-score-val');
    const l = document.getElementById('pac-lives-val');
    const h = document.getElementById('pac-high-val');
    if (s) s.textContent = score;
    if (l) l.innerHTML = '🌰'.repeat(Math.max(0, lives));
    if (h) h.textContent = hiScore;
  }

  /* ── Game loop ── */
  let raf;
  function loop() {
    frameCount++;
    if (state === 'playing') {
      movePac();
      ghosts.forEach(moveGhost);
      checkPellets();
      checkGhosts();
      if (scared) { scaredTimer--; if (scaredTimer <= 0) scared = false; }
    }
    draw();
    raf = requestAnimationFrame(loop);
  }

  function startGame() {
    if (!buildMaze()) { setTimeout(startGame, 300); return; }
    score = 0; lives = 3; scared = false; scaredTimer = 0; frameCount = 0;
    initEntities();
    updateHUD();
    state = 'playing';
    lastScrollY = window.scrollY;
  }

  /* ── Input ── */
  function setupInput() {
    const PAC_SPD = 2.2;
    const keyMap = {
      ArrowLeft:[-PAC_SPD,0], ArrowRight:[PAC_SPD,0], ArrowUp:[0,-PAC_SPD], ArrowDown:[0,PAC_SPD],
      a:[-PAC_SPD,0], d:[PAC_SPD,0], w:[0,-PAC_SPD], s:[0,PAC_SPD],
      A:[-PAC_SPD,0], D:[PAC_SPD,0], W:[0,-PAC_SPD], S:[0,PAC_SPD],
    };
    document.addEventListener('keydown', e => {
      if (keyMap[e.key]) {
        e.preventDefault();
        if (state === 'playing') [pac.ndx, pac.ndy] = keyMap[e.key];
      }
      if (e.code === 'Space') {
        e.preventDefault();
        if (state === 'idle' || state === 'gameover' || state === 'win') startGame();
        else if (state === 'playing') state = 'paused';
        else if (state === 'paused') state = 'playing';
      }
    });

    // Scroll-driven movement
    window.addEventListener('scroll', () => {
      if (state === 'idle' || state === 'gameover' || state === 'win') {
        startGame();
        return;
      }
      handleScroll();
    }, { passive: true });

    // Touch
    let tx = 0, ty = 0;
    const w = document.getElementById('pacman-game-wrapper');
    if (w) {
      w.addEventListener('touchstart', e => { tx = e.touches[0].clientX; ty = e.touches[0].clientY; }, {passive:true});
      w.addEventListener('touchend', e => {
        if (state === 'idle' || state === 'gameover' || state === 'win') { startGame(); return; }
        const dx = e.changedTouches[0].clientX - tx, dy = e.changedTouches[0].clientY - ty;
        if (Math.abs(dx) > Math.abs(dy)) { pac.ndx = dx > 0 ? PAC_SPD : -PAC_SPD; pac.ndy = 0; }
        else { pac.ndy = dy > 0 ? PAC_SPD : -PAC_SPD; pac.ndx = 0; }
      }, {passive:true});
    }

    window.addEventListener('resize', () => {
      clearTimeout(window._pacResizeTid);
      window._pacResizeTid = setTimeout(() => {
        if (state === 'playing') startGame();
        else { buildMaze(); }
      }, 400);
    }, { passive: true });
  }

  /* ── Init ── */
  function init() {
    const wrapper = document.getElementById('pacman-game-wrapper');
    if (!wrapper) return;

    canvas = document.createElement('canvas');
    canvas.id = 'pacman-overlay';
    canvas.style.cssText = 'position:absolute;top:0;left:0;z-index:20;pointer-events:auto;border-radius:inherit;display:block';
    wrapper.appendChild(canvas);
    ctx = canvas.getContext('2d');

    if (buildMaze()) {
      initEntities();
      state = 'idle';
    }
    loop();
    setupInput();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => setTimeout(init, 300));
  else setTimeout(init, 300);

})();
