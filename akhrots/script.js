(function () {
  'use strict';

  /* ── CRT scanline overlay (subtle, global) ── */
  if (!document.querySelector('.crt-overlay')) {
    const crt = document.createElement('div');
    crt.className = 'crt-overlay';
    crt.setAttribute('aria-hidden', 'true');
    document.body.appendChild(crt);
  }

  /* ── Navbar visibility safety ── */
  const _nav = document.getElementById('navbar');
  if (_nav && !document.body.classList.contains('loading')) {
    _nav.style.opacity = '1';
  }

  /* ═════════════════════════════════════════════════════
     PHASE MAP
  ═════════════════════════════════════════════════════ */
  const PHASES = [
    { id: 'phase-1', start: 0,  end: 18 },
    { id: 'phase-2', start: 18, end: 38 },
    { id: 'phase-3', start: 38, end: 58 },
    { id: 'phase-4', start: 58, end: 78 },
    { id: 'phase-5', start: 78, end: 100 }
  ];

  const scrollContainer = document.getElementById('scroll-container');
  const scrollHint = document.getElementById('scroll-hint');

  let currentActivePhase = -1;

  function resetPhaseAnimations(el) {
    // Clear any stale inline styles so the CSS .active rules can take over
    // when this phase re-activates. We intentionally do NOT set opacity:0 inline,
    // because that would stick and prevent re-activation from making text visible.
    const seqs = el.querySelectorAll('.seq');
    seqs.forEach(s => {
      s.style.transition = '';
      s.style.opacity = '';
      s.style.transform = '';
    });
  }

  function updatePhases(pct) {
    let targetPhase = -1;
    PHASES.forEach((phase, i) => {
      const inRange = pct >= phase.start && pct < phase.end;
      const isLastActive = (i === PHASES.length - 1) && pct >= phase.start;
      if (inRange || isLastActive) targetPhase = i;
    });
    if (targetPhase === -1) targetPhase = 0;

    if (targetPhase === currentActivePhase) return;

    PHASES.forEach((phase, i) => {
      const el = document.getElementById(phase.id);
      if (!el) return;
      if (i !== targetPhase) {
        const wasActive = el.classList.contains('active');
        el.classList.remove('active');
        if (i < targetPhase) {
          el.classList.remove('exit-backward');
          el.classList.add('exit-forward');
        } else {
          el.classList.remove('exit-forward');
          el.classList.add('exit-backward');
        }
        if (wasActive) resetPhaseAnimations(el);
      }
    });

    const targetEl = document.getElementById(PHASES[targetPhase].id);
    if (targetEl && !targetEl.classList.contains('active')) {
      targetEl.classList.remove('exit-forward', 'exit-backward');
      void targetEl.offsetWidth;
      targetEl.classList.add('active');
    }
    currentActivePhase = targetPhase;
  }

  function getScrollProgress() {
    if (!scrollContainer) return 0;
    const max = scrollContainer.scrollHeight - window.innerHeight;
    if (max <= 0) return 0;
    return Math.min(100, Math.max(0, (window.scrollY / max) * 100));
  }

  /* ═════════════════════════════════════════════════════
     SMOOTH SCROLL HANDLER (RAF-throttled)
  ═════════════════════════════════════════════════════ */
  let rafPending = false;
  function onScroll() {
    if (!scrollContainer) return;
    if (!rafPending) {
      rafPending = true;
      requestAnimationFrame(() => {
        rafPending = false;
        const pct = getScrollProgress();
        if (scrollHint) scrollHint.classList.toggle('hidden', pct > 2);
        updatePhases(pct);

        if (pct >= 80) {
          document.body.classList.add('reveal-active');
        } else {
          document.body.classList.remove('reveal-active');
          document.documentElement.style.setProperty('--mask-r', '0px');
        }

        if (window._videoDuration && window._video &&
            (!window._videoSeekable || window._videoSeekable())) {
          const t = (pct / 100) * window._videoDuration;
          const clamped = Math.max(0, Math.min(window._videoDuration, t));

          // Throttle seeks — 60fps on touch (events less frequent), 30fps on desktop
          const seekInterval = isTouchDevice ? 16 : 33;
          const now = performance.now();
          if (now - (window._lastSeekTime || 0) >= seekInterval) {
            window._lastSeekTime = now;
            const v = window._video;
            if (typeof v.fastSeek === 'function') {
              v.fastSeek(clamped);
            } else {
              v.currentTime = clamped;
            }
          }
        }
      });
    }
  }

  if (scrollContainer) {
    window.addEventListener('scroll', onScroll, { passive: true });
    // Ensure phase-1 is active and animated on load — critical for first-paint visibility
    currentActivePhase = -1;
    updatePhases(0);
  }

  /* ═════════════════════════════════════════════════════
     VIDEO INIT — XHR Preloader for perfect smoothness
  ═════════════════════════════════════════════════════ */
  const video = document.getElementById('main-video');
  const isTouchDevice = window.matchMedia('(pointer: coarse)').matches;

  if (video) {
    window._video = video;
    window._videoDuration = 0;
    let videoReady = false;
    let videoSeekable = false;

    // We can use WebM on both since Chrome/Safari/Firefox support it, but
    // let's follow the data attributes. (WebM is 2.8MB, much faster to download)
    const videoUrl = isTouchDevice ? video.getAttribute('data-src-webm') : (video.getAttribute('data-src-mp4') || video.getAttribute('data-src-webm'));
    const progressBar = document.querySelector('.loader-progress-bar span');

    if (videoUrl) {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', videoUrl, true);
      xhr.responseType = 'blob';

      xhr.onprogress = function(e) {
        if (e.lengthComputable && progressBar) {
          const percentComplete = (e.loaded / e.total) * 100;
          progressBar.style.width = percentComplete + '%';
        }
      };

      xhr.onload = function() {
        if (this.status === 200) {
          const blob = this.response;
          const objectUrl = URL.createObjectURL(blob);
          video.src = objectUrl;

          video.muted = true;
          video.playsInline = true;

          function setVideoReady() {
            if (videoReady) return;
            if (video.duration && isFinite(video.duration) && video.duration > 0.1) {
              window._videoDuration = video.duration;
              videoReady = true;
              videoSeekable = true;
              video.pause();
              
              // The video is 100% downloaded and ready — dismiss the loader
              if (window.hideLoader) {
                setTimeout(window.hideLoader, 200);
              }
            }
          }

          ['loadedmetadata', 'loadeddata', 'canplay', 'canplaythrough', 'durationchange'].forEach(evt => {
            video.addEventListener(evt, setVideoReady);
          });
          
          video.addEventListener('canplay', () => { videoSeekable = true; });
          video.addEventListener('canplaythrough', () => { videoSeekable = true; });

          video.load();
          
          // Prime video for mobile to ensure seekability on iOS/Android
          if (isTouchDevice) {
            const primeVideo = () => {
              const p = video.play();
              if (p && p.then) {
                p.then(() => { video.pause(); video.currentTime = 0; setVideoReady(); }).catch(() => {});
              }
            };
            setTimeout(primeVideo, 100);
            document.addEventListener('touchstart', primeVideo, { once: true, passive: true });
          } else {
            setTimeout(() => {
              if (videoReady) return;
              const p = video.play();
              if (p && p.then) {
                p.then(() => { video.pause(); video.currentTime = 0; setVideoReady(); }).catch(() => {});
              }
            }, 100);
          }
          
          // Safety fallback to hide loader if events fail
          setTimeout(() => { if (window.hideLoader) window.hideLoader(); }, 2500);

        } else {
          // Fallback if XHR fails (e.g. CORS or 404)
          video.src = videoUrl;
          video.load();
          if (window.hideLoader) window.hideLoader();
        }
      };

      xhr.onerror = function() {
        video.src = videoUrl;
        video.load();
        if (window.hideLoader) window.hideLoader();
      };

      xhr.send();
    } else {
      if (window.hideLoader) window.hideLoader();
    }

    window._videoSeekable = () => videoSeekable;
  }

  /* ═════════════════════════════════════════════════════
     MASK X/Y FOLLOWS POINTER (reveal layer)
  ═════════════════════════════════════════════════════ */
  document.addEventListener('mousemove', (e) => {
    if (document.body.classList.contains('reveal-active')) {
      const root = document.documentElement;
      root.style.setProperty('--mask-x', ((e.clientX / window.innerWidth)  * 100).toFixed(2) + '%');
      root.style.setProperty('--mask-y', ((e.clientY / window.innerHeight) * 100).toFixed(2) + '%');
      root.style.setProperty('--mask-r', '250px');
    }
  }, { passive: true });

  /* ═════════════════════════════════════════════════════
     FLIP LINK INIT
  ═════════════════════════════════════════════════════ */
  document.querySelectorAll('.flip-link').forEach(link => {
    const text = link.textContent.trim();
    link.textContent = '';
    const createChars = (text, className) => {
      const wrapper = document.createElement('div');
      wrapper.className = className;
      text.split('').forEach((char, i) => {
        const span = document.createElement('span');
        span.className = 'char';
        span.textContent = char === ' ' ? ' ' : char;
        span.style.transitionDelay = `${i * 20}ms`;
        wrapper.appendChild(span);
      });
      return wrapper;
    };
    link.appendChild(createChars(text, 'char-wrapper char-front'));
    link.appendChild(createChars(text, 'char-wrapper char-back'));
  });

  /* ═════════════════════════════════════════════════════
     LOADER SAFETY NET
  ═════════════════════════════════════════════════════ */
  window.addEventListener('load', () => {
    const loader = document.getElementById('loader');
    if (loader) {
      setTimeout(() => {
        if (!loader.classList.contains('done')) {
          loader.style.opacity = '0';
          loader.classList.add('done');
          document.body.classList.remove('loading');
        }
      }, 3500);
    }
  });

  /* ═════════════════════════════════════════════════════
     FADE-IN + STAGGER OBSERVERS
  ═════════════════════════════════════════════════════ */
  const fadeEls = document.querySelectorAll('.fade-in');
  if (fadeEls.length > 0) {
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add('visible');
      });
    }, { threshold: 0.12 });
    fadeEls.forEach(el => obs.observe(el));
  }

  const staggerEls = document.querySelectorAll('.stagger-in');
  if (staggerEls.length > 0) {
    let staggerIdx = 0;
    const staggerObs = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && !entry.target.classList.contains('visible')) {
          const delay = (staggerIdx++ % 6) * 80;
          setTimeout(() => entry.target.classList.add('visible'), delay);
        }
      });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });
    staggerEls.forEach(el => staggerObs.observe(el));
  }

  /* ═════════════════════════════════════════════════════
     RESIZE
  ═════════════════════════════════════════════════════ */
  let rTid;
  window.addEventListener('resize', () => {
    clearTimeout(rTid);
    rTid = setTimeout(onScroll, 200);
  });

  /* ═════════════════════════════════════════════════════
     GRID ANIMATION BACKGROUND
  ═════════════════════════════════════════════════════ */
  const gridCanvas = document.getElementById('grid-canvas');
  if (gridCanvas) {
    const gridCtx = gridCanvas.getContext('2d');
    const SPACING = 38, STROKE_LEN = 11, STROKE_W = 0.8;

    let gW = 0, gH = 0, gCols = 0, gRows = 0;
    let gridRAF = null, gridRunning = false;
    const gBall = { x: 0, y: 0, vx: 0, vy: 0, targetX: 0, targetY: 0 };
    let gMouseOver = false;

    function resizeGrid() {
      gW = window.innerWidth; gH = window.innerHeight;
      gridCanvas.width = gW; gridCanvas.height = gH;
      gCols = Math.ceil(gW / SPACING) + 1;
      gRows = Math.ceil(gH / SPACING) + 1;
      if (!gMouseOver) {
        gBall.x = gW / 2; gBall.y = gH / 2;
        gBall.targetX = gBall.x; gBall.targetY = gBall.y;
      }
    }

    let resizeTid;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTid);
      resizeTid = setTimeout(resizeGrid, 150);
    });
    resizeGrid();

    document.addEventListener('mousemove', e => {
      gMouseOver = true;
      gBall.targetX = Math.round(e.clientX / SPACING) * SPACING;
      gBall.targetY = Math.round(e.clientY / SPACING) * SPACING;
    });
    document.addEventListener('mouseleave', () => {
      gMouseOver = false;
      gBall.targetX = gW / 2; gBall.targetY = gH / 2;
    });

    function stepGrid() {
      gBall.vx += (gBall.targetX - gBall.x) * 0.12;
      gBall.vy += (gBall.targetY - gBall.y) * 0.12;
      gBall.vx *= 0.78; gBall.vy *= 0.78;
      gBall.x += gBall.vx; gBall.y += gBall.vy;

      gridCtx.clearRect(0, 0, gW, gH);
      gridCtx.strokeStyle = 'rgba(156, 138, 118, 0.42)';
      gridCtx.lineWidth = STROKE_W;
      gridCtx.beginPath();
      for (let col = 0; col < gCols; col++) {
        for (let row = 0; row < gRows; row++) {
          const px = col * SPACING, py = row * SPACING;
          const dx = gBall.x - px, dy = gBall.y - py;
          const distSq = dx * dx + dy * dy;
          if (distSq < 225) continue;
          const inv = STROKE_LEN / Math.sqrt(distSq);
          gridCtx.moveTo(px, py);
          gridCtx.lineTo(px - dx * inv, py - dy * inv);
        }
      }
      gridCtx.stroke();
      gridCtx.fillStyle = 'rgba(156, 138, 118, 0.72)';
      gridCtx.beginPath();
      gridCtx.arc(gBall.x, gBall.y, 3.5, 0, Math.PI * 2);
      gridCtx.fill();

      gridRAF = requestAnimationFrame(stepGrid);
    }

    function startGrid() { if (gridRunning) return; gridRunning = true; stepGrid(); }
    function stopGrid()  { gridRunning = false; if (gridRAF) { cancelAnimationFrame(gridRAF); gridRAF = null; } }

    document.addEventListener('visibilitychange', () => {
      document.hidden ? stopGrid() : startGrid();
    });
    startGrid();
  }

})();
