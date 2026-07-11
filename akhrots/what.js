/* ═══════════════════════════════════════════════════════
   WHAT IS AKHROT — Story-driven scene choreography
   GSAP ScrollTrigger · line/word reveals · rail progress
   ═══════════════════════════════════════════════════════ */

(function () {
  'use strict';
  if (!document.body.classList.contains('what-body')) return;

  function waitFor(check, cb, interval) {
    if (check()) return cb();
    const t = setInterval(() => { if (check()) { clearInterval(t); cb(); } }, interval || 35);
  }

  waitFor(() => window.gsap && window.ScrollTrigger, init);

  function init() {
    gsap.registerPlugin(ScrollTrigger);

    const scenes        = document.querySelectorAll('.scene');
    const railFill      = document.querySelector('.rail-fill');
    const railChapters  = document.querySelectorAll('.rail-chapters li');
    const markerNum     = document.querySelector('.marker-num');
    const markerName    = document.querySelector('.marker-name');

    /* ── Set initial state for every reveal element ── */
    gsap.set('[data-reveal]', { opacity: 0, y: 28 });

    /* ── Scene-level reveal choreography ──
       For each scene we animate:
         1. The .scene-eyebrow / .scene-tag (small label)
         2. Each .line in the heading (word-by-word lift)
         3. Each [data-reveal] body element (staggered fade-up)
       The scene also flips a body class for in-CSS triggers
       (e.g. strike-anim, grow-underline). */
    scenes.forEach(scene => {
      const tag    = scene.querySelector('.scene-eyebrow, .scene-tag');
      const lines  = scene.querySelectorAll('.line');
      const reveals = scene.querySelectorAll('[data-reveal]');

      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: scene,
          start: 'top 78%',
          end: 'bottom 20%',
          toggleClass: { targets: scene, className: 'scene-active' },
          once: false
        }
      });

      if (tag) {
        tl.fromTo(tag,
          { opacity: 0, y: 16 },
          { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out' },
        0);
      }

      // Word-by-word lift inside each line
      lines.forEach((line, i) => {
        const words = line.querySelectorAll('.word');
        if (!words.length) return;
        tl.fromTo(words,
          { yPercent: 110 },
          { yPercent: 0, duration: 0.95, stagger: 0.06, ease: 'power3.out' },
          0.12 + i * 0.08);
      });

      if (reveals.length) {
        tl.fromTo(reveals,
          { opacity: 0, y: 28 },
          { opacity: 1, y: 0, duration: 0.85, stagger: 0.1, ease: 'power3.out' },
          0.35);
      }
    });

    /* ── Big quote scene — pinned moment, large reveal ── */
    const quoteScene = document.querySelector('.scene-quote');
    if (quoteScene) {
      const lines = quoteScene.querySelectorAll('.big-q-line');
      gsap.set(lines, { opacity: 0, y: 60 });
      ScrollTrigger.create({
        trigger: quoteScene,
        start: 'top 70%',
        once: true,
        onEnter: () => {
          gsap.to(lines, {
            opacity: 1, y: 0,
            duration: 1.1, stagger: 0.18, ease: 'power3.out'
          });
          quoteScene.classList.add('scene-active');
        }
      });
      // Subtle parallax on the entire stage as user scrolls past
      gsap.to('.quote-stage', {
        y: -40,
        ease: 'none',
        scrollTrigger: {
          trigger: quoteScene,
          start: 'top bottom',
          end: 'bottom top',
          scrub: 0.6
        }
      });
    }

    /* ── Big-item list — sequence each item with a brighter highlight ── */
    document.querySelectorAll('.big-list .big-item').forEach((item, i) => {
      gsap.fromTo(item.querySelector('.bi-num'),
        { opacity: 0, x: -14 },
        {
          opacity: 1, x: 0,
          duration: 0.65, ease: 'power2.out',
          scrollTrigger: { trigger: item, start: 'top 88%', once: true }
        });
      gsap.fromTo(item.querySelector('.bi-text'),
        { opacity: 0, y: 22 },
        {
          opacity: 1, y: 0,
          duration: 0.85, ease: 'power3.out',
          scrollTrigger: { trigger: item, start: 'top 88%', once: true }
        });
      gsap.fromTo(item.querySelector('.bi-line'),
        { scaleX: 0, transformOrigin: 'left' },
        {
          scaleX: 1, duration: 0.65, ease: 'power2.out',
          scrollTrigger: { trigger: item, start: 'top 85%', once: true }
        });
    });

    /* ── Pillars — staggered card lift ── */
    gsap.fromTo('.pillars .pillar',
      { opacity: 0, y: 40 },
      {
        opacity: 1, y: 0,
        duration: 0.95, stagger: 0.14, ease: 'power3.out',
        scrollTrigger: { trigger: '.pillars', start: 'top 80%', once: true }
      });

    /* ── Vision list ── */
    document.querySelectorAll('.vision-list li').forEach((li, i) => {
      gsap.fromTo(li,
        { opacity: 0, x: -20 },
        {
          opacity: 1, x: 0,
          duration: 0.75, delay: i * 0.06, ease: 'power3.out',
          scrollTrigger: { trigger: li, start: 'top 88%', once: true }
        });
    });

    /* ── Status panel rows ── */
    document.querySelectorAll('.status-list li').forEach((li, i) => {
      gsap.fromTo(li,
        { opacity: 0, x: -16 },
        {
          opacity: 1, x: 0,
          duration: 0.65, delay: i * 0.07, ease: 'power2.out',
          scrollTrigger: { trigger: '.status-panel', start: 'top 80%', once: true }
        });
    });

    /* ── Welcome title — letter-by-letter (re-split words to characters) ── */
    const welcomeTitle = document.querySelector('.welcome-title');
    if (welcomeTitle) {
      welcomeTitle.querySelectorAll('.word').forEach(w => {
        const txt = w.textContent;
        w.textContent = '';
        txt.split('').forEach(ch => {
          const s = document.createElement('span');
          s.className = 'char';
          s.style.display = 'inline-block';
          s.style.willChange = 'transform';
          s.textContent = ch === ' ' ? ' ' : ch;
          w.appendChild(s);
        });
      });
      const chars = welcomeTitle.querySelectorAll('.char');
      gsap.set(chars, { yPercent: 100, opacity: 0 });
      ScrollTrigger.create({
        trigger: welcomeTitle,
        start: 'top 80%',
        once: true,
        onEnter: () => {
          gsap.to(chars, {
            yPercent: 0, opacity: 1,
            duration: 0.85, stagger: 0.025, ease: 'power3.out'
          });
        }
      });
    }

    /* ── Welcome CTA pop ── */
    const cta = document.querySelector('.welcome-cta');
    if (cta) {
      gsap.fromTo(cta,
        { opacity: 0, y: 24, scale: 0.96 },
        {
          opacity: 1, y: 0, scale: 1,
          duration: 0.85, ease: 'back.out(1.4)',
          scrollTrigger: { trigger: cta, start: 'top 90%', once: true }
        });
    }

    /* ── Idea scene visual: counter-rotate rings on scroll for organic motion ── */
    const walnut = document.querySelector('.walnut-svg');
    if (walnut) {
      gsap.fromTo(walnut,
        { rotate: -10, scale: 0.9, opacity: 0 },
        {
          rotate: 0, scale: 1, opacity: 1,
          duration: 1.3, ease: 'power3.out',
          scrollTrigger: { trigger: walnut, start: 'top 85%', once: true }
        });
      // Subtle scroll-linked rotation
      gsap.to(walnut, {
        rotate: 12,
        ease: 'none',
        scrollTrigger: {
          trigger: walnut.closest('.scene'),
          start: 'top bottom',
          end: 'bottom top',
          scrub: 1.2
        }
      });
    }

    /* ── Vertical rail progress + chapter sync ── */
    const NAMES = Array.from(scenes).map(s => ({
      el: s,
      num: s.dataset.num || '',
      name: s.dataset.name || ''
    }));

    function onScroll() {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const pct = Math.min(1, Math.max(0, window.scrollY / Math.max(1, max)));
      if (railFill) railFill.style.height = (pct * 100).toFixed(2) + '%';

      // Find the scene whose center is closest to viewport center
      const vmid = window.scrollY + window.innerHeight * 0.4;
      let activeIdx = 0;
      let bestDist = Infinity;
      NAMES.forEach((s, i) => {
        const rc = s.el.getBoundingClientRect();
        const top = rc.top + window.scrollY;
        const bot = top + rc.height;
        const mid = (top + bot) / 2;
        const d = Math.abs(vmid - mid);
        if (d < bestDist) { bestDist = d; activeIdx = i; }
      });

      // Find chapter index by data-num
      const activeNum = NAMES[activeIdx].num;
      railChapters.forEach((li, i) => {
        const ch = li.dataset.chapter;
        if (ch === activeNum) {
          li.classList.add('active');
          li.classList.remove('passed');
        } else if (parseInt(ch, 10) < parseInt(activeNum, 10)) {
          li.classList.add('passed');
          li.classList.remove('active');
        } else {
          li.classList.remove('active');
          li.classList.remove('passed');
        }
      });

      if (markerNum)  markerNum.textContent  = NAMES[activeIdx].num;
      if (markerName) markerName.textContent = NAMES[activeIdx].name;
    }

    let rafPending = false;
    window.addEventListener('scroll', () => {
      if (rafPending) return;
      rafPending = true;
      requestAnimationFrame(() => {
        rafPending = false;
        onScroll();
      });
    }, { passive: true });

    /* Click on rail chapter to scroll to the matching scene */
    railChapters.forEach(li => {
      li.addEventListener('click', () => {
        const ch = li.dataset.chapter;
        const target = Array.from(scenes).find(s => s.dataset.num === ch);
        if (target) {
          window.scrollTo({
            top: target.offsetTop - 40,
            behavior: 'smooth'
          });
        }
      });
    });

    onScroll();

    /* Refresh ScrollTrigger once fonts are loaded so initial measurements are right */
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => ScrollTrigger.refresh());
    }
  }
})();
