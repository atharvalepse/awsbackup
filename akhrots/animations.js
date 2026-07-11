/* ═══ AKHROT ANIMATIONS v5 — Professional Cursor + Refined Motion ═══ */
'use strict';

/* ── Utility ── */
function waitFor(check, cb, interval) {
  if (check()) return cb();
  const t = setInterval(() => { if (check()) { clearInterval(t); cb(); } }, interval || 30);
}
const onGSAP = cb => waitFor(() => window.gsap, cb);

/* ══════════════════════════════════════════════════════
   1. MINIMAL PROFESSIONAL CURSOR
   — Tight dot + lagging ring with smooth easing
   — Context-aware hover states (link / text / press)
   — Zero blur, zero glassmorphism, no excessive scaling
   — Touch devices skip the cursor entirely
══════════════════════════════════════════════════════ */
(function buildCursor() {
  if (matchMedia('(pointer: coarse)').matches) return;

  const dot  = document.getElementById('cursor-dot');
  const ring = document.getElementById('cursor-ring');
  if (!dot || !ring) return;

  let mx = -100, my = -100;
  let dx = -100, dy = -100;
  let rx = -100, ry = -100;
  let visible = false;
  let state = 'default';
  const body = document.body;

  document.addEventListener('mousemove', e => {
    mx = e.clientX; my = e.clientY;
    if (!visible) {
      dot.style.opacity = '1';
      ring.style.opacity = '1';
      visible = true;
    }
  }, { passive: true });

  document.addEventListener('mouseleave', () => {
    dot.style.opacity = '0';
    ring.style.opacity = '0';
    visible = false;
  });
  document.addEventListener('mouseenter', () => {
    if (mx > 0) { dot.style.opacity = '1'; ring.style.opacity = '1'; visible = true; }
  });

  /* Smooth tracking — tight dot, lagging ring */
  (function tick() {
    dx += (mx - dx) * 0.42;
    dy += (my - dy) * 0.42;
    rx += (mx - rx) * 0.16;
    ry += (my - ry) * 0.16;
    dot.style.transform  = `translate3d(${dx}px, ${dy}px, 0) translate(-50%, -50%)`;
    ring.style.transform = `translate3d(${rx}px, ${ry}px, 0) translate(-50%, -50%)`;
    requestAnimationFrame(tick);
  })();

  /* Context-aware state */
  function setState(name) {
    if (state === name) return;
    state = name;
    body.classList.toggle('cur-hover', name === 'hover');
    body.classList.toggle('cur-text',  name === 'text');
  }

  document.addEventListener('mouseover', e => {
    const t = e.target;
    if (!t || !t.closest) return;
    if (t.closest('a, button, [data-magnetic], .cta-link, .submit-btn, .outline-btn, .world-cta, .bento-cta-btn, input, textarea, label[for], [role="button"], .room-walnut, .pellet, .bento-card')) {
      setState('hover');
    } else if (t.closest('p, li, h1, h2, h3, h4, h5, h6, blockquote, span:not([class*="cursor"])')) {
      setState('text');
    } else {
      setState('default');
    }
  }, { passive: true });

  document.addEventListener('mousedown', () => body.classList.add('cur-press'));
  document.addEventListener('mouseup',   () => body.classList.remove('cur-press'));
})();

/* ══════════════════════════════════════════════════════
   2. AMBIENT POINTER GLOW — single RAF loop, low cost
══════════════════════════════════════════════════════ */
(function() {
  const el = document.getElementById('_akhrot_glow') || (() => {
    const d = document.createElement('div');
    d.id = '_akhrot_glow';
    d.setAttribute('aria-hidden','true');
    d.style.cssText = 'position:fixed;inset:0;z-index:0;pointer-events:none;will-change:background;transition:opacity 0.6s ease';
    document.body.prepend(d);
    return d;
  })();
  let gx = 0, gy = 0, tx = 0, ty = 0;
  document.addEventListener('mousemove', e => { tx = e.clientX; ty = e.clientY; }, { passive: true });
  (function g() {
    gx += (tx - gx) * 0.04;
    gy += (ty - gy) * 0.04;
    el.style.background = `radial-gradient(620px circle at ${gx}px ${gy}px, rgba(156, 138, 118, 0.012), transparent 55%)`;
    requestAnimationFrame(g);
  })();
})();

/* ══════════════════════════════════════════════════════
   3. NAVBAR — Smart hide / show
══════════════════════════════════════════════════════ */
(function() {
  const nav = document.getElementById('navbar');
  if (!nav) return;
  let last = 0, tid;
  window.addEventListener('scroll', () => {
    clearTimeout(tid);
    tid = setTimeout(() => {
      const y = window.scrollY;
      nav.style.transition = 'transform 0.55s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.45s';
      if (y > 80) {
        nav.style.transform = y > last ? 'translateX(-50%) translateY(-120%)' : 'translateX(-50%) translateY(0)';
      } else {
        nav.style.transform = 'translateX(-50%) translateY(0)';
      }
      last = y;
    }, 12);
  }, { passive: true });
})();

/* ══════════════════════════════════════════════════════
   4. GSAP SCROLL REVEALS + REFINED CARD TILT
══════════════════════════════════════════════════════ */
onGSAP(() => {
  if (!window.ScrollTrigger) return;
  gsap.registerPlugin(ScrollTrigger);

  /* 3D card tilt — gentler, more refined */
  document.querySelectorAll('.ab-card, .pac-card, .bento-card, .mosaic-card').forEach(card => {
    if (card.dataset.tilt) return;
    card.dataset.tilt = '1';
    card.style.transformStyle = 'preserve-3d';
    const sheen = document.createElement('div');
    sheen.style.cssText = 'position:absolute;inset:0;border-radius:inherit;background:radial-gradient(circle at 50% 0%, rgba(255,255,255,0.06), transparent 55%);pointer-events:none;opacity:0;z-index:2;transition:opacity 0.4s var(--ease)';
    card.appendChild(sheen);
    card.addEventListener('mousemove', e => {
      const rc = card.getBoundingClientRect();
      const x = ((e.clientX - rc.left) / rc.width  - 0.5) * 7;
      const y = ((e.clientY - rc.top)  / rc.height - 0.5) * -5;
      gsap.to(card, { rotateY: x, rotateX: y, scale: 1.012, duration: 0.45, ease: 'power2.out', transformPerspective: 1000 });
      sheen.style.opacity = '0.7';
    });
    card.addEventListener('mouseleave', () => {
      gsap.to(card, { rotateY: 0, rotateX: 0, scale: 1, duration: 0.8, ease: 'power3.out' });
      sheen.style.opacity = '0';
    });
  });

  /* Bento entrance */
  document.querySelectorAll('.bento-grid .bento-card').forEach((el, i) => {
    gsap.set(el, { y: 50, opacity: 0 });
    ScrollTrigger.create({
      trigger: el, start: 'top 90%', once: true,
      onEnter: () => gsap.to(el, { y: 0, opacity: 1, duration: 0.95, delay: i * 0.06, ease: 'power3.out' })
    });
  });

  /* Page headings — softer entrance */
  document.querySelectorAll('.page-heading, .about-headline, .hs-hero-title, .scene-heading, .what-heading').forEach(h => {
    if (h.dataset.anim) return; h.dataset.anim = '1';
    gsap.set(h, { opacity: 0, y: 40 });
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        gsap.to(h, { opacity: 1, y: 0, duration: 1.2, ease: 'power3.out' });
        obs.unobserve(h);
      }
    }, { threshold: 0.12 });
    obs.observe(h);
  });

  /* Eyebrows */
  document.querySelectorAll('.page-eyebrow, .about-eyebrow, .what-eyebrow, .scene-eyebrow').forEach(el => {
    if (el.dataset.anim) return; el.dataset.anim = '1';
    gsap.set(el, { opacity: 0, x: -16 });
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        gsap.to(el, { opacity: 1, x: 0, duration: 0.7, ease: 'power2.out' });
        obs.unobserve(el);
      }
    }, { threshold: 0.2 });
    obs.observe(el);
  });

  /* Landing phase titles — observe activations */
  const phaseObs = new MutationObserver(muts => {
    muts.forEach(m => {
      if (!m.target.classList.contains('active')) return;
      gsap.fromTo(m.target.querySelectorAll('.title-main'),
        { y: 50, opacity: 0, skewY: 1.5 },
        { y: 0, opacity: 1, skewY: 0, duration: 1.15, stagger: 0.1, ease: 'power3.out' });
      const ey = m.target.querySelector('.eyebrow');
      if (ey) gsap.fromTo(ey, { y: 18, opacity: 0 }, { y: 0, opacity: 1, duration: 0.6, ease: 'power2.out' });
    });
  });
  document.querySelectorAll('.phase-container').forEach(p => phaseObs.observe(p, { attributes: true, attributeFilter: ['class'] }));

  /* Trigger phase 1 immediately if already active on load */
  requestAnimationFrame(() => {
    const p1 = document.getElementById('phase-1');
    if (p1 && p1.classList.contains('active')) {
      gsap.fromTo(p1.querySelectorAll('.title-main'),
        { y: 50, opacity: 0, skewY: 1.5 },
        { y: 0, opacity: 1, skewY: 0, duration: 1.15, stagger: 0.1, ease: 'power3.out', delay: 0.2 });
      const ey = p1.querySelector('.eyebrow');
      if (ey) gsap.fromTo(ey, { y: 18, opacity: 0 }, { y: 0, opacity: 1, duration: 0.6, ease: 'power2.out', delay: 0.1 });
    }
  });

  /* Stat counters */
  document.querySelectorAll('[data-count]').forEach(el => {
    const n = +el.dataset.count;
    ScrollTrigger.create({
      trigger: el, start: 'top 96%', once: true,
      onEnter: () => {
        let obj = { v: 0 };
        gsap.to(obj, { v: n, duration: 2.2, ease: 'power2.out',
          onUpdate: () => { el.textContent = Math.round(obj.v).toLocaleString(); }
        });
      }
    });
  });
});

/* ══════════════════════════════════════════════════════
   5. MAGNETIC ELEMENTS — refined pull
══════════════════════════════════════════════════════ */
function initMagnetic() {
  document.querySelectorAll('[data-magnetic]').forEach(el => {
    if (el.dataset.mag) return; el.dataset.mag = '1';
    el.addEventListener('mousemove', e => {
      const rc = el.getBoundingClientRect();
      const x = (e.clientX - rc.left - rc.width / 2) * 0.28;
      const y = (e.clientY - rc.top  - rc.height / 2) * 0.28;
      onGSAP(() => gsap.to(el, { x, y, duration: 0.32, ease: 'power2.out' }));
    });
    el.addEventListener('mouseleave', () => {
      onGSAP(() => gsap.to(el, { x: 0, y: 0, duration: 0.9, ease: 'elastic.out(1, 0.45)' }));
    });
  });
}
initMagnetic();
window._reinitMagnetic = initMagnetic;

/* ══════════════════════════════════════════════════════
   6. VELOCITY.JS — refined form interactions
══════════════════════════════════════════════════════ */
waitFor(() => window.Velocity, () => {
  document.querySelectorAll('.waitlist-form input, .waitlist-form textarea').forEach(inp => {
    inp.addEventListener('focus', () => Velocity(inp, { borderBottomColor: '#9c8a76', paddingLeft: '4px' }, { duration: 240, easing: 'easeOutQuad' }));
    inp.addEventListener('blur',  () => Velocity(inp, { paddingLeft: '0px' }, { duration: 220 }));
  });
  function vFooter() {
    document.querySelectorAll('.footer-nav-link:not([data-v])').forEach(link => {
      link.dataset.v = '1';
      link.addEventListener('mouseenter', () => Velocity(link, { translateX: '6px' }, { duration: 200, easing: 'easeOutQuad' }));
      link.addEventListener('mouseleave', () => Velocity(link, { translateX: '0px' }, { duration: 240, easing: 'easeOutQuad' }));
    });
  }
  setTimeout(vFooter, 600);
  document.addEventListener('footer-injected', vFooter);
});

/* ══════════════════════════════════════════════════════
   7. FOOTER TICKER — seamless GSAP marquee
══════════════════════════════════════════════════════ */
function initTicker() {
  onGSAP(() => {
    const inner = document.querySelector('.footer-ticker-inner');
    if (!inner || inner.dataset.tickerGSAP) return;
    inner.dataset.tickerGSAP = '1';
    const track = inner.parentNode;
    if (!track.querySelector('.footer-ticker-inner + .footer-ticker-inner')) {
      track.appendChild(inner.cloneNode(true));
    }
    const all = track.querySelectorAll('.footer-ticker-inner');
    gsap.killTweensOf(all);
    gsap.fromTo(all, { x: '0%' }, { x: '-50%', duration: 28, repeat: -1, ease: 'none' });
  });
}
initTicker();
setTimeout(initTicker, 700);
document.addEventListener('footer-injected', initTicker);

/* ══════════════════════════════════════════════════════
   8. INTERSECTION OBSERVERS — fade-in + stagger-in
══════════════════════════════════════════════════════ */
(function() {
  const fObs = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
  }, { threshold: 0.08 });
  document.querySelectorAll('.fade-in').forEach(el => fObs.observe(el));

  let idx = 0;
  const sObs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting && !e.target.classList.contains('visible')) {
        setTimeout(() => e.target.classList.add('visible'), (idx++ % 8) * 70);
      }
    });
  }, { threshold: 0.06, rootMargin: '0px 0px -25px 0px' });
  document.querySelectorAll('.stagger-in').forEach(el => sObs.observe(el));
})();

/* ══════════════════════════════════════════════════════
   9. LOADER EXIT — fast and reliable
══════════════════════════════════════════════════════ */
(function() {
  const loader = document.getElementById('loader');
  if (!loader) return;

  window.hideLoader = function() {
    if (loader.classList.contains('done')) return;
    onGSAP(() => {
      gsap.to(loader, {
        opacity: 0, duration: 0.85, ease: 'power2.inOut',
        onComplete: () => {
          loader.classList.add('done');
          document.body.classList.remove('loading');
        }
      });
    });
    // Fallback if GSAP fails to load
    setTimeout(() => {
      if (!loader.classList.contains('done')) {
        loader.style.opacity = '0';
        loader.classList.add('done');
        document.body.classList.remove('loading');
      }
    }, 1200);
  };
})();

/* ══════════════════════════════════════════════════════
   10. D3 NEURAL NET — footer ornament
══════════════════════════════════════════════════════ */
function buildD3() {
  if (!window.d3) return;
  const col = document.querySelector('.footer-brand-col');
  if (!col || col.querySelector('svg.d3-neural')) return;
  const W = 200, H = 70;
  const svg = d3.select(col).insert('svg', ':first-child')
    .attr('class', 'd3-neural').attr('width', W).attr('height', H)
    .style('opacity', '0.16').style('display', 'block').style('margin-bottom', '1.2rem');
  const pts = Array.from({ length: 12 }, () => ({ x: 10 + Math.random() * (W - 20), y: 6 + Math.random() * (H - 12) }));
  pts.forEach((n, i) => {
    const j = (i + 1 + Math.floor(Math.random() * 2)) % pts.length;
    svg.append('line').attr('x1', n.x).attr('y1', n.y).attr('x2', pts[j].x).attr('y2', pts[j].y)
      .attr('stroke', 'rgba(156, 138, 118, 0.5)').attr('stroke-width', 0.6);
  });
  const circles = svg.selectAll('circle').data(pts).enter().append('circle')
    .attr('cx', d => d.x).attr('cy', d => d.y).attr('r', 2).attr('fill', 'rgba(156, 138, 118, 0.9)');
  (function pulse() {
    circles.transition().duration(1600 + Math.random() * 800).attr('r', 3.2).attr('fill', 'rgba(200, 163, 104, 1)')
      .transition().duration(1200).attr('r', 2).attr('fill', 'rgba(156, 138, 118, 0.9)').on('end', pulse);
  })();
}
setTimeout(buildD3, 800);
document.addEventListener('footer-injected', () => setTimeout(buildD3, 100));

/* ══════════════════════════════════════════════════════
   11. RESIZE — debounced ScrollTrigger refresh
══════════════════════════════════════════════════════ */
let _resizeTid;
window.addEventListener('resize', () => {
  clearTimeout(_resizeTid);
  _resizeTid = setTimeout(() => { if (window.ScrollTrigger) ScrollTrigger.refresh(); }, 250);
}, { passive: true });
