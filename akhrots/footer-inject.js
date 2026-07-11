/* footer-inject.js — Minimal full-width logo footer */
(function () {
  'use strict';

  const FOOTER = `
<footer id="main-footer" class="akhrot-footer">

  <!-- Ticker -->
  <div class="footer-ticker">
    <div class="footer-ticker-track">
      <div class="footer-ticker-inner">
        <span class="ticker-item">Intelligence was never artificial</span>
        <span class="ticker-dot"></span>
        <span class="ticker-item">Nature-inspired AI</span>
        <span class="ticker-dot"></span>
        <span class="ticker-item">Built by dreamers</span>
        <span class="ticker-dot"></span>
        <span class="ticker-item">Akhrot — 2025</span>
        <span class="ticker-dot"></span>
        <span class="ticker-item">The future feels different</span>
        <span class="ticker-dot"></span>
        <span class="ticker-item">Intelligence was never artificial</span>
        <span class="ticker-dot"></span>
        <span class="ticker-item">Nature-inspired AI</span>
        <span class="ticker-dot"></span>
        <span class="ticker-item">Built by dreamers</span>
        <span class="ticker-dot"></span>
      </div>
    </div>
  </div>

  <!-- Full-width logo -->
  <div class="footer-logo-section">
    <img src="LOGO.svg" alt="Akhrot" class="footer-full-logo" />
  </div>

  <!-- Tagline -->
  <div class="footer-tagline-row">
    <p>Intelligence was never artificial.</p>
  </div>

  <!-- Bottom bar -->
  <div class="footer-bottom-bar">
    <span>&copy; 2025 Akhrot. All rights reserved.</span>
    <div class="footer-status-pill">
      <span class="footer-status-dot"></span>
      <span>All systems operational</span>
    </div>
    <span>Built with intention.</span>
  </div>

</footer>`;

  /* ── Inject ── */
  const el = document.getElementById('main-footer');
  if (el) { el.outerHTML = FOOTER; } else { document.body.insertAdjacentHTML('beforeend', FOOTER); }

  /* ── Re-init magnetic ── */
  setTimeout(() => { if (window._reinitMagnetic) window._reinitMagnetic(); }, 100);

  /* ── Ticker clone for seamless loop ── */
  setTimeout(() => {
    const inner = document.querySelector('.footer-ticker-inner');
    const track = inner && inner.parentNode;
    if (track && !track.querySelector('.footer-ticker-inner + .footer-ticker-inner')) {
      track.appendChild(inner.cloneNode(true));
    }
  }, 50);

  /* Dispatch event */
  setTimeout(() => document.dispatchEvent(new Event('footer-injected')), 80);

})();
