/* PyCaLiAI PWA service worker — network-first (never serve stale bets when online),
   cache only as an offline fallback. Bump CACHE to invalidate old caches on deploy. */
const CACHE = 'pycaliai-v2';

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  e.respondWith((async () => {
    try {
      // no-store: fetch() with the default cache mode can be silently satisfied by the
      // browser's own HTTP cache (heuristic freshness, since the origin sends no
      // Cache-Control) without a real network round-trip — defeating "network-first"
      // even though this handler's own logic never touches a stale value on purpose.
      const net = await fetch(req, { cache: "no-store" });
      // Populate the offline fallback cache with same-origin successful GETs only.
      if (net && net.ok && new URL(req.url).origin === self.location.origin) {
        const c = await caches.open(CACHE);
        c.put(req, net.clone());
      }
      return net;
    } catch (err) {
      const cached = await caches.match(req);
      if (cached) return cached;
      throw err;
    }
  })());
});
