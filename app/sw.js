const CACHE_NAME = 'cementsight-v1';
const ASSETS = [
  '/',
  '/app/dashboard.html',
  '/app/tailwind.css',
  '/app/input.css'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => response || fetch(event.request))
  );
});
