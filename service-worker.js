const cacheName = 'v1';
const cacheAssets = [
  '/',
  '/index.html',
  '/css/style.css',
  '/js/main.js',
  '/icons/icon-192x192.png'
];

// Install Service Worker
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(cacheName).then(cache => {
      console.log('Caching files');
      return cache.addAll(cacheAssets);
    })
  );
});

// Fetch from Cache
self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(response => {
      return response || fetch(e.request);
    })
  );
});
