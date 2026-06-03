import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import { writeFileSync, mkdirSync } from 'node:fs'
import { resolve } from 'node:path'

// Build-time version stamp emitted to dist/version.json so the in-app
// UpdatePrompt can poll it as a fallback when the service worker is
// unavailable. Hash = current ISO build time (changes every build) so any
// new deploy is detected immediately.
const BUILD_TIME = new Date().toISOString()
const APP_VERSION = process.env.npm_package_version ?? '0.0.0'

function emitVersionJson(): Plugin {
  return {
    name: 'emit-version-json',
    apply: 'build',
    closeBundle() {
      try {
        const out = resolve(__dirname, 'dist')
        mkdirSync(out, { recursive: true })
        writeFileSync(
          resolve(out, 'version.json'),
          JSON.stringify({ version: APP_VERSION, build: BUILD_TIME }, null, 2),
        )
      } catch (e) {
        console.warn('[emit-version-json] failed:', e)
      }
    },
  }
}

function nodePolyfills(): Plugin {
  const V_BUFFER = '\0node-polyfill:buffer'
  const V_STREAM = '\0node-polyfill:stream'
  const V_ASSERT = '\0node-polyfill:assert'
  return {
    name: 'node-polyfills',
    resolveId(id) {
      if (id === 'buffer' || id === 'buffer/') return V_BUFFER
      if (id === 'stream' || id === 'stream/') return V_STREAM
      if (id === 'assert' || id === 'assert/') return V_ASSERT
      return null
    },
    load(id) {
      if (id === V_BUFFER) return `
const Buf = globalThis.Buffer ?? (() => {
  function Buffer(arg) {
    if (typeof arg === 'number') return new Uint8Array(arg);
    if (typeof arg === 'string') return new TextEncoder().encode(arg);
    return new Uint8Array(arg);
  }
  Buffer.from = (a, enc) => {
    if (typeof a === 'string') {
      if (enc === 'base64') { const bin=atob(a),b=new Uint8Array(bin.length); for(let i=0;i<bin.length;i++) b[i]=bin.charCodeAt(i); return b; }
      return new TextEncoder().encode(a);
    }
    return new Uint8Array(a);
  };
  Buffer.isBuffer=b=>b instanceof Uint8Array; Buffer.alloc=n=>new Uint8Array(n);
  Buffer.allocUnsafe=n=>new Uint8Array(n);
  Buffer.concat=bs=>{ const t=bs.reduce((s,b)=>s+b.length,0),o=new Uint8Array(t); let x=0; for(const b of bs){o.set(b,x);x+=b.length;} return o; };
  return Buffer;
})();
export { Buf as Buffer }; export default { Buffer: Buf };`

      if (id === V_STREAM) return `
class EE { constructor(){this._e={};} on(e,f){(this._e[e]??=[]).push(f);return this;} emit(e,...a){(this._e[e]??[]).forEach(f=>f(...a));} removeListener(e,f){this._e[e]=(this._e[e]??[]).filter(x=>x!==f);return this;} }
class Stream extends EE { pipe(d){return d;} }
class Readable extends Stream { read(){} }
class Writable extends Stream { write(){return true;} end(){} }
class Transform extends Writable { constructor(o){super();this._t=o;} }
class PassThrough extends Transform {}
function pipeline(...args){const cb=args[args.length-1];if(typeof cb==='function')cb(null);}
export {Readable,Writable,Transform,PassThrough,pipeline};
export default {Readable,Writable,Transform,PassThrough,pipeline,Stream};`

      if (id === V_ASSERT) return `
function assert(v,m){if(!v)throw new Error(m??'Assertion failed');}
assert.ok=assert;assert.equal=(a,b,m)=>assert(a==b,m);assert.strictEqual=(a,b,m)=>assert(a===b,m);
assert.throws=fn=>{try{fn();}catch(e){return;}throw new Error('Expected throw');};assert.deepEqual=()=>{};
export default assert;export {assert};`

      return null
    },
  }
}

export default defineConfig({
  plugins: [
    react(),
    nodePolyfills(),
    emitVersionJson(),
    VitePWA({
      registerType: 'autoUpdate',
      // 'prompt' would surface needRefresh always; 'autoUpdate' silently
      // installs in the background and exposes needRefresh too, so the
      // UpdatePrompt component can show a toast on top of it.
      includeAssets: ['logo.png', 'pwa-192.png', 'pwa-512.png'],
      manifest: {
        name: 'uSTAT - Statistical Analysis',
        short_name: 'uSTAT',
        description: 'SPSS-like statistical analysis in your browser',
        theme_color: '#6366f1',
        background_color: '#ffffff',
        display: 'standalone',
        orientation: 'any',
        start_url: '/',
        scope: '/',
        icons: [
          { src: '/pwa-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' },
          { src: '/pwa-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' },
          { src: '/pwa-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        maximumFileSizeToCacheInBytes: 10 * 1024 * 1024,
        globPatterns: ['**/*.{js,css,html,png,svg,ico,woff2}'],
        // version.json carries the build stamp; it must NEVER be precached,
        // otherwise the fallback poll always sees the old version.
        globIgnores: ['**/version.json'],
        // Don't claim navigation requests for /api/* (FastAPI backend).
        navigateFallbackDenylist: [/^\/api\//],
        // Activate the new SW as soon as it finishes installing — without
        // this, users would need TWO reloads to pick up a deploy.
        skipWaiting: true,
        clientsClaim: true,
        runtimeCaching: [
          {
            urlPattern: /^https?:\/\/.*\/api\//,
            handler: 'NetworkFirst',
            options: { cacheName: 'api-cache', expiration: { maxEntries: 50, maxAgeSeconds: 300 } },
          },
          {
            // Always fetch the version manifest fresh; never serve from cache.
            urlPattern: /\/version\.json$/,
            handler: 'NetworkOnly',
          },
        ],
      },
    }),
  ],

  define: {
    global: 'globalThis',
    __APP_VERSION__: JSON.stringify(APP_VERSION),
    __BUILD_TIME__: JSON.stringify(BUILD_TIME),
  },

  optimizeDeps: {
    include: ['plotly.js', 'react-plotly.js', 'xlsx'],
  },

  server: {
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } },
  },
})
