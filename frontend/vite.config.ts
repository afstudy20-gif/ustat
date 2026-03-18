import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Polyfill plugin for Node.js built-ins that some npm packages reference.
 *
 * Covered modules:
 *   buffer / buffer/   – used by plotly.js (typedarray-pool, is-buffer, etc.)
 *   stream             – required by probe-image-size → pptxgenjs (Node path)
 *   assert             – required by stream-parser → pptxgenjs (Node path)
 *
 * We intercept these with virtual modules that provide just enough surface
 * for the code to load without crashing.  pptxgenjs itself is imported
 * dynamically in PlotExporter so it never blocks the app's initial render.
 */
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
      if (id === V_BUFFER) {
        return `
const Buf = globalThis.Buffer ?? (() => {
  function Buffer(arg) {
    if (typeof arg === 'number') return new Uint8Array(arg);
    if (typeof arg === 'string') return new TextEncoder().encode(arg);
    return new Uint8Array(arg);
  }
  Buffer.from = (a, enc) => {
    if (typeof a === 'string') {
      if (enc === 'base64') {
        const bin = atob(a), b = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) b[i] = bin.charCodeAt(i);
        return b;
      }
      return new TextEncoder().encode(a);
    }
    return new Uint8Array(a);
  };
  Buffer.isBuffer    = b  => b instanceof Uint8Array;
  Buffer.alloc       = n  => new Uint8Array(n);
  Buffer.allocUnsafe = n  => new Uint8Array(n);
  Buffer.concat      = bs => {
    const t = bs.reduce((s,b) => s+b.length, 0), o = new Uint8Array(t);
    let x = 0; for (const b of bs) { o.set(b,x); x+=b.length; }
    return o;
  };
  return Buffer;
})();
export { Buf as Buffer };
export default { Buffer: Buf };
`
      }

      if (id === V_STREAM) {
        // Minimal EventEmitter + Transform stub so stream-parser can import
        // without throwing.  The actual stream functionality is never called
        // in a browser PPTX export flow.
        return `
class EventEmitter {
  constructor() { this._ev = {}; }
  on(e,fn)   { (this._ev[e]??=[]).push(fn); return this; }
  emit(e,...a){ (this._ev[e]??[]).forEach(f=>f(...a)); return this; }
  removeListener(e,fn){ this._ev[e]=(this._ev[e]??[]).filter(f=>f!==fn); return this; }
}
class Stream extends EventEmitter { pipe(d){ return d; } }
class Readable extends Stream { read(){} }
class Writable extends Stream { write(){return true;} end(){} }
class Transform extends Writable { constructor(o){super();this._t=o;} }
class PassThrough extends Transform {}
function pipeline(...args){ const cb=args[args.length-1]; if(typeof cb==='function') cb(null); }
export { Readable, Writable, Transform, PassThrough, pipeline };
export default { Readable, Writable, Transform, PassThrough, pipeline, Stream };
`
      }

      if (id === V_ASSERT) {
        return `
function assert(v,msg){ if(!v) throw new Error(msg??'Assertion failed'); }
assert.ok      = assert;
assert.equal   = (a,b,m)=>assert(a==b,m);
assert.strictEqual=(a,b,m)=>assert(a===b,m);
assert.throws  = (fn)=>{ try{fn();}catch(e){return;} throw new Error('Expected throw'); };
assert.deepEqual=(a,b,m)=>{};
export default assert;
export { assert };
`
      }

      return null
    },
  }
}

export default defineConfig({
  plugins: [react(), nodePolyfills()],

  optimizeDeps: {
    include: ['plotly.js', 'react-plotly.js'],
  },

  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
