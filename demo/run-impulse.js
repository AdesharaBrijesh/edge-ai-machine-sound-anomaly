// Wrapper around the Edge Impulse WebAssembly build (edge-impulse-standalone.js / .wasm), based on
// the run-impulse.js Edge Impulse ships with the build. Differences: Module.init() is always called
// exactly once (the shipped version skips it if the runtime finished loading before init() was
// called, and without it the classifier's buffers are not set up: "Would write outside feature
// buffer"), and results only carry what an audio classifier needs.
// The runtime may already be up when this file loads (fast browsers compile the .wasm before the
// next <script> arrives); onRuntimeInitialized has then fired already, so check Module.calledRun too.
let classifierInitialized = Boolean(Module.calledRun);
let moduleInitCalled = false;
Module.onRuntimeInitialized = function () {
  classifierInitialized = true;
};

function initModuleOnce() {
  if (moduleInitCalled) return;
  moduleInitCalled = true;
  const ret = typeof Module.init === 'function' ? Module.init() : 0;
  if (typeof ret === 'number' && ret !== 0) throw new Error('Module.init() failed with code ' + ret);
}

class EdgeImpulseClassifier {
  init() {
    if (classifierInitialized || Module.calledRun) {
      classifierInitialized = true;
      initModuleOnce();
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      Module.onRuntimeInitialized = () => {
        classifierInitialized = true;
        try { initModuleOnce(); resolve(); } catch (e) { reject(e); }
      };
    });
  }

  getProperties() {
    if (!classifierInitialized) throw new Error('Module is not initialized');
    const raw = Module.get_properties();
    const proto = Module.emcc_classification_properties_t && Module.emcc_classification_properties_t.prototype;
    if (!proto) return raw;
    const props = {};
    for (const key of Object.getOwnPropertyNames(proto)) {
      const d = Object.getOwnPropertyDescriptor(proto, key);
      if (d && typeof d.get === 'function') props[key] = raw[key];
    }
    return props;
  }

  classify(rawData, debug = false) {
    if (!classifierInitialized) throw new Error('Module is not initialized');
    const obj = this._arrayToHeap(rawData);
    const ret = Module.run_classifier(obj.buffer.byteOffset, rawData.length, debug);
    Module._free(obj.ptr);
    if (ret.result !== 0) {
      throw new Error('Classification failed (err code: ' + ret.result + ')');
    }
    const jsResult = { anomaly: ret.anomaly, results: [] };
    for (let cx = 0; cx < ret.size(); cx++) {
      const c = ret.get(cx);
      jsResult.results.push({ label: c.label, value: c.value });
      c.delete();
    }
    ret.delete();
    return jsResult;
  }

  _arrayToHeap(data) {
    const typedArray = new Float32Array(data);
    const numBytes = typedArray.length * typedArray.BYTES_PER_ELEMENT;
    const ptr = Module._malloc(numBytes);
    const heapBytes = new Uint8Array(Module.HEAPU8.buffer, ptr, numBytes);
    heapBytes.set(new Uint8Array(typedArray.buffer));
    return { ptr: ptr, buffer: heapBytes };
  }
}
