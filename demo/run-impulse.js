// Wrapper around the Edge Impulse WebAssembly build (edge-impulse-standalone.js / .wasm),
// following the documented EdgeImpulseClassifier from the Edge Impulse "WebAssembly (browser)"
// deployment guide. The standalone script defines the global `Module`.
let classifierInitialized = false;
Module.onRuntimeInitialized = function () {
  classifierInitialized = true;
};

class EdgeImpulseClassifier {
  init() {
    if (classifierInitialized) return Promise.resolve();
    return new Promise((resolve) => {
      Module.onRuntimeInitialized = () => {
        classifierInitialized = true;
        resolve();
      };
    });
  }

  getProperties() {
    if (!classifierInitialized) throw new Error('Module is not initialized');
    return Module.get_properties();
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
