#!/usr/bin/env python3
"""Create, configure, train and test the impulse through the Edge Impulse API.

Follows Edge Impulse's official Python API-bindings example (edgeimpulse/notebooks,
python-api-bindings-example.ipynb), so the whole Studio workflow runs from this laptop.

  python scripts/build_impulse.py info                 # devices, data summary, current impulse
  python scripts/build_impulse.py target --device "nano 33"   # on-device estimate target
  python scripts/build_impulse.py create               # (re)create the main impulse (default impulse)
  python scripts/build_impulse.py configure [--autotune]
  python scripts/build_impulse.py features
  python scripts/build_impulse.py train
  python scripts/build_impulse.py test
  python scripts/build_impulse.py all [--autotune]     # create -> configure -> features -> train -> test
  python scripts/build_impulse.py reference            # extra experiment with the reference project's settings

Main impulse: audio 16 kHz, 1000 ms window / 500 ms stride
  MFE -> 1D CNN classifier (normal / abnormal / background)
  Spectral Analysis -> GMM anomaly detection (scores sounds unlike the training data)
Metrics are written to results/*.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import edgeimpulse_api as ei

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, EIClient, get_config  # noqa: E402

API_HOST = "https://studio.edgeimpulse.com/v1"
RESULTS = ROOT / "results"

# Block IDs are project-wide, so the reference experiment uses its own range.
MAIN = {"input": 1, "mfe": 2, "spectral": 3, "classifier": 4, "anomaly": 5}
REF = {"input": 11, "mfe": 12, "classifier": 14}

MFE_CONFIG = {
    "frame_length": 0.032, "frame_stride": 0.016, "num_filters": 40, "fft_length": 512,
    "low_frequency": 0, "high_frequency": 8000, "win_size": 101, "noise_floor_db": -72,
}
SPECTRAL_CONFIG = {
    "scale-axes": 1, "input-decimation-ratio": 1, "filter-type": "none", "analysis-type": "FFT",
    "fft-length": 128, "do-log": True, "do-fft-overlap": True, "extra-low-freq": False,
}


# Edge Impulse's own generated script for the visual model below, with ONE change: the
# training set is shuffled completely every epoch. Studio's default shuffle buffer (batch x 4)
# cannot mix the class-ordered windows Edge Impulse feeds in, so every batch held one class and
# the network never learned (loss stuck at ln 2 = 0.693). See docs/report, section 7.
EXPERT_SCRIPT = """import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, AveragePooling2D, BatchNormalization, Permute, ReLU, Softmax, Activation, Cropping1D
from tensorflow.keras.optimizers.legacy import Adam

# Only the MFE features feed the network. (If extra features follow them in the input, they are
# cropped off. Do not rely on that for int8: one input scale for mixed-range features breaks
# quantisation, which is why the final impulse uses a single MFE block for both learn blocks.)
MFE_FEATURES = {MFE_FEATURES}
EPOCHS = {EPOCHS}
LEARNING_RATE = {LEARNING_RATE}
ENSURE_DETERMINISM = args.ensure_determinism
BATCH_SIZE = {BATCH_SIZE}
TIME_BUDGET_S = {TIME_BUDGET_S}   # stop fitting after this many seconds (0 = no limit); compute is metered


import time
class TimeBudget(tf.keras.callbacks.Callback):
    # Stops training when the wall-clock budget is used up and restores the epoch with the lowest val_loss.
    def __init__(self, seconds):
        super().__init__()
        self.seconds, self.best, self.best_loss, self.best_epoch = seconds, None, float('inf'), 0
    def on_train_begin(self, logs=None):
        self.t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        val_loss = (logs or {}).get('val_loss')
        if val_loss is not None and val_loss < self.best_loss:
            self.best_loss, self.best, self.best_epoch = val_loss, self.model.get_weights(), epoch + 1
        if self.seconds and time.time() - self.t0 > self.seconds:
            print('Time budget of %d s reached after epoch %d; stopping' % (self.seconds, epoch + 1), flush=True)
            self.model.stop_training = True
    def on_train_end(self, logs=None):
        if self.best is not None:
            self.model.set_weights(self.best)
            print('Restored weights of epoch %d (val_loss %.4f)' % (self.best_epoch, self.best_loss), flush=True)

# Full shuffle (the default buffer of BATCH_SIZE*4 leaves the class-ordered data in single-class batches)
train_dataset = train_dataset.shuffle(buffer_size=train_sample_count, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

# model architecture (same as the visual editor: Reshape -> 2 x [Conv1D + MaxPool + Dropout] -> Dense)
model = Sequential()
if input_length > MFE_FEATURES:
    model.add(Reshape((input_length, 1), input_shape=(input_length, )))
    model.add(Cropping1D(cropping=(0, input_length - MFE_FEATURES)))   # keep the MFE features only
    model.add(Reshape((int(MFE_FEATURES / {NUM_FILTERS}), {NUM_FILTERS})))
else:
    model.add(Reshape((int(input_length / {NUM_FILTERS}), {NUM_FILTERS}), input_shape=(input_length, )))
model.add(Conv1D(16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
model.add(Dropout(0.25))
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(classes, name='y_pred', activation='softmax'))

opt = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999)
callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS, ensure_determinism=ENSURE_DETERMINISM))
callbacks.append(TimeBudget(TIME_BUDGET_S))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, verbose=2, callbacks=callbacks, class_weight=ei_tensorflow.training.get_class_weights(Y_train))

disable_per_channel_quantization = False
"""


def keras_config(num_filters: int, epochs: int, lr: float, augment: bool = False, expert: bool = True,
                 mfe_features: int = 2440, batch_size: int = 32, time_budget_s: int = 0) -> dict:
    if expert:
        return {
            "mode": "expert",
            # expert mode ignores trainingCycles/learningRate, so they are written into the script itself
            "script": (EXPERT_SCRIPT.replace("{NUM_FILTERS}", str(num_filters))
                       .replace("{MFE_FEATURES}", str(mfe_features))
                       .replace("{BATCH_SIZE}", str(batch_size))
                       .replace("{TIME_BUDGET_S}", str(time_budget_s))
                       .replace("{EPOCHS}", str(epochs)).replace("{LEARNING_RATE}", repr(lr))),
            "trainingCycles": epochs,
            "learningRate": lr,
            "batchSize": batch_size,
            "trainTestSplit": 0.2,
            "autoClassWeights": True,
            "profileInt8": True,
            "skipEmbeddingsAndMemory": False,
        }
    return {
        "mode": "visual",
        "trainingCycles": epochs,
        "learningRate": lr,
        "batchSize": 32,
        "trainTestSplit": 0.2,          # validation fraction (0.8 here would validate on 80 %)
        "autoClassWeights": True,
        "profileInt8": True,
        "skipEmbeddingsAndMemory": False,
        "augmentationPolicySpectrogram": {
            "enabled": augment, "warping": False,
            "freqMasking": "low", "timeMasking": "low", "gaussianNoise": "low",
        },
        "visualLayers": [
            {"type": "reshape", "columns": num_filters},
            {"type": "conv1d", "neurons": 16, "kernelSize": 3, "stack": 1},
            {"type": "dropout", "dropoutRate": 0.25},
            {"type": "conv1d", "neurons": 32, "kernelSize": 3, "stack": 1},
            {"type": "dropout", "dropoutRate": 0.25},
            {"type": "flatten"},
        ],
    }


class Studio:
    def __init__(self):
        key, pid = get_config()
        self.pid = int(pid)
        # Plain JSON client for training calls: the generated request models serialise extra
        # default fields (e.g. assertGpuUsagePercentage) that the API refuses from normal users.
        self.http = EIClient(key, pid)
        client = ei.ApiClient(ei.Configuration(host=API_HOST, api_key={"ApiKeyAuthentication": key}))
        self.projects = ei.ProjectsApi(client)
        self.impulse = ei.ImpulseApi(client)
        self.dsp = ei.DSPApi(client)
        self.jobs = ei.JobsApi(client)
        self.learn = ei.LearnApi(client)
        self.classify = ei.ClassifyApi(client)
        self.deployment = ei.DeploymentApi(client)

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def ok(resp, what: str):
        if getattr(resp, "success", True) is False:
            raise RuntimeError(f"{what} failed: {getattr(resp, 'error', resp)}")
        return resp

    def wait(self, job_id: int, what: str, timeout: float = 3600) -> list[str]:
        start, last = time.time(), 0.0
        while True:
            job = self.ok(self.jobs.get_job_status(self.pid, job_id), "job status").job
            if job.finished:
                lines = self.log(job_id)
                took = time.time() - start
                if not job.finished_successful:
                    print("\n".join(lines[-30:]))
                    raise RuntimeError(f"{what}: job {job_id} failed after {took:.0f}s "
                                       f"(https://studio.edgeimpulse.com/studio/{self.pid}/jobs#show-job-{job_id})")
                print(f"[{what}] job {job_id} done in {took:.0f}s")
                return lines
            if time.time() - start > timeout:
                raise TimeoutError(f"{what}: job {job_id} still running after {timeout:.0f}s")
            if time.time() - last > 30:
                print(f"[{what}] job {job_id} running ({time.time() - start:.0f}s)")
                last = time.time()
            time.sleep(5)

    def log(self, job_id: int) -> list[str]:
        resp = self.jobs.get_jobs_logs(self.pid, job_id).to_dict().get("stdout", [])
        resp = sorted(resp, key=lambda e: str(e.get("created", "")))
        return "".join(e.get("data", "") for e in resp).splitlines()

    @staticmethod
    def metrics_from_log(lines: list[str], variant: str | None = None) -> dict | None:
        """Parse 'calculate_classification_metrics{...}' lines (as in the official notebook)."""
        tag, found, metrics = "calculate_classification_metrics", variant is None, None
        for line in lines:
            line = line.strip()
            if variant and line.startswith(f"Calculating {variant} accuracy"):
                found = True
            if found and line.startswith(tag):
                try:
                    metrics = json.loads(line[len(tag):])
                except json.JSONDecodeError:
                    pass
        return metrics

    def save(self, name: str, data) -> None:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    # ---- commands --------------------------------------------------------
    def info(self) -> dict:
        info = self.ok(self.projects.get_project_info(self.pid), "project info").to_dict()
        print(f"Project {self.pid}: {info['project']['name']}")
        print("Latency devices (use with `target --device`):")
        for dev in info.get("latencyDevices", []):
            print(f"  {'*' if dev.get('selected') else ' '} {dev['mcu']:32s} {dev['name']}")
        for cat, summary in (info.get("dataSummaryPerCategory") or {}).items():
            print(f"  data {cat}: {summary}")
        imp = info.get("impulse") or {}
        print(f"  impulse created: {imp.get('created')}  configured: {imp.get('configured')}  complete: {imp.get('complete')}")
        return info

    def set_target(self, needle: str) -> None:
        info = self.ok(self.projects.get_project_info(self.pid), "project info").to_dict()
        devices = info.get("latencyDevices", [])
        match = [d for d in devices if needle.lower() in (d["name"] + " " + d["mcu"]).lower()]
        if not match:
            sys.exit(f"No latency device matches '{needle}'. Run `info` to list them.")
        dev = match[0]
        self.ok(self.projects.update_project(self.pid, ei.UpdateProjectRequest.from_dict({"latencyDevice": dev["mcu"]})),
                "set target device")
        print(f"Target device set to {dev['name']} ({dev['mcu']})")

    def latest_versions(self) -> dict[str, int]:
        blocks = self.ok(self.impulse.get_impulse_blocks(self.pid), "impulse blocks").to_dict()
        versions = {}
        for block in blocks.get("dspBlocks", []):
            version = block.get("latestImplementationVersion")
            if version:
                versions[block["type"]] = version
        return versions

    def create_main(self, window_ms: int, stride_ms: int, replace: bool = True, anomaly: str = "gmm") -> None:
        versions = self.latest_versions()
        dsp = lambda bid, typ, name: {k: v for k, v in {  # noqa: E731
            "id": bid, "type": typ, "name": name, "title": name, "axes": ["audio"], "input": MAIN["input"],
            "implementationVersion": versions.get(typ)}.items() if v is not None}
        impulse = {
            "name": "Pump sound - MFE CNN + GMM",
            "inputBlocks": [{"id": MAIN["input"], "type": "time-series", "name": "Time series", "title": "Audio 16 kHz",
                             "windowSizeMs": window_ms, "windowIncreaseMs": stride_ms, "frequencyHz": 16000,
                             "padZeros": True}],
            # One processing block feeds both learning blocks. Two DSP blocks feeding different learn
            # blocks overflow the SDK's feature buffer at run time ("Would write outside feature
            # buffer"); concatenating them into the classifier fixes that but breaks int8 (one input
            # scale for 0-0.8 MFE values and ~400-valued spectral statistics). The GMM therefore uses
            # a slice of the MFE features (see --gmm-frames).
            "dspBlocks": [dsp(MAIN["mfe"], "mfe", "MFE")],
            "learnBlocks": [
                {"id": MAIN["classifier"], "type": "keras", "name": "Classifier", "title": "Classification",
                 "dsp": [MAIN["mfe"]]},
                # GMM by default; --anomaly kmeans switches to the K-means anomaly block.
                {"id": MAIN["anomaly"], "type": "anomaly" if anomaly == "kmeans" else "anomaly-gmm",
                 "name": "Anomaly detection",
                 "title": "Anomaly detection (K-means)" if anomaly == "kmeans" else "Anomaly detection (GMM)",
                 "dsp": [MAIN["mfe"]]},
            ][: 1 if anomaly == "none" else 2],
        }
        if replace:
            try:
                self.impulse.delete_impulse(self.pid)
            except ei.ApiException:
                pass  # no impulse yet
        # Posting without deleting first updates the impulse in place (like Studio's "Save impulse"),
        # keeping DSP configurations and generated features of unchanged blocks.
        self.http.post("impulse", impulse)
        print(f"Main impulse {'created' if replace else 'updated'}: window {window_ms} ms, stride {stride_ms} ms, "
              f"DSP versions {versions}")

    def dsp_params(self, dsp_id: int) -> dict:
        resp = self.ok(self.dsp.get_dsp_config(self.pid, dsp_id), "get DSP config")
        params = {}
        for group in resp.config or []:
            for item in group.items:
                params[item.param] = item.value
        return params

    def configure_dsp(self, dsp_id: int, wanted: dict) -> dict:
        current = self.dsp_params(dsp_id)
        unknown = sorted(set(wanted) - set(current))
        if unknown:
            print(f"  DSP {dsp_id}: skipping unknown params {unknown}; available: {sorted(current)}")
        config = {k: v for k, v in wanted.items() if k in current}
        self.ok(self.dsp.set_dsp_config(self.pid, dsp_id, ei.DSPConfigRequest.from_dict({"config": config})),
                f"set DSP {dsp_id}")
        final = self.dsp_params(dsp_id)
        print(f"  DSP {dsp_id} configured: " + ", ".join(f"{k}={final[k]}" for k in config))
        return final

    def autotune(self, dsp_id: int) -> None:
        job = self.ok(self.jobs.autotune_dsp_job(self.pid, ei.AutotuneDspRequest.from_dict({"dspId": dsp_id})), "autotune")
        self.wait(job.id, f"autotune DSP {dsp_id}")
        res = self.dsp.get_autotuner_results(self.pid, dsp_id).to_dict()
        self.save(f"autotune_dsp{dsp_id}", res)
        best = (res.get("results") or [{}])[0]
        print(f"  autotuner best: {best}")

    def generate(self, dsp_id: int, importance: bool = False) -> None:
        req = ei.GenerateFeaturesRequest.from_dict(
            {"dspId": dsp_id, "calculateFeatureImportance": importance, "skipFeatureExplorer": False})
        job = self.ok(self.jobs.generate_features_job(self.pid, req), "generate features")
        self.wait(job.id, f"features DSP {dsp_id}")

    def train_classifier(self, learn_id: int, num_filters: int, epochs: int, lr: float, tag: str,
                         augment: bool = False, batch_size: int = 32, time_budget_s: int = 0) -> dict:
        mfe_features = int(self.http.get(f"dsp/{MAIN['mfe']}/metadata").get("featureCount") or 2440)
        cfg = keras_config(num_filters, epochs, lr, augment, mfe_features=mfe_features,
                           batch_size=batch_size, time_budget_s=time_budget_s)
        job_id = self.http.post(f"jobs/train/keras/{learn_id}", cfg)["id"]
        lines = self.wait(job_id, f"train {tag}", timeout=3600)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"train_{tag}.log").write_text("\n".join(lines), encoding="utf-8")
        result = {"job_id": job_id, "float32": self.metrics_from_log(lines, "float32"),
                  "int8": self.metrics_from_log(lines, "int8")}
        try:  # plain JSON: the generated model fails to parse threshold values
            result["metadata"] = self.http.get(f"training/keras/{learn_id}/metadata")
        except Exception as exc:  # metadata is optional
            result["metadata_error"] = str(exc)
        meta = result.get("metadata") or {}
        labels = meta.get("labels") or meta.get("classNames")
        for m in meta.get("modelValidationMetrics") or []:   # one entry per model variant (float32 / int8)
            result[m.get("type")] = {"accuracy": m.get("accuracy"), "loss": m.get("loss"),
                                     "confusion_matrix": m.get("confusionMatrix"), "class_names": labels,
                                     "report": m.get("report"), "on_device": m.get("onDevicePerformance")}
        self.save(f"train_{tag}", result)
        for variant in ("float32", "int8"):
            m = result.get(variant) or {}
            if m:
                print(f"  {tag} {variant}: validation accuracy {m.get('accuracy')}, loss {m.get('loss')}")
        return result

    def train_anomaly(self, learn_id: int, clusters: int, frames: list[int] | None = None, bands: int = 40) -> dict:
        """Train the GMM. With `frames`, it uses all mel bands of those MFE frames (features are stored
        frame-major: index = frame * bands + band), which keeps the model small enough for an MCU."""
        cfg = self.http.get(f"training/anomaly/{learn_id}")
        n_axes = len(cfg.get("axes") or [])
        if not n_axes:
            raise RuntimeError("anomaly block reports no input axes; generate features first")
        if frames:
            axes = [f * bands + b for f in frames for b in range(bands) if f * bands + b < n_axes]
        else:
            axes = list(range(n_axes))
        job_id = self.http.post(f"jobs/train/anomaly/{learn_id}", {"axes": axes, "clusterCount": clusters})["id"]
        self.wait(job_id, "train GMM")
        try:
            meta = self.http.get(f"training/anomaly/{learn_id}/gmm/metadata")
        except Exception as exc:  # metadata is optional
            meta = {"error": str(exc)}
        self.save("train_anomaly", {"axes": len(axes), "frames": frames, "clusters": clusters, "metadata": meta})
        print(f"  GMM trained on {len(axes)} of {n_axes} features (frames {frames or 'all'}), {clusters} components")
        return meta

    def test(self, impulse_id: int | None = None, tag: str = "main") -> dict:
        req = ei.StartClassifyJobRequest.from_dict({"modelVariants": ["int8"]})
        kwargs = {"impulse_id": impulse_id} if impulse_id is not None else {}
        job = self.ok(self.jobs.start_classify_job(self.pid, req, **kwargs), "model testing")
        self.wait(job.id, f"model testing {tag}")
        return self.test_result(impulse_id, tag)

    def test_result(self, impulse_id: int | None = None, tag: str = "main", variant: str = "int8") -> dict:
        """Read the latest model-testing result as plain JSON (the generated model fails on threshold values)."""
        params = {"variant": variant, **({"impulseId": impulse_id} if impulse_id is not None else {})}
        res = self.http.get("classify/all/result", **params)
        self.save(f"test_{tag}_full", res)
        acc = res.get("accuracy") or {}
        summary = {"variant": variant, "accuracy": acc}
        self.save(f"test_{tag}", summary)
        print(f"  test {tag} ({variant}): accuracy {acc.get('accuracyScore')}  summary {acc.get('totalSummary')}")
        return summary

    def performance(self) -> dict:
        """On-device estimates for the project's target device, per learning block and model variant.

        Read from the detailed impulse (the same numbers Studio shows on each learning block). The
        MFE/Spectral DSP estimates are not exposed by the API; Studio shows them on the DSP pages."""
        detailed = self.http.get("impulses-detailed").get("impulses", [{}])[0]
        summary: dict = {}
        blocks = [("classifier", detailed.get("learnBlockKerasConfigs") or []),
                  ("anomaly", detailed.get("learnBlockAnomalyConfigs") or [])]
        for name, configs in blocks:
            for cfg in configs:
                for m in (cfg.get("metadata") or {}).get("modelValidationMetrics") or []:
                    for dev in m.get("onDevicePerformance") or []:
                        if not dev.get("isDefault"):
                            continue
                        mem = dev.get("eon") or dev.get("tflite") or {}
                        key = f"{name}_{m.get('type', 'model')}"
                        summary[key] = {"device": dev.get("name"), "latency_ms": dev.get("latency"),
                                        "ram_kb": round((mem.get("ramRequired") or 0) / 1024, 1),
                                        "flash_kb": round((mem.get("romRequired") or 0) / 1024, 1)}
        cls = summary.get("classifier_int8") or {}
        gmm = next((v for k, v in summary.items() if k.startswith("anomaly")), {})
        summary.update({
            "device": cls.get("device"),
            "nn_ms": cls.get("latency_ms"), "gmm_ms": gmm.get("latency_ms"),
            "learn_ms": (cls.get("latency_ms") or 0) + (gmm.get("latency_ms") or 0),
            "ram_kb": max(cls.get("ram_kb") or 0, gmm.get("ram_kb") or 0) or None,
            "flash_kb": round((cls.get("flash_kb") or 0) + (gmm.get("flash_kb") or 0), 1) or None,
        })
        self.save("performance", summary)
        print("  on-device estimate:", json.dumps(summary, indent=1))
        return summary

    def performance_legacy(self) -> dict:
        """Earlier attempt (kept for reference): per-DSP-block endpoint returns no numbers on this plan."""
        raw = {}
        for name, path in (("mfe", f"dsp/{MAIN['mfe']}/performance"), ("spectral", f"dsp/{MAIN['spectral']}/performance"),
                           ("classifier", f"training/keras/{MAIN['classifier']}/metadata")):
            try:
                raw[name] = self.http.get(path)
            except Exception as exc:  # keep going; the report shows "—" for missing numbers
                raw[name] = {"error": str(exc)}
        self.save("performance_raw", raw)
        summary = {}
        for m in (raw.get("classifier") or {}).get("modelValidationMetrics") or []:
            if m.get("type") != "int8":
                continue
            for dev in m.get("onDevicePerformance") or []:
                if dev.get("isDefault") or "nano-33" in str(dev.get("mcu", "")):
                    eon = dev.get("eon") or dev.get("tflite") or {}
                    summary.update({"device": dev.get("name"), "nn_ms": dev.get("latency"),
                                    "nn_ram_kb": (eon.get("ramRequired") or 0) / 1024,
                                    "nn_flash_kb": (eon.get("romRequired") or 0) / 1024})
        for name in ("mfe", "spectral"):
            perf = (raw.get(name) or {}).get("performance") or {}
            summary[f"{name}_ms"] = perf.get("timePerInferenceMs")
            summary[f"{name}_ram_kb"] = (perf.get("ramRequired") or 0) / 1024 if perf.get("ramRequired") else None
        dsp_ms = (summary.get("mfe_ms") or 0) + (summary.get("spectral_ms") or 0)
        summary["dsp_ms"] = dsp_ms or None
        if summary.get("nn_ms") is not None:
            summary["total_latency_ms"] = dsp_ms + summary["nn_ms"]
        rams = [v for k, v in summary.items() if k.endswith("_ram_kb") and v]
        summary["ram_kb"] = max(rams) if rams else None
        summary["flash_kb"] = summary.get("nn_flash_kb")
        self.save("performance", summary)
        print("  on-device estimate:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in summary.items()})
        return summary

    def tuner(self) -> None:
        """Start the EON Tuner with the project's default configuration (target = project latency device)."""
        try:
            cfg = self.http.get("optimize/config")
            self.save("eon_tuner_config", cfg)
            print("  tuner target:", (cfg.get("targetDevice") or {}), "latency budget:", cfg.get("targetLatency"))
        except Exception as exc:
            print(f"  could not read tuner config ({exc})")
        job = self.http.post("jobs/optimize", {})
        print(f"  EON Tuner started: job {job.get('id')} — results appear in Studio > EON Tuner")

    def reference(self) -> None:
        """Extra experiment: the public reference project's settings (2000/1000 ms, MFE, 1D CNN)."""
        resp = self.ok(self.impulse.create_new_empty_impulse(self.pid), "new impulse")
        impulse_id = resp.id
        versions = self.latest_versions()
        impulse = {
            "name": "Reference settings - 2 s window MFE CNN",
            "inputBlocks": [{"id": REF["input"], "type": "time-series", "name": "Time series", "title": "Audio 16 kHz",
                             "windowSizeMs": 2000, "windowIncreaseMs": 1000, "frequencyHz": 16000, "padZeros": True}],
            "dspBlocks": [{k: v for k, v in {"id": REF["mfe"], "type": "mfe", "name": "MFE", "title": "MFE",
                                              "axes": ["audio"], "input": REF["input"],
                                              "implementationVersion": versions.get("mfe")}.items() if v is not None}],
            "learnBlocks": [{"id": REF["classifier"], "type": "keras", "name": "Classifier", "title": "Classification",
                             "dsp": [REF["mfe"]]}],
        }
        self.ok(self.impulse.create_impulse(self.pid, ei.CreateImpulseRequest.from_dict(impulse), impulse_id=impulse_id),
                "create reference impulse")
        print(f"Reference impulse {impulse_id} created")
        self.configure_dsp(REF["mfe"], MFE_CONFIG)
        self.generate(REF["mfe"])
        self.train_classifier(REF["classifier"], MFE_CONFIG["num_filters"], 100, 0.005, "reference")
        self.test(impulse_id, "reference")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["info", "target", "create", "configure", "features", "train", "test", "all",
                                        "reference", "perf", "tuner", "testresult", "update-impulse", "train-gmm"])
    ap.add_argument("--device", default="nano 33", help="text to match a latency device (target command)")
    ap.add_argument("--window-ms", type=int, default=1000)
    ap.add_argument("--stride-ms", type=int, default=500)
    ap.add_argument("--low-hz", type=int, help="override MFE low frequency (e.g. 300 for speaker playback)")
    ap.add_argument("--autotune", action="store_true", help="run the MFE autotuner before generating features")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.001,
                    help="0.005 (Studio default) diverged after ~75 epochs on this data; 0.001 trains stably")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="per-step cost is mostly overhead, so e.g. 256 (with --lr 0.003) trains ~8x faster")
    ap.add_argument("--time-budget", type=int, default=0,
                    help="stop fitting after this many seconds and keep the best epoch (0 = no limit)")
    ap.add_argument("--clusters", type=int, default=5, help="GMM components (use ~32 clusters for --anomaly kmeans)")
    ap.add_argument("--anomaly", choices=["gmm", "kmeans", "none"], default="gmm",
                    help="anomaly learning block type (none: classifier only; EI's GMM/K-means profiling "
                         "crashes on a slice of MFE features)")
    ap.add_argument("--gmm-frames", default="15,45",
                    help="MFE frames (0-60) whose 40 mel bands feed the GMM; 2 frames = 80 features")
    ap.add_argument("--augment", action="store_true", help="enable spectrogram augmentation (noise + masking)")
    args = ap.parse_args()

    s = Studio()
    mfe = dict(MFE_CONFIG)
    if args.low_hz is not None:
        mfe["low_frequency"] = args.low_hz

    if args.command == "info":
        s.info()
    elif args.command == "target":
        s.set_target(args.device)
    if args.command in ("create", "all"):
        s.create_main(args.window_ms, args.stride_ms, anomaly=args.anomaly)
    if args.command == "update-impulse":
        s.create_main(args.window_ms, args.stride_ms, replace=False, anomaly=args.anomaly)
    if args.command in ("configure", "all"):
        s.configure_dsp(MAIN["mfe"], mfe)
        if args.autotune:
            s.autotune(MAIN["mfe"])
    if args.command in ("features", "all"):
        s.generate(MAIN["mfe"])
    gmm_frames = [int(f) for f in args.gmm_frames.split(",") if f.strip()]
    if args.command in ("train", "all"):
        s.train_classifier(MAIN["classifier"], int(s.dsp_params(MAIN["mfe"]).get("num_filters", 40)),
                           args.epochs, args.lr, "main", augment=args.augment,
                           batch_size=args.batch_size, time_budget_s=args.time_budget)
        if args.anomaly != "none":
            s.train_anomaly(MAIN["anomaly"], args.clusters, gmm_frames)
    if args.command == "train-gmm":
        s.train_anomaly(MAIN["anomaly"], args.clusters, gmm_frames)
    if args.command in ("test", "all"):
        s.test()
    if args.command == "testresult":
        s.test_result()
    if args.command in ("perf", "all"):
        s.performance()
    if args.command == "tuner":
        s.tuner()
    if args.command == "reference":
        s.reference()


if __name__ == "__main__":
    main()
