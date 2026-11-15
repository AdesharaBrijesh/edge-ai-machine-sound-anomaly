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


def keras_config(num_filters: int, epochs: int, lr: float) -> dict:
    return {
        "mode": "visual",
        "trainingCycles": epochs,
        "learningRate": lr,
        "batchSize": 32,
        "trainTestSplit": 0.8,
        "autoClassWeights": True,
        "profileInt8": True,
        "skipEmbeddingsAndMemory": False,
        "augmentationPolicySpectrogram": {
            "enabled": True, "warping": False,
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

    def create_main(self, window_ms: int, stride_ms: int) -> None:
        versions = self.latest_versions()
        dsp = lambda bid, typ, name: {k: v for k, v in {  # noqa: E731
            "id": bid, "type": typ, "name": name, "title": name, "axes": ["audio"], "input": MAIN["input"],
            "implementationVersion": versions.get(typ)}.items() if v is not None}
        impulse = {
            "name": "Pump sound - MFE CNN + GMM",
            "inputBlocks": [{"id": MAIN["input"], "type": "time-series", "name": "Time series", "title": "Audio 16 kHz",
                             "windowSizeMs": window_ms, "windowIncreaseMs": stride_ms, "frequencyHz": 16000,
                             "padZeros": True}],
            "dspBlocks": [dsp(MAIN["mfe"], "mfe", "MFE"), dsp(MAIN["spectral"], "spectral-analysis", "Spectral features")],
            "learnBlocks": [
                {"id": MAIN["classifier"], "type": "keras", "name": "Classifier", "title": "Classification",
                 "dsp": [MAIN["mfe"]]},
                {"id": MAIN["anomaly"], "type": "anomaly-gmm", "name": "Anomaly detection", "title": "Anomaly detection (GMM)",
                 "dsp": [MAIN["spectral"]]},
            ],
        }
        try:
            self.impulse.delete_impulse(self.pid)
        except ei.ApiException:
            pass  # no impulse yet
        self.ok(self.impulse.create_impulse(self.pid, ei.CreateImpulseRequest.from_dict(impulse)), "create impulse")
        print(f"Main impulse created: window {window_ms} ms, stride {stride_ms} ms, DSP versions {versions}")

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

    def train_classifier(self, learn_id: int, num_filters: int, epochs: int, lr: float, tag: str) -> dict:
        job_id = self.http.post(f"jobs/train/keras/{learn_id}", keras_config(num_filters, epochs, lr))["id"]
        lines = self.wait(job_id, f"train {tag}", timeout=3600)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"train_{tag}.log").write_text("\n".join(lines), encoding="utf-8")
        result = {"job_id": job_id, "float32": self.metrics_from_log(lines, "float32"),
                  "int8": self.metrics_from_log(lines, "int8")}
        try:
            result["metadata"] = self.learn.get_keras_metadata(self.pid, learn_id).to_dict()
        except ei.ApiException as exc:
            result["metadata_error"] = str(exc)
        self.save(f"train_{tag}", result)
        for variant in ("float32", "int8"):
            m = result.get(variant) or {}
            if m:
                print(f"  {tag} {variant}: validation accuracy {m.get('accuracy')}, loss {m.get('loss')}")
        return result

    def train_anomaly(self, learn_id: int, clusters: int) -> dict:
        cfg = self.learn.get_anomaly(self.pid, learn_id).to_dict()
        axes = list(range(len(cfg.get("axes") or [])))
        if not axes:
            raise RuntimeError("anomaly block reports no input axes; generate spectral features first")
        job_id = self.http.post(f"jobs/train/anomaly/{learn_id}", {"axes": axes, "clusterCount": clusters})["id"]
        self.wait(job_id, "train GMM")
        meta = {}
        try:
            meta = self.learn.get_gmm_metadata(self.pid, learn_id).to_dict()
        except ei.ApiException as exc:
            meta = {"error": str(exc)}
        self.save("train_anomaly", {"axes": len(axes), "clusters": clusters, "metadata": meta})
        print(f"  GMM trained on {len(axes)} spectral features, {clusters} components")
        return meta

    def test(self, impulse_id: int | None = None, tag: str = "main") -> dict:
        req = ei.StartClassifyJobRequest.from_dict({"modelVariants": ["int8"]})
        kwargs = {"impulse_id": impulse_id} if impulse_id is not None else {}
        job = self.ok(self.jobs.start_classify_job(self.pid, req, **kwargs), "model testing")
        lines = self.wait(job.id, f"model testing {tag}")
        summary = {"log_metrics": self.metrics_from_log(lines)}
        try:
            res = self.classify.get_classify_job_result(self.pid, variant="int8", **kwargs).to_dict()
            summary["accuracy"] = res.get("accuracy")
            self.save(f"test_{tag}_full", res)
        except ei.ApiException as exc:
            summary["result_error"] = str(exc)
        self.save(f"test_{tag}", summary)
        acc = summary.get("accuracy") or {}
        print(f"  test {tag}: accuracy {acc.get('accuracyScore')}  summary {acc.get('totalSummary')}")
        return summary

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
    ap.add_argument("command", choices=["info", "target", "create", "configure", "features", "train", "test", "all", "reference"])
    ap.add_argument("--device", default="nano 33", help="text to match a latency device (target command)")
    ap.add_argument("--window-ms", type=int, default=1000)
    ap.add_argument("--stride-ms", type=int, default=500)
    ap.add_argument("--low-hz", type=int, help="override MFE low frequency (e.g. 300 for speaker playback)")
    ap.add_argument("--autotune", action="store_true", help="run the MFE autotuner before generating features")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--clusters", type=int, default=5)
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
        s.create_main(args.window_ms, args.stride_ms)
    if args.command in ("configure", "all"):
        s.configure_dsp(MAIN["mfe"], mfe)
        s.configure_dsp(MAIN["spectral"], SPECTRAL_CONFIG)
        if args.autotune:
            s.autotune(MAIN["mfe"])
    if args.command in ("features", "all"):
        s.generate(MAIN["mfe"])
        s.generate(MAIN["spectral"], importance=True)
    if args.command in ("train", "all"):
        s.train_classifier(MAIN["classifier"], int(s.dsp_params(MAIN["mfe"]).get("num_filters", 40)),
                           args.epochs, args.lr, "main")
        s.train_anomaly(MAIN["anomaly"], args.clusters)
    if args.command in ("test", "all"):
        s.test()
    if args.command == "reference":
        s.reference()


if __name__ == "__main__":
    main()
