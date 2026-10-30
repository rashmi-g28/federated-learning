"""
fl_iot_ae.server_app -- the federated protocol (all 9 devices, FedAvg, no raw data shared).

  Phase 0  federated scaler   clients send per-feature min/max of their own data -> server builds global min/max
  Phase 1  FedAvg training    num-server-rounds rounds; every round all clients train locally, server averages
                              (weighted by number of local records). Every `eval-every` rounds the global model
                              is scored on the server's labelled hold-out (AUC only, threshold-free).
  Phase 2  threshold          clients send a HISTOGRAM (counts only) of reconstruction error on their OWN normal
                              threshold-set; server sums the counts -> exact pooled percentile per FPR budget
                              ('FL pooled', primary) and the stricter 'FL max' (worst single client)
  Phase 3  final evaluation   recall / precision / FPR / F1 / worst-attack-file recall at each threshold,
                              + exact communication and compute accounting -> results/fl_run.json
"""
import json
import time

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from . import task as T

app = ServerApp()


def _wait_for_nodes(grid: Grid, n_expected: int, timeout: float = 180.0):
    t0 = time.time()
    while True:
        ids = sorted(grid.get_node_ids())
        if len(ids) >= n_expected or time.time() - t0 > timeout:
            return ids
        time.sleep(1.0)


def _bytes(rd: RecordDict) -> int:
    """Exact serialized size of the records inside a RecordDict (sum of its records)."""
    return int(sum(rec.count_bytes() for rec in rd.values()))


def _query(grid: Grid, node_ids, make_content, group_id: str, timeout: float = 900.0):
    """Send one 'query' message to every node; return (replies, bytes_down, bytes_up)."""
    msgs = [Message(content=make_content(), dst_node_id=n, message_type="query", group_id=group_id)
            for n in node_ids]
    down = sum(_bytes(m.content) for m in msgs)
    replies = list(grid.send_and_receive(msgs, timeout=timeout))
    bad = [r for r in replies if r.has_error()]
    if bad or len(replies) != len(msgs):
        raise RuntimeError(f"query round failed: {len(replies)}/{len(msgs)} replies, errors: "
                           f"{[r.error.reason if r.has_error() else None for r in bad][:3]}")
    up = sum(_bytes(r.content) for r in replies)
    return replies, down, up


def _fmt(x):
    return f"{x:.4f}"


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    ddir = T.data_dir(cfg)
    rdir = T.results_dir(cfg)
    meta = T.load_meta(ddir)
    n_clients, n_feat = int(meta["n_clients"]), int(meta["n_features"])
    rounds, lr = int(cfg["num-server-rounds"]), float(cfg["learning-rate"])
    every = int(cfg["eval-every"])
    device = T.pick_device()

    node_ids = _wait_for_nodes(grid, n_clients)
    if len(node_ids) != n_clients:
        raise RuntimeError(
            f"Found {len(node_ids)} connected nodes but the data has {n_clients} client partitions.\n"
            f"Simulation: add  --federation-config \"num-supernodes={n_clients}\"  to the flwr run command.\n"
            f"Deployment: start exactly {n_clients} SuperNodes with partition-id 0..{n_clients - 1}.")
    print(f"[server] {n_clients} clients connected, {n_feat} features, {rounds} rounds, "
          f"{cfg['local-epochs']} local epochs, batch {cfg['batch-size']}, lr {lr}", flush=True)

    comm = {"scaler_round": 0, "calibration_round": 0}

    # ---------------------------------------------------------------- Phase 0: federated scaler
    replies, d, u = _query(grid, node_ids, lambda: RecordDict({"config": ConfigRecord({"what": "stats"})}), "stats")
    comm["scaler_round"] = d + u
    mins = np.min([r.content["arrays"].to_numpy_ndarrays()[0] for r in replies], axis=0)
    maxs = np.max([r.content["arrays"].to_numpy_ndarrays()[1] for r in replies], axis=0)
    smin, smax = mins.astype(np.float64), maxs.astype(np.float64)
    print("[server] global min/max built from per-client statistics (no raw data shared)", flush=True)

    # ---------------------------------------------------------------- Phase 1: FedAvg
    test = T.load_server_test(ddir)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)                      # seeds the global model's starting weights
    global_model = T.build_model(n_feat)
    arrays0 = ArrayRecord(global_model.state_dict())
    train_config = ConfigRecord({"lr": lr, "smin": smin.tolist(), "smax": smax.tolist()})

    def central_eval(server_round: int, arrays: ArrayRecord):
        if server_round != 0 and server_round % every != 0 and server_round != rounds:
            return None
        m = T.build_model(n_feat)
        m.load_state_dict(arrays.to_torch_state_dict())
        r = T.evaluate_global(m, smin, smax, test, None, device)
        print(f"[server] round {server_round:3d}: AUC {_fmt(r['auc_all'])} | worst attack-file AUC "
              f"{_fmt(r['auc_worst_file'])} | normal median RE {r['normal_median_re']:.5f}", flush=True)
        return MetricRecord({"auc_all": r["auc_all"], "auc_worst_file": r["auc_worst_file"],
                             "normal_median_re": r["normal_median_re"]})

    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=0.0, min_train_nodes=n_clients,
                      min_available_nodes=n_clients)
    t0 = time.time()
    result = strategy.start(grid=grid, initial_arrays=arrays0, train_config=train_config,
                            num_rounds=rounds, evaluate_fn=central_eval)
    fl_seconds = time.time() - t0
    final_arrays = result.arrays

    # ---------------------------------------------------------------- Phase 2: federated threshold calibration
    budgets = list(T.BUDGETS)
    lo, hi, n_bins = -9.0, 3.0, 1200            # shared log-spaced error grid, 1e-9 .. 1e3
    edges = np.logspace(lo, hi, n_bins + 1)
    replies, d, u = _query(
        grid, node_ids,
        lambda: RecordDict({"arrays": ArrayRecord(final_arrays.to_torch_state_dict()),
                            "config": ConfigRecord({"what": "calibrate", "smin": smin.tolist(),
                                                    "smax": smax.tolist(), "budgets": budgets,
                                                    "lo": lo, "hi": hi, "n_bins": n_bins})}),
        "calibrate")
    comm["calibration_round"] = d + u
    counts = np.sum([r.content["arrays"].to_numpy_ndarrays()[0] for r in replies], axis=0).astype(np.float64)
    cdf = np.cumsum(counts) / counts.sum()
    q_i = np.array([r.content["metrics"]["q"] for r in replies], dtype=np.float64)   # clients x budgets
    thr_max = q_i.max(axis=0)                     # conservative: the worst single client's own percentile
    thresholds = {}
    for j, b in enumerate(budgets):
        k = int(np.searchsorted(cdf, 1.0 - b))     # first bin whose cumulative share reaches 1 - b
        k = min(k, n_bins - 1)
        prev = cdf[k - 1] if k > 0 else 0.0
        frac = (1.0 - b - prev) / max(cdf[k] - prev, 1e-12)
        thresholds[f"FL pooled {b * 100:g}%"] = float(edges[k] * (edges[k + 1] / edges[k]) ** min(max(frac, 0.0), 1.0))
        thresholds[f"FL max {b * 100:g}%"] = float(thr_max[j])

    # window rule: strict cutoff = the highest value ANY client's normal windows showed
    win_taus = {W: (max(float(r.content["metrics"][f"w{W}_rep_max"]) for r in replies),
                    max(float(r.content["metrics"][f"w{W}_re_max"]) for r in replies)) for W in T.WINDOWS}

    # ---------------------------------------------------------------- Phase 3: final evaluation + accounting
    final_model = T.build_model(n_feat)
    final_model.load_state_dict(final_arrays.to_torch_state_dict())
    ev = T.evaluate_global(final_model, smin, smax, test, thresholds, device)
    win = T.window_evaluate(final_model, smin, smax, test, [c["n_test"] for c in meta["clients"]], win_taus, device)

    arrays_bytes = final_arrays.count_bytes()
    down_rec = RecordDict({"arrays": final_arrays, "config": train_config})
    up_rec = RecordDict({"arrays": final_arrays,
                         "metrics": MetricRecord({"train_loss": 0.0, "train_seconds": 0.0, "num-examples": 0})})
    per_client_round_down, per_client_round_up = _bytes(down_rec), _bytes(up_rec)
    train_bytes = rounds * n_clients * (per_client_round_down + per_client_round_up)
    total_bytes = train_bytes + comm["scaler_round"] + comm["calibration_round"]

    tm = {int(k): dict(v) for k, v in getattr(result, "train_metrics_clientapp", {}).items()}
    secs = [v.get("train_seconds") for v in tm.values() if v.get("train_seconds") is not None]
    losses = {k: v.get("train_loss") for k, v in tm.items()}
    curve = {int(k): dict(v) for k, v in getattr(result, "evaluate_metrics_serverapp", {}).items()}

    summary = dict(
        model="3C 115-32-6-32-115", n_clients=n_clients, n_features=n_feat,
        config=dict(rounds=rounds, local_epochs=int(cfg["local-epochs"]), batch_size=int(cfg["batch-size"]),
                    lr=lr, strategy="FedAvg", seed=seed),
        params=T.count_params(final_model), macs_per_record=T.count_macs(final_model),
        communication=dict(
            model_payload_bytes=arrays_bytes,
            per_client_per_round_down_bytes=per_client_round_down,
            per_client_per_round_up_bytes=per_client_round_up,
            training_total_bytes=train_bytes, scaler_round_bytes=comm["scaler_round"],
            calibration_round_bytes=comm["calibration_round"], total_bytes=total_bytes,
            note="exact serialized record sizes; excludes gRPC/protobuf framing overhead"),
        compute=dict(fedavg_wall_clock_seconds=fl_seconds,
                     mean_client_train_seconds_per_round=float(np.mean(secs)) if secs else None,
                     max_client_train_seconds_per_round=float(np.max(secs)) if secs else None,
                     training_flops_per_record_approx=6 * T.count_macs(final_model),
                     inference=T.bench_inference(final_model, T.scale(test["normal"][:20000], smin, smax))),
        thresholds=thresholds, evaluation=ev, window=win, train_loss_by_round=losses, central_eval_curve=curve,
    )
    ts = time.strftime("%Y%m%d_%H%M%S")
    (rdir / "fl_run.json").write_text(json.dumps(T.jsonable(summary), indent=2))
    (rdir / f"fl_run_{ts}.json").write_text(json.dumps(T.jsonable(summary), indent=2))
    (rdir / f"fl_run_seed{seed}.json").write_text(json.dumps(T.jsonable(summary), indent=2))
    torch.save(dict(state_dict=final_model.state_dict(), smin=smin, smax=smax, thresholds=thresholds,
                    columns=meta["columns"]), rdir / "fl_global_3C.pt")

    print("\n" + "=" * 86, flush=True)
    print(f"FEDERATED RESULT  (3C, {n_clients} clients, {rounds} rounds, FedAvg)   AUC all = {_fmt(ev['auc_all'])}   "
          f"worst attack-file AUC = {_fmt(ev['auc_worst_file'])}")
    print("=" * 86)
    print(f"{'threshold rule':<18}{'thr':>10}{'FPR':>9}{'recall':>9}{'precision':>11}{'F1':>9}{'worst-file recall':>20}")
    for name, op in ev["operating_points"].items():
        print(f"{name:<18}{op['threshold']:>10.5f}{op['fpr']:>9.4f}{op['recall']:>9.4f}{op['precision']:>11.4f}"
              f"{op['f1']:>9.4f}{op['worst_file_recall']:>20.4f}")
    print("\nWINDOW RULE (strict 'max' cutoffs from clients' normal windows; alert = repeat-rate OR median-RE above cutoff)")
    print(f"{'W':>5}{'window FPR':>12}{'recall(any)':>13}{'worst file':>12}{'  tcp/udp via repeat rule':>26}")
    for W, w in win.items():
        pf = w["recall_per_file"]
        tu = min(pf[k]["rep"] for k in pf if k in ("Gafgyt__tcp", "Gafgyt__udp")) if "Gafgyt__tcp" in pf else float("nan")
        print(f"{W:>5}{w['window_fpr']['any']:>12.4f}{w['window_recall']['any']:>13.4f}{w['worst_file_recall']['any']:>12.4f}{tu:>26.4f}")
    print(f"\nCommunication: model payload {arrays_bytes / 1024:.1f} KB; per client per round "
          f"{per_client_round_down / 1024:.1f} KB down + {per_client_round_up / 1024:.1f} KB up; "
          f"whole run {total_bytes / 1e6:.2f} MB for {n_clients} clients x {rounds} rounds")
    print(f"Saved {rdir / 'fl_run.json'} and {rdir / 'fl_global_3C.pt'}", flush=True)
