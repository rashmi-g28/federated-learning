"""
fl_iot_ae.client_app -- one Flower client = one IoT device.

A client only reads its OWN partition (client_<partition-id>/train.npy and thr.npy). What leaves the client:
    query "stats"      : per-feature min / max of its training data (2 x N numbers) + its record count
    train              : updated model weights (31 KB for 3C) + training loss / seconds / record count
    query "calibrate"  : a histogram (counts only) of reconstruction errors on its OWN normal threshold-set,
                         plus 2 numbers per window size (its highest window repeat-rate and median-RE)
Raw traffic records never leave the client.
"""
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from . import task as T

app = ClientApp()


def _load_model(msg: Message, n_features: int, device):
    model = T.build_model(n_features)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model.to(device)


@app.train()
def train(msg: Message, context: Context):
    """Local training on this device's normal traffic, starting from the global model."""
    device = T.pick_device()
    pid = int(context.node_config["partition-id"])
    ddir = T.data_dir(context.run_config)
    meta = T.load_meta(ddir)
    cfg = msg.content["config"]
    model = _load_model(msg, meta["n_features"], device)

    X = T.scale(T.load_client(ddir, pid, "train"), cfg["smin"], cfg["smax"])
    rnd = int(msg.metadata.group_id) if str(msg.metadata.group_id).isdigit() else 0
    loss, secs = T.train_local(model, X,
                               epochs=int(context.run_config["local-epochs"]),
                               batch_size=int(context.run_config["batch-size"]),
                               lr=float(cfg["lr"]), device=device,
                               seed=int(context.run_config["seed"]) * 1_000_003 + 1000 * rnd + pid)

    reply = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": float(loss), "train_seconds": float(secs),
                                 "num-examples": int(len(X))}),
    })
    return Message(content=reply, reply_to=msg)


@app.query()
def query(msg: Message, context: Context):
    """Two small side-protocols: scaler statistics and threshold calibration."""
    pid = int(context.node_config["partition-id"])
    ddir = T.data_dir(context.run_config)
    cfg = msg.content["config"]
    what = cfg["what"]

    if what == "stats":
        X = T.load_client(ddir, pid, "train").astype(np.float64)
        reply = RecordDict({
            "arrays": ArrayRecord.from_numpy_ndarrays([X.min(axis=0), X.max(axis=0)]),
            "metrics": MetricRecord({"num-examples": int(len(X))}),
        })
        return Message(content=reply, reply_to=msg)

    if what == "calibrate":
        device = T.pick_device()
        meta = T.load_meta(ddir)
        model = _load_model(msg, meta["n_features"], device)
        Xthr = T.scale(T.load_client(ddir, pid, "thr"), cfg["smin"], cfg["smax"])
        re = T.reconstruction_error(model, Xthr, device)
        q = [float(np.quantile(re, 1.0 - b)) for b in cfg["budgets"]]
        edges = np.logspace(float(cfg["lo"]), float(cfg["hi"]), int(cfg["n_bins"]) + 1)
        counts = np.histogram(np.clip(re, edges[0], edges[-1]), bins=edges)[0].astype(np.int64)
        # window rule: highest repeat rate / highest median-RE this device's own normal windows ever showed
        ids = T.record_ids(Xthr)
        wmax = {}
        for W in T.WINDOWS:
            s_w, m_w = T.window_stats(ids, re, W, max(W // 2, 1))
            wmax[f"w{W}_rep_max"] = float(s_w.max())
            wmax[f"w{W}_re_max"] = float(m_w.max())
        reply = RecordDict({"arrays": ArrayRecord.from_numpy_ndarrays([counts]),
                            "metrics": MetricRecord({"num-examples": int(len(Xthr)), "q": q, **wmax})})
        return Message(content=reply, reply_to=msg)

    raise ValueError(f"unknown query: {what}")
