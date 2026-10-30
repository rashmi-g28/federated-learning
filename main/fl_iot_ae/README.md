# fl_iot_ae - Federated 3C autoencoder (115-32-6-32-115) on N-BaIoT, with Flower (local)

One Flower client = one IoT device (9 normal CSVs -> 9 non-IID clients). FedAvg. No raw data leaves a client.

## Run order (PowerShell, from this folder)

    python -m venv .venv ; .\.venv\Scripts\Activate.ps1
    pip install -e .
    python prepare_partitions.py --data-root "C:\Users\Rashmi Gaonkar\Desktop\major-1\comparision" --out ..\fl_data
    flwr run . --federation-config "num-supernodes=9" --stream
    python baselines.py --mode centralized
    python baselines.py --mode local
    python report.py

Outputs land in `results/`: `fl_run.json`, `fl_global_3C.pt`, `baseline_*.json`, `fl_comparison.csv/.txt`.

## Tuning knobs (pyproject.toml -> [tool.flwr.app.config], or override on the command line)

    flwr run . --federation-config "num-supernodes=9" --run-config "num-server-rounds=60 local-epochs=5 learning-rate=0.0005" --stream

## Notes
- If the number of supernodes differs from the number of partitions the server stops with a message telling you the exact value.
- Ray simulation is experimental on native Windows. If it misbehaves, run the same commands inside WSL2 (Ubuntu).
- "FL pooled x%" = threshold from summed client error histograms (primary). "FL max x%" = strictest single-client threshold.
- Gate to check on real data: worst-attack-file recall (Gafgyt tcp/udp), not just average recall.
