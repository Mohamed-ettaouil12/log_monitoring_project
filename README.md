# Systeme intelligent de surveillance des logs RCA

Ce projet applique le document `nlp.pdf` au code: collecte de logs, nettoyage, classification des causes racines, alertes automatiques, API REST et dashboard temps reel.

## Fonctionnalites principales

- Collecte de logs Docker en temps reel avec `docker logs -f`.
- Nettoyage des logs: timestamps, IPs et identifiants longs sont normalises.
- Classification RCA avec moteur hybride: DistilBERT entraine + regles metier robustes.
- Detection de 5 causes racines: `disk_full`, `memory_leak`, `network_failure`, `authentication_error`, `resource_exhaustion`.
- Alertes si la confiance depasse `70%`, avec severite, priorite, contact et actions recommandees.
- Dashboard Streamlit rafraichi toutes les secondes.
- API FastAPI pour integrations externes.
- Historique audit dans `data/alerts.jsonl`, `data/stream_predictions.jsonl` et `results/realtime_summary.json`.

## Lancement production local

```bash
./start_monitoring.sh docker
```

Cette commande demarre:

- API FastAPI: `http://localhost:8000`
- Dashboard Streamlit: `http://localhost:8501`
- Collecte Docker: tous les conteneurs en cours

Pour choisir certains conteneurs:

```bash
DOCKER_CONTAINERS="container_api,container_mlflow" ./start_monitoring.sh docker
```

## Lancement separe

API:

```bash
python3 -m uvicorn src.api_monitoring:app --host 0.0.0.0 --port 8000
```

Dashboard:

```bash
python3 -m streamlit run src/dashboard.py --server.port 8501
```

Collecteur Docker:

```bash
.venv/bin/python -m src.docker_log_stream --all-running --tail 100
```

Test fonctionnel:

```bash
.venv/bin/python test_monitoring.py
```

## API REST

Prediction simple:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"log_text":"CRITICAL Disk space low on /var: 99% used"}'
```

Endpoints utiles:

- `GET /health`
- `POST /predict`
- `POST /predict_batch`
- `GET /alerts`
- `GET /stats`
- `GET /causes`

## Fichiers de resultats

- `data/raw/docker_logs.csv`: logs Docker bruts.
- `data/stream_predictions.jsonl`: logs analyses avec prediction, confiance, severite et latence.
- `data/alerts.jsonl`: alertes declenchees avec actions recommandees.
- `data/performance.json`: compteurs globaux.
- `results/realtime_summary.json`: resume de la derniere session.

## Correspondance avec le document

- Collecter les logs: `src/docker_log_stream.py`
- Nettoyer les donnees: `src/preprocessing.py`
- Analyser avec IA: `src/runtime_loader.py`, `src/models.py`, `src/hybrid_model.py`
- Declencher alertes: `src/send_alert.py`, `src/rca_catalog.py`
- Enregistrer et apprendre: fichiers JSON/JSONL dans `data/`
- API REST: `src/api_monitoring.py`
- Dashboard temps reel: `src/dashboard.py`
