docker build -t monitoring-load-data:latest monitoring/tasks/load_data
docker build -t monitoring-kl-div:latest monitoring/tasks/compute_kl_divergence

python monitoring/monitoring_flow.py