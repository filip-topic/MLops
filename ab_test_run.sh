for d in ab_test/tasks/*; do docker build -t "ab-$(basename $d):latest" "$d"; done

#--no-cache

#docker build -t  ab-load_config:latest ./ab_test/tasks/load_config --no-cache

echo "✔ Step images built."

echo "Starting the A/B TEST FLOW"

python ab_test/ab_test_flow.py