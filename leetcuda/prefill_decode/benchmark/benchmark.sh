

MODEL=/work/cache/Qwen3-0.6B

python3 benchmark_serving.py --port 9000 --seed $(date +%s) \
    --model $MODEL \
    --dataset-name random --random-input-len 7500 --random-output-len 200 \
    --num-prompts 30 --burstiness 100 --request-rate 1 --ignore-eos
