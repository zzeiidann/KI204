#!/bin/bash
# Simple evaluation script using curl

API_URL="http://localhost:8080"

echo "=========================================="
echo "LLM SERVICE EVALUATION"
echo "=========================================="

# Check health
echo -e "\n1. Health Check:"
curl -s $API_URL/health
echo

# Get model metadata
echo -e "\n2. Model Information:"
curl -s $API_URL/metadata | python3 -m json.tool

# Test prompts
prompts=(
    "Artificial intelligence is"
    "The future of technology"
    "Machine learning enables"
    "Neural networks are"
    "Deep learning models"
)

echo -e "\n=========================================="
echo "INFERENCE SPEED TEST"
echo "=========================================="

total_time=0
total_tokens=0
count=0

for prompt in "${prompts[@]}"; do
    echo -e "\nPrompt: '$prompt'"
    
    # Run 3 times for each prompt
    for i in {1..3}; do
        result=$(curl -s -X POST $API_URL/generate \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"$prompt\", \"max_new_tokens\": 20}")
        
        time_ms=$(echo $result | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_time_ms', 0))")
        tokens=$(echo $result | python3 -c "import sys, json; print(json.load(sys.stdin).get('tokens_generated', 0))")
        tps=$(echo $result | python3 -c "import sys, json; print(json.load(sys.stdin).get('tokens_per_second', 0))")
        
        echo "  Run $i: ${time_ms}ms, ${tokens} tokens, ${tps} tokens/s"
        
        total_time=$(echo "$total_time + $time_ms" | bc)
        total_tokens=$(echo "$total_tokens + $tokens" | bc)
        count=$((count + 1))
    done
done

echo -e "\n=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
avg_time=$(echo "scale=2; $total_time / $count" | bc)
avg_tokens=$(echo "scale=2; $total_tokens / $count" | bc)
avg_tps=$(echo "scale=2; ($total_tokens * 1000) / $total_time" | bc)

echo "Total runs: $count"
echo "Average latency: ${avg_time}ms"
echo "Average tokens generated: $avg_tokens"
echo "Average throughput: ${avg_tps} tokens/s"

echo -e "\n=========================================="
echo "GENERATION QUALITY SAMPLES"
echo "=========================================="

# Test 3 different prompts with longer output
test_prompts=(
    "Artificial intelligence is"
    "The benefits of machine learning include"
    "In the future, robots will"
)

for prompt in "${test_prompts[@]}"; do
    echo -e "\nPrompt: '$prompt'"
    curl -s -X POST $API_URL/generate \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\", \"max_new_tokens\": 30}" | \
        python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Output: {d.get('completion', 'N/A')}\"); print(f\"Tokens: {d.get('tokens_generated', 0)}, Time: {d.get('total_time_ms', 0)}ms\")"
done

echo -e "\n=========================================="
echo "EVALUATION COMPLETE"
echo "=========================================="
