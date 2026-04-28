#!/bin/bash
# Launch vLLM-Omni server for CosyVoice3.
#
# Usage:
#   ./run_server.sh              # async_chunk mode
#   ./run_server.sh async_chunk
#   ./run_server.sh sync
#
# Set COSYVOICE3_MODEL to override the model path.
# Set COSYVOICE3_TOKENIZER to override the tokenizer path.

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

MODE="${1:-async_chunk}"
MODEL="${COSYVOICE3_MODEL:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
TOKENIZER="${COSYVOICE3_TOKENIZER:-${MODEL}/CosyVoice-BlankEN}"
DEPLOY_CONFIG="${COSYVOICE3_DEPLOY_CONFIG:-${REPO_ROOT}/vllm_omni/deploy/cosyvoice3.yaml}"
EXTRA_ARGS=()

case "$MODE" in
    async_chunk)
        ;;
    sync)
        EXTRA_ARGS+=("--no-async-chunk")
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Supported: async_chunk, sync"
        exit 1
        ;;
esac

echo "Starting CosyVoice3 server with model: $MODEL (mode: $MODE)"

vllm-omni serve "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --deploy-config "$DEPLOY_CONFIG" \
    --host 0.0.0.0 \
    --port 8091 \
    --trust-remote-code \
    --stage-init-timeout 900 \
    --init-timeout 1200 \
    --omni \
    "${EXTRA_ARGS[@]}"
