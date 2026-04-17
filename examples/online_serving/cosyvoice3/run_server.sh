#!/bin/bash
# Launch vLLM-Omni server for CosyVoice3
#
# Usage:
#   ./run_server.sh                    # Default: async_chunk mode (streaming)
#   ./run_server.sh async_chunk        # Async chunk mode (streaming, lower TTFA)
#   ./run_server.sh sync               # Sync mode (non-streaming)
#
# Set COSYVOICE3_MODEL to override the model path, e.g.:
#   COSYVOICE3_MODEL=/local/path/to/model ./run_server.sh

set -e

MODE="${1:-async_chunk}"
MODEL="${COSYVOICE3_MODEL:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
TOKENIZER="${MODEL}/CosyVoice-BlankEN"

case "$MODE" in
    async_chunk)
        STAGE_CONFIG="vllm_omni/model_executor/stage_configs/cosyvoice3_async_chunk.yaml"
        ;;
    sync)
        STAGE_CONFIG="vllm_omni/model_executor/stage_configs/cosyvoice3.yaml"
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
    --stage-configs-path "$STAGE_CONFIG" \
    --host 0.0.0.0 \
    --port 8091 \
    --trust-remote-code \
    --stage-init-timeout 900 \
    --init-timeout 1200 \
    --omni
