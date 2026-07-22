MODEL := "./models/gpt-oss-20b-MXFP4.gguf"
MODEL_URL := "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf?download=true"

[private]
default:
    @just --choose


up provider="llama_cpp":
    docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" up --build --detach

attach provider="llama_cpp":
    docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" attach agent

down provider="llama_cpp":
    docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" down

down-volumes provider="llama_cpp":
    docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" down --volumes

[script("bash")]
run provider="llama_cpp":
    set -Eeuo pipefail
    just init "{{provider}}"

    cleanup() {
    	status="$1"
    	trap - EXIT INT TERM HUP
    	docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" down >/dev/null 2>&1 || true
    	exit "$status"
    }

    trap 'cleanup $?' EXIT
    trap 'cleanup 129' HUP
    trap 'cleanup 130' INT
    trap 'cleanup 143' TERM

    docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" run --rm --service-ports --build agent

[script("bash")]
init provider="llama_cpp":
    if [ "{{provider}}" = "llama_cpp" ]; then
        mkdir -p models
        wget -c -O "{{MODEL}}" "{{MODEL_URL}}"
    fi
