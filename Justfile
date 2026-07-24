MODEL := "./models/gpt-oss-20b-MXFP4.gguf"
MODEL_SIZE_FILE := MODEL + ".size"
MODEL_URL := "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf?download=true"

[private]
default:
    @just --choose

build provider="nvidia":
    PROVIDER="{{provider}}" docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" build --progress=plain

up provider="nvidia":
    PROVIDER="{{provider}}" docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" up --build --detach

attach provider="nvidia":
    PROVIDER="{{provider}}" docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" attach agent

down provider="nvidia":
    PROVIDER="{{provider}}" docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" down --remove-orphans

[confirm("Warning: this will permanently delete stored data. Continue?")]
down-volumes provider="nvidia":
    PROVIDER="{{provider}}" docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" down --remove-orphans --volumes

[script("bash")]
run provider="nvidia":
    set -Eeuo pipefail
    just init "{{provider}}"

    cleanup() {
    	status="$1"
    	trap - EXIT INT TERM HUP
    	PROVIDER="{{provider}}" docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" down >/dev/null 2>&1 || true
    	exit "$status"
    }

    trap 'cleanup $?' EXIT
    trap 'cleanup 129' HUP
    trap 'cleanup 130' INT
    trap 'cleanup 143' TERM

    PROVIDER="{{provider}}" docker compose -f docker-compose.yml -f "docker-compose.{{provider}}.yml" run --rm --service-ports --build agent

[script("bash")]
init provider="llama_cpp":
    if [ "{{provider}}" = "llama_cpp" ]; then
        mkdir -p models

        local_size=$(python3 -c 'import os,sys; print(os.path.getsize(sys.argv[1]))' "{{MODEL}}" 2>/dev/null || echo 0)
        if [ -f "{{MODEL_SIZE_FILE}}" ]; then
            remote_size=$(cat "{{MODEL_SIZE_FILE}}")
        else
            remote_size=$(curl -fsSLI "{{MODEL_URL}}" | awk 'tolower($1)=="content-length:" {size=$2} END {print size}' | tr -d '\r')
            echo "$remote_size" > "{{MODEL_SIZE_FILE}}"
        fi

        if [ "$local_size" != "$remote_size" ]; then
            wget -c -O "{{MODEL}}" "{{MODEL_URL}}"
        fi
    fi
