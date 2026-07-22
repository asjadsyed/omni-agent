[private]
default:
    @just --choose

up:
    docker compose up --build --detach

attach:
    docker compose attach agent

down:
    docker compose down

down-volumes:
    docker compose down --volumes

[script("bash")]
run:
    set -Eeuo pipefail

    cleanup() {
    	status="$1"
    	trap - EXIT INT TERM HUP
    	docker compose down >/dev/null 2>&1 || true
    	exit "$status"
    }

    trap 'cleanup $?' EXIT
    trap 'cleanup 129' HUP
    trap 'cleanup 130' INT
    trap 'cleanup 143' TERM

    docker compose run --rm --service-ports --build agent
