default_goal: all

MODEL := ./models/gpt-oss-20b-MXFP4.gguf

init: $(MODEL)

$(MODEL):
	wget -c -O "./models/gpt-oss-20b-MXFP4.gguf" "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf?download=true"

.PHONY: run
run: init
	docker compose run -it --rm --service-ports agent

.PHONY: all
all: run
