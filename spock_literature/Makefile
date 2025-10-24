mps:
	@echo "Activating MPS..."
	@source activate_mps.sh

# -------- configurable variables ----------
PYTHON      ?= python3.10
VENV        ?= .venv
SRC_MODEL   ?= meta-llama/Llama-2-7b-chat-hf
ENGINE_DIR  ?= engines/llama2-7b
PORT        ?= 6060
TENSORT_SERVICE_DIR ?= tensort_service
# ------------------------------------------

.PHONY: help venv deps build-engine serve test clean

help:
	@echo "Targets: venv  deps  build-engine  serve  test  clean"

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip wheel

deps: venv
	. $(VENV)/bin/activate && \
	    pip install tensorrt-llm[serve] torch==2.3.0 --extra-index-url https://download.pytorch.org/whl/cu121

build-engine: deps
	. $(VENV)/bin/activate && \
	    python -m tensorrt_llm.commands.build \
	        --checkpoint_dir $(SRC_MODEL) \
	        --dtype fp8 \
	        --output_dir $(ENGINE_DIR)

serve:
	. $(VENV)/bin/activate && \
	    trtllm-serve $(ENGINE_DIR) \
	        --model-type llama \
	        --enable-inflight-batching \
	        --port $(PORT)

test:
	. $(VENV)/bin/activate && pytest -v tests/

clean:
	rm -rf $(ENGINE_DIR) $(VENV)

compile_grpc:
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. chat.proto

