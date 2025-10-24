FROM nvcr.io/nvidia/pytorch:24.05-py3  # CUDA 12-based

RUN pip install --no-cache-dir tensorrt-llm[serve]

# Copy pre-built engine; or build inside by uncommenting two lines
COPY engines/llama2-7b /models/llama
# COPY build.sh /tmp/build.sh
# RUN bash /tmp/build.sh

EXPOSE 6060
ENTRYPOINT ["trtllm-serve", "/models/llama", "--model-type", "llama",
            "--enable-inflight-batching", "--port", "6060"]
