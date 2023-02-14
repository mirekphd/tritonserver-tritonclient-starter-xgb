docker rm -f triton && \
docker run -d --rm --name triton -p 8000-8002 -v $(pwd)/model_repository:/model_repository nvcr.io/nvidia/tritonserver:23.01-py3 tritonserver --model-repository=/model_repository && \
docker logs -f triton
