
docker rm -f test-py39 && \
docker inspect triton | grep IPAddr && \
docker run --rm -it --name test-py39 -u $(id -u) -v $(pwd):/home/jovyan mirekphd/ml-cpu-py39-jup-cust:latest python /home/jovyan/tritonclient-script.py
