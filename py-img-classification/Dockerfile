FROM rocm/tensorflow@sha256:3d5ba86a2cc3b6a4c2e160f6bffc7c5503e14e2fbb8d9712bc70ec8708f72d8c

VOLUME /tmp/.X11-unix

ENV DISPLAY=:0
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

WORKDIR /pic

RUN apt-get update && apt-get install -y python3-tk

RUN pip3 install matplotlib cryptography~=42.0.5

CMD ["bash"]

# Help
# sudo docker build -t pic-amd-gpu .
# sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --mount type=bind,source="$(pwd)",target=/pic pic-amd-gpu
