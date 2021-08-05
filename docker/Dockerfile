# Usage: docker build --build-arg UNAME=$(id -un) --build-arg UID=$(id -u) --build-arg GID=$(id -g) --tag $(id -un)/rbexplore .

FROM nvcr.io/nvidia/pytorch:20.06-py3

RUN pip install --upgrade pip

COPY custom-requirements.txt /root/
RUN cat /root/custom-requirements.txt | xargs -n 1 -L 1 pip install -U

COPY requirements.txt /root/
RUN cat /root/requirements.txt | xargs -n 1 -L 1 pip install

COPY requirements-upgrade.txt /root/
RUN cat /root/requirements-upgrade.txt | xargs -n 1 -L 1 pip install -U

COPY atari_roms.zip /root/
RUN unzip -d /opt/conda/lib/python3.6/site-packages/atari_py/ /root/atari_roms.zip

ENV OMP_NUM_THREADS=1
ARG UNAME
ARG GID
ARG UID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME
WORKDIR /workspace
