FROM python:3.10-slim-bullseye
ARG USERNAME=atcoder
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    && apt-get -y install git iproute2 procps lsb-release gdb\
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt autoremove -y \
    && apt clean -y

ENV PATH /home/vscode/.local/bin:$PATH

COPY requirements.txt .

# install python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/not522/ac-library-python
