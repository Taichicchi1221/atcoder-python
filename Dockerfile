FROM python:3.8
ARG USERNAME=vscode
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

# install python packages
RUN pip install --upgrade pip \
    && pip install numpy==1.18.2 \
    && pip install scipy==1.4.1 \
    && pip install scikit-learn==0.22.2.post1 \
    && pip install numba==0.48.0 \
    && pip install networkx==2.4 \
    && pip install pylint \
    && pip install autopep8 \ 
    && pip install online-judge-tools
