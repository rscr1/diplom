FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# RUN useradd -m ${USER} --uid=${UID} -s /bin/bash && echo "${USER}:${PW}" | chpasswd
RUN echo root:1 | chpasswd

ENV TZ=Europe/Moscow  
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# General installs
RUN apt -y update && apt -y upgrade && apt install -y openssh-server && apt install -y automake && apt install -y sudo \
    && apt install -y wget && apt install -y curl && apt install -y git && apt install -y net-tools && apt install -y gcc \
    && apt install -y g++ && apt install -y nano

RUN apt update && apt -y install libpcre3 libpcre3-dev && apt-get -y install iputils-ping && apt-get install -y python3-opencv

RUN wget http://intra.gosniias.lan/intra/docs/root-ca.crt -O /usr/local/share/ca-certificates/root_ca.crt && \
    wget http://intra.gosniias.lan/intra/docs/gitlab.gosniias.lan.crt -O /usr/local/share/ca-certificates/gitlab.gosniias.lan.crt && \
    wget http://intra.gosniias.lan/intra/docs/isuing-ca.crt -O /usr/local/share/ca-certificates/isuing-ca.crt && \
    update-ca-certificates

ENV REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'
echo REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt' >> /home/docker_user/.bashrc


# SSH acess set
RUN mkdir /var/run/sshd
# RUN echo 'root:pass' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# PIP installs
RUN pip install ipython -U && pip install ipywidgets -U && \
    pip install jupyterlab -U && pip install -U virtualenv

RUN pip install -U opencv-contrib-python && pip install opencv-python

RUN service ssh start

EXPOSE 8888
EXPOSE 22
