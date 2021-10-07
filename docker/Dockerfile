FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
COPY requirements.txt .
RUN apt-get update -qq
RUN apt-get install -qq apt-utils curl git
RUN ln -snf /usr/share/zoneinfo/$(curl https://ipapi.co/timezone) /etc/localtime
RUN apt-get update -qq && apt-get install -qq libmysqlclient-dev libsystemd-dev libcups2-dev libcairo2-dev gdb wget libcurl4-openssl-dev libboost-all-dev libssl-dev cmake
RUN git clone https://github.com/Tencent/rapidjson.git && cd rapidjson && cmake . && make install
RUN git clone https://github.com/zaphoyd/websocketpp.git && cd websocketpp && cmake . && make install
RUN git clone https://github.com/jpbarrette/curlpp.git && cd curlpp && cmake . && make install
RUN git clone https://github.com/SOCI/soci.git && cd soci && cmake -DSOCI_CXX11=ON . && make install
RUN apt-get install -qq python3-pip
RUN python3 -m pip install -r requirements.txt
