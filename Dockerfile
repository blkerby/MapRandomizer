FROM ubuntu:focal-20221019

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3.9 software-properties-common fonts-freefont-ttf

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25
RUN add-apt-repository "deb [ arch=amd64 ] https://downloads.skewed.de/apt focal main"
RUN apt-get install -y python3-graph-tool

COPY backend_requirements.txt /app/backend_requirements.txt
RUN pip3 install -r app/backend_requirements.txt
COPY maps /app/maps
COPY patches /app/patches
COPY gfx /app/gfx
COPY sm-json-data /app/sm-json-data
COPY cpp /app/cpp
COPY python /app/python
WORKDIR /app
RUN c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) cpp/reachability.cpp -o python/reachability$(python3-config --extension-suffix)
COPY CHANGELOG.html /app/
ENV PYTHONPATH /app/python
CMD ["python3", "python/rando/main.py"]
