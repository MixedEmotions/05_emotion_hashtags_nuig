from gsiupm/senpy:0.7.0-dev3-python3.5
# from nuig/senpy:0.6.1-python3.4
# from gsiupm/senpy:0.6.2-python3.4

RUN mkdir -p /senpy-plugins

RUN pip install pytest
RUN pip install regex
ADD . /senpy-plugins
RUN senpy -f /senpy-plugins --only-install

WORKDIR /senpy-plugins/
