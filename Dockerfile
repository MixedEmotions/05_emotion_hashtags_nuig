from gsiupm/senpy:0.6.1-python2.7

RUN mkdir -p /senpy-plugins

RUN pip install pytest
ADD . /senpy-plugins
RUN senpy -f /senpy-plugins --only-install

WORKDIR /senpy-plugins/
