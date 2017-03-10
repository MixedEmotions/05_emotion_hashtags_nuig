from gsiupm/senpy:0.8.4-python3.5
# from nuig/senpy:0.7.0-dev3-python3.5

RUN mkdir -p /senpy-plugins

RUN pip install pytest
RUN pip install regex
ADD . /senpy-plugins
COPY logo-Insight.png /usr/src/app/senpy/static/img/gsi.png
RUN senpy -f /senpy-plugins --only-install

WORKDIR /senpy-plugins/
