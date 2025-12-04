FROM prefecthq/prefect:2-python3.10

COPY ./ /tmp/rca_data_tools

RUN pip install uv
RUN uv pip install --system prefect-aws /tmp/rca_data_tools
