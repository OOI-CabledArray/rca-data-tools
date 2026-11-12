FROM prefecthq/prefect:3-python3.11

COPY ./ /tmp/rca_data_tools

RUN pip install uv
RUN uv pip install --system prefect-aws /tmp/rca_data_tools
