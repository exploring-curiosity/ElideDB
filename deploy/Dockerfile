# ElideDB demo service. Build from the REPO ROOT:
#   docker build -f deploy/Dockerfile -t elidedb-demo .
# Run:
#   docker run -p 7860:7860 elidedb-demo
#
# The image carries the code and the standalone demo store
# (deploy/demo/lake, built by scripts/build_demo_store.py). Model
# weights are fetched from Hugging Face on first start and cached in
# the container; ~5 GB once, then queries run warm at about a second
# on two CPU cores.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r deploy/requirements.txt

COPY python /app/python
COPY scripts/get_iv2.py /app/scripts/get_iv2.py
COPY deploy/entrypoint.sh /app/deploy/entrypoint.sh
COPY deploy/serve.py /app/deploy/serve.py
# the demo store arrives at boot from DEMO_STORE_DATASET when set
# (Space git repos cap at 1 GB); local builds may bake it instead:
#   COPY deploy/demo/lake /app/lake
RUN chmod +x /app/deploy/entrypoint.sh

ENV PYTHONPATH=/app/python \
    ELIDEDB_DEVICE=cpu \
    ELIDEDB_TEXT_BACKEND=torch \
    ELIDEDB_TEXT_ONLY=1 \
    DEMO_STORE_DATASET=SudharshanR/elidedb-demo-store \
    DESK_READONLY=1 \
    DESK_HOST=0.0.0.0 \
    PORT=7860 \
    HF_HOME=/app/hf_cache

EXPOSE 7860
CMD ["/app/deploy/entrypoint.sh"]
