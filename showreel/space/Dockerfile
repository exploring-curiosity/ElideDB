# The demo as one container, sized for a free Hugging Face CPU Space
# (2 vCPU, 16 GB). It fits because the read path stopped holding the corpus:
# the ranker reads the traces a query shortlisted, from S3, and nothing else.
#
#   resident   about 1.4 GB across both processes
#   boot       about 1 s after the image is warm
#   query      20 ms text, 51 ms stage 2
#
# Three things live outside this image, and that is the architecture rather than
# an accident: the memory is CockroachDB, the corpus is S3, and the encoder is
# offline and already finished.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PRECEDENT_TRACES=/home/user/cache/traces \
    OMP_NUM_THREADS=2 \
    SHOWREEL_RELMO_PYTHON=/usr/local/bin/python

# libpq for psycopg2, plus the cluster's root explicitly.
#
# Two things that look like they should work do not. Under sslmode=verify-full
# libpq does NOT fall back to the system trust store: it opens
# ~/.postgresql/root.crt and errors when the file is absent, which in a container
# it always is. And sslrootcert=system does not rescue it either, because
# psycopg2-binary statically links its own OpenSSL and that name resolves to a
# compiled-in directory rather than this image's /etc/ssl/certs.
#
# So the root travels with the image and the DSN names it by path. root.crt is a
# public CA certificate, not a secret, and this is the same thing deploy.sh does
# for the Lambda, which is the version that was already proven to connect.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 ca-certificates && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user:user space/requirements.txt .
RUN pip install -r requirements.txt

USER user
COPY --chown=user:user . /home/user/app
# PRECEDENT_DSN must end with sslrootcert=/home/user/app/root.crt

# Bake the text tower in. Downloading 1.5 GB of weights on first request would
# make the first query look like a 90 s hang, and a Space that sleeps and wakes
# would pay it again every time.
RUN python -c "\
from transformers import AutoModel, AutoProcessor; \
m='google/siglip2-base-patch16-224'; \
AutoModel.from_pretrained(m); AutoProcessor.from_pretrained(m)" \
 && mkdir -p /home/user/cache/traces

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/healthz').read()"
CMD ["python", "showreel/server.py"]
