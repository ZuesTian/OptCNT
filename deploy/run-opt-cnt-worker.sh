#!/usr/bin/env bash
set -euo pipefail

set -a
source /etc/opt-cnt-worker/worker.env
source /etc/opt-cnt-worker/proxy.env
set +a
cd /opt/opt-cnt-worker/current
exec sudo \
    --preserve-env=OPTCNT_SERVER_URL,OPTCNT_WORKER_API_TOKEN,OPTCNT_WORKER_ID,OPTCNT_WORKER_CAPACITY,HTTP_PROXY,HTTPS_PROXY,NO_PROXY \
    -u amd \
    /opt/opt-cnt-worker/runtime/.venv/bin/python -m src.webapp.worker
