FROM postgres:17-bookworm

# Build tools + PostgreSQL 17 server headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    postgresql-server-dev-17 \
    python3 \
    python3-pip \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# pgvector 0.8.0 — pinned; Tier 1 reads its page layout directly
RUN git clone --depth 1 --branch v0.8.0 \
    https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config \
    && make install PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config \
    && rm -rf /tmp/pgvector

# Python bench dependencies (optional — only needed for docker-bench)
COPY bench/requirements.txt /tmp/bench-requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/bench-requirements.txt

# pg_acorn source is volume-mounted at /workspace at runtime.
# The image contains only the toolchain; no source copy here.

ENV PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config
ENV PGDATA=/var/lib/postgresql/data
ENV PGUSER=postgres

COPY docker/init-test.sh /usr/local/bin/init-test.sh
RUN chmod +x /usr/local/bin/init-test.sh
