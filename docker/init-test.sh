#!/bin/bash
# init-test.sh — start PostgreSQL, install extensions, run tests
# Runs inside the container as root; drops to postgres for DB operations.
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
PG_CONFIG=${PG_CONFIG:-/usr/lib/postgresql/17/bin/pg_config}

# Initialize data directory if needed
if [ ! -s "$PGDATA/PG_VERSION" ]; then
    gosu postgres initdb --locale=C --encoding=UTF8
fi

# Start server (background)
gosu postgres pg_ctl start -D "$PGDATA" \
    -o "-c listen_addresses='localhost' -c shared_preload_libraries=''" \
    -w

# Install extensions into the running cluster
gosu postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
gosu postgres psql -c "CREATE EXTENSION IF NOT EXISTS pg_acorn;"

echo "[init-test] Extensions installed. Running tests..."

# Build and install pg_acorn from mounted source, then run regression tests
cd "$WORKSPACE"
make PG_CONFIG="$PG_CONFIG" install
make PG_CONFIG="$PG_CONFIG" installcheck PGUSER=postgres || {
    echo "[init-test] installcheck failed — dumping regression.diffs"
    cat test/regression.diffs 2>/dev/null || true
    exit 1
}

# Isolation tests (pg_isolation_regress)
make PG_CONFIG="$PG_CONFIG" isolationcheck PGUSER=postgres || {
    echo "[init-test] isolationcheck failed"
    exit 1
}

echo "[init-test] All tests passed."
