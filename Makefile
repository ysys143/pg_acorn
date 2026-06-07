EXTENSION      = pg_acorn
EXTVERSION     = 0.1.0
DATA           = sql/pg_acorn--$(EXTVERSION).sql
MODULE_big     = pg_acorn

OBJS = \
	src/pg_acorn.o \
	src/acorn_hook.o \
	src/acorn_am.o \
	src/acorn_build.o \
	src/acorn_scan.o \
	src/acorn_cost.o

REGRESS = \
	smoke \
	tier1_hook
# Step 3 tests (acorn_hnsw AM required — re-enable after implementing acorn_build.c):
# tier2_am recall_filter recall_gamma recall_insert no_regression

REGRESS_OPTS = --inputdir=test --outputdir=test

ISOLATION      = concurrent_insert_scan concurrent_gamma_build
ISOLATION_OPTS = --inputdir=test --outputdir=test

PG_CPPFLAGS = -I./src -Wno-unused-parameter
PG_CONFIG   ?= pg_config
PGXS        := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Header dependency tracking (PGXS implicit rule does not track .h changes)
src/pg_acorn.o:   src/acorn_hook.h src/acorn_am.h
src/acorn_hook.o: src/acorn_hook.h src/acorn_scan.h
src/acorn_am.o:   src/acorn_am.h   src/acorn_scan.h src/acorn_cost.h
src/acorn_build.o: src/acorn_am.h
src/acorn_scan.o:  src/acorn_scan.h src/hnsw_compat.h
src/acorn_cost.o:  src/acorn_cost.h

# Unit tests — standalone C binaries, no PostgreSQL dependency.
# Tests cover algorithm logic extracted in test/unit/*.c (self-contained stubs).
.PHONY: unit
unit:
	$(CC) $(CFLAGS) -o test/unit/test_acorn_scan test/unit/test_acorn_scan.c
	$(CC) $(CFLAGS) -o test/unit/test_acorn_build test/unit/test_acorn_build.c
	test/unit/test_acorn_scan
	test/unit/test_acorn_build

.PHONY: bench
bench:
	python3 bench/run_bench.py $(BENCH_ARGS)

# -----------------------------------------------------------------------
# Docker targets — mirrors pg_cuvs VM-based workflow
#
#   make docker-build   build test image (PG17 + pgvector 0.8.0)
#   make docker-test    regression + isolation tests inside container
#   make docker-unit    standalone C unit tests (no PG needed)
#   make docker-bench   compose up: postgres + qdrant, then run_bench.py
#   make docker-shell   interactive bash session in container
#   make docker-clean   remove containers and image
# -----------------------------------------------------------------------

DOCKER_IMAGE   = pg_acorn_test
DOCKER_RUN     = docker run --rm \
    -v $(CURDIR):/workspace \
    -e WORKSPACE=/workspace \
    -w /workspace \
    $(DOCKER_IMAGE)

.PHONY: docker-build docker-test docker-unit docker-bench docker-shell docker-clean

docker-build:
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE) .

docker-test: docker-build
	$(DOCKER_RUN) bash -c "\
	    make -C /workspace PG_CONFIG=\$$PG_CONFIG \
	    && /usr/local/bin/init-test.sh"

docker-unit:
	$(DOCKER_RUN) make -C /workspace unit

docker-bench: docker-build
	BENCH_SCENARIO=$(BENCH_SCENARIO) \
	docker compose -f docker/docker-compose.yml \
	    --profile bench run --rm bench

docker-shell: docker-build
	docker run --rm -it \
	    -v $(CURDIR):/workspace \
	    -w /workspace \
	    $(DOCKER_IMAGE) bash

docker-clean:
	docker compose -f docker/docker-compose.yml down -v 2>/dev/null || true
	docker rmi $(DOCKER_IMAGE) 2>/dev/null || true
