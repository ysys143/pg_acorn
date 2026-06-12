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
	src/acorn_cost.o \
	src/acorn_dist.o \
	src/acorn_codecache.o

# Distance kernels must compile exactly like pgvector 0.8.0 (same flags ->
# same auto-vectorized summation order -> bit-identical distances vs fmgr).
# Mirrors pgvector's Makefile OPTFLAGS logic (no -march=native on Darwin/arm).
DIST_OPTFLAGS = -march=native
ifeq ($(shell uname -s), Darwin)
	ifeq ($(shell uname -p), arm)
		DIST_OPTFLAGS =
	endif
endif
src/acorn_dist.o: CFLAGS += $(DIST_OPTFLAGS) -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math

REGRESS = \
	smoke \
	tier1_hook \
	tier2_am \
	tier2_infilter \
	tier2_2hop \
	tier2_ef_search \
	tier2_payload_edges \
	tier2_diversify \
	tier2_inline_vectors \
	tier2_code_cache \
	tier2_code_cache_dml \
	tier2_code_cache_evict \
	tier2_emission_order \
	tier2_build_mwm \
	tier2_build_parallel \
	no_regression \
	recall_filter \
	recall_gamma \
	recall_insert

REGRESS_OPTS = --inputdir=test --outputdir=test

ISOLATION      = concurrent_insert_scan concurrent_gamma_build concurrent_cache_insert_scan concurrent_cache_evict_scan
ISOLATION_OPTS = --inputdir=test --outputdir=test

PG_CPPFLAGS = -I./src -Wno-unused-parameter
PG_CONFIG   ?= pg_config
PGXS        := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Header dependency tracking (PGXS implicit rule does not track .h changes)
src/pg_acorn.o:   src/acorn_hook.h src/acorn_am.h
src/acorn_hook.o: src/acorn_hook.h src/acorn_scan.h
src/acorn_am.o:   src/acorn_am.h   src/acorn_scan.h src/acorn_cost.h src/acorn_codecache.h
src/acorn_build.o: src/acorn_am.h   src/hnsw_compat.h src/acorn_t2_page.h src/pg_acorn.h src/acorn_dist.h src/acorn_scan.h src/acorn_codecache.h
src/acorn_scan.o:  src/acorn_scan.h src/hnsw_compat.h src/acorn_t2_page.h src/pg_acorn.h src/acorn_dist.h src/acorn_codecache.h
src/acorn_cost.o:  src/acorn_cost.h src/acorn_am.h
src/acorn_dist.o:  src/acorn_dist.h
src/acorn_codecache.o: src/acorn_codecache.h src/hnsw_compat.h src/acorn_t2_page.h src/pg_acorn.h src/acorn_dist.h

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

# -----------------------------------------------------------------------
# GCP VM remote targets — mirrors pg_cuvs gpu-* workflow
#
#   make vm-start              start GCP instance
#   make vm-stop               stop GCP instance
#   make vm-ip                 show current external IP
#   make sync                  rsync local → VM
#   make vm-build              build extension on VM
#   make vm-install            install extension on VM
#   make vm-test               regression + isolation on VM
#   make vm-test-all           full ladder: unit → regress → isolation
#   make vm-bench-cohere       Cohere 50M benchmark (async, nohup)
#   make vm-bench-cohere-log   tail benchmark log
#   make vm-bench-cohere-result pull result CSV
#   make vm-bench-incremental  incremental insert recall harness (async)
#   make vm-shell              interactive SSH session
#   make vm-sql                ad-hoc SQL via stdin
#   make vm-cycle              sync → build → install → test
# -----------------------------------------------------------------------

-include .env.vm
export

GCP_USER     ?= ubuntu
VM_IP         = $(shell gcloud compute instances describe $(GCP_INSTANCE) \
                    --zone $(GCP_ZONE) \
                    $(if $(GCP_PROJECT),--project $(GCP_PROJECT)) \
                    --format='value(networkInterfaces[0].accessConfigs[0].natIP)' \
                    2>/dev/null)
VM_HOST       = $(if $(VM_IP),$(GCP_USER)@$(VM_IP),$(GCP_VM))
unexport VM_IP VM_HOST

.PHONY: vm-start vm-stop vm-ip sync \
	vm-build vm-install vm-test \
	vm-test-unit vm-test-regress vm-test-isolation vm-test-all \
	vm-bench-cohere vm-bench-cohere-log vm-bench-cohere-result \
	vm-bench-incremental vm-bench-incremental-log vm-bench-incremental-result \
	vm-shell vm-sql vm-cycle

vm-start:
	@test -n "$(GCP_INSTANCE)" || (echo "ERROR: set GCP_INSTANCE in .env.vm"; exit 1)
	@test -n "$(GCP_PROJECT)"  || (echo "ERROR: set GCP_PROJECT in .env.vm";  exit 1)
	gcloud compute instances start $(GCP_INSTANCE) --zone $(GCP_ZONE) --project $(GCP_PROJECT)
	@echo "Waiting for SSH..."
	@until ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $(VM_HOST) true 2>/dev/null; \
		do sleep 3; done
	@echo "VM ready: $(VM_HOST)"

vm-stop:
	@test -n "$(GCP_PROJECT)" || (echo "ERROR: set GCP_PROJECT in .env.vm"; exit 1)
	gcloud compute instances stop $(GCP_INSTANCE) --zone $(GCP_ZONE) --project $(GCP_PROJECT)

vm-ip:
	@gcloud compute instances describe $(GCP_INSTANCE) --zone $(GCP_ZONE) \
		$(if $(GCP_PROJECT),--project $(GCP_PROJECT)) \
		--format='value(status,networkInterfaces[0].accessConfigs[0].natIP)'

# rsync local → VM. Excludes build artifacts so VM's compiled .o/.so stay intact.
sync:
	rsync -avz --delete \
		--exclude '.git' \
		--exclude 'src/*.o' \
		--exclude 'src/*.bc' \
		--exclude '*.so' \
		--exclude '.env.vm' \
		./ $(VM_HOST):~/pg_acorn/

vm-build:
	ssh -tt $(VM_HOST) "cd ~/pg_acorn && \
		make PG_CONFIG=$(PG_CONFIG_REMOTE) 2>&1 | tee /tmp/pg_acorn_build.log"

vm-install:
	ssh -tt $(VM_HOST) "cd ~/pg_acorn && \
		sudo make install PG_CONFIG=$(PG_CONFIG_REMOTE)"

vm-test:
	ssh -tt $(VM_HOST) "cd ~/pg_acorn && \
		make installcheck PG_CONFIG=$(PG_CONFIG_REMOTE) PGUSER=postgres"

vm-test-unit:
	ssh -tt $(VM_HOST) "cd ~/pg_acorn && \
		make unit"

vm-test-regress:
	ssh -tt $(VM_HOST) "cd ~/pg_acorn && \
		make installcheck PG_CONFIG=$(PG_CONFIG_REMOTE) PGUSER=postgres"

vm-test-isolation:
	ssh -tt $(VM_HOST) "cd ~/pg_acorn && \
		make installcheck-isolation PG_CONFIG=$(PG_CONFIG_REMOTE) PGUSER=postgres"

vm-test-all: vm-test-unit vm-test-regress vm-test-isolation

# Cohere 50M benchmark — runs async via nohup; poll with vm-bench-cohere-log.
# N defaults to 5000000 (5M subset for iteration speed; override for full 35M).
vm-bench-cohere:
	@mkdir -p design/bench
	@echo "[vm-bench-cohere] Launching Cohere benchmark (nohup, async)"
	ssh $(VM_HOST) "cd ~/pg_acorn && \
		nohup python3 bench/harness/cohere_bench.py \
			--n $(if $(N),$(N),5000000) \
			--scenarios $(if $(SCENARIOS),$(SCENARIOS),selectivity,range,multi) \
			> /tmp/cohere_bench.log 2>&1 &"
	@echo "Poll: make vm-bench-cohere-log"
	@echo "Pull: make vm-bench-cohere-result"

vm-bench-cohere-log:
	ssh $(VM_HOST) "tail -60 /tmp/cohere_bench.log"

vm-bench-cohere-result:
	@mkdir -p design/bench
	ssh $(VM_HOST) "cat ~/pg_acorn/bench/results/cohere_summary.jsonl 2>/dev/null \
		|| echo 'Not ready — check: make vm-bench-cohere-log'"
	rsync -avz $(VM_HOST):~/pg_acorn/bench/results/ design/bench/ 2>/dev/null || true

# Incremental insert recall harness — inserts in rounds, measures recall@10 at
# each checkpoint, asserts >= 0.85, writes design/bench/incremental.jsonl.
vm-bench-incremental:
	@mkdir -p design/bench
	@echo "[vm-bench-incremental] Launching incremental harness (nohup, async)"
	ssh $(VM_HOST) "cd ~/pg_acorn && \
		nohup python3 bench/harness/incremental.py \
			--initial $(if $(INITIAL),$(INITIAL),10000000) \
			--batch   $(if $(BATCH),$(BATCH),5000000) \
			--rounds  $(if $(ROUNDS),$(ROUNDS),5) \
			--recall-floor 0.85 \
			> /tmp/incremental_bench.log 2>&1 &"
	@echo "Poll: make vm-bench-incremental-log"
	@echo "Pull: make vm-bench-incremental-result"

vm-bench-incremental-log:
	ssh $(VM_HOST) "tail -60 /tmp/incremental_bench.log"

vm-bench-incremental-result:
	@mkdir -p design/bench
	rsync -avz $(VM_HOST):~/pg_acorn/bench/results/incremental.jsonl design/bench/ 2>/dev/null || true
	@cat design/bench/incremental.jsonl 2>/dev/null | python3 -c "\
import sys, json; rows = [json.loads(l) for l in sys.stdin]; \
[print(f\"round {r['round']:2d}  rows={r['total_rows']:>10,}  recall={r['recall_at_10']:.3f}  qps={r['qps']:.0f}\") for r in rows]" \
	2>/dev/null || echo "(no results yet)"

vm-shell:
	ssh -tt $(VM_HOST)

# Run ad-hoc SQL from stdin. Usage: make vm-sql < query.sql
vm-sql:
	ssh -o StrictHostKeyChecking=accept-new $(VM_HOST) \
		"psql -d $(if $(DB),$(DB),postgres) -U postgres -P pager=off -A -F '|'"

vm-cycle: sync vm-build vm-install vm-test
