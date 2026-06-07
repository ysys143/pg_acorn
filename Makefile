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
	tier1_hook \
	tier2_am \
	recall_filter \
	recall_gamma \
	recall_insert \
	no_regression

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
