# concurrent_cache_evict_scan.spec
# Validates M3 eviction safety under a concurrent scan: a scanner holds an
# acorn_hnsw index scan open (a cursor keeps the per-slot active_scans ref
# raised) while another session force-evicts the SAME index's cache slot.
#
# This is exactly the eviction-vs-reader race the M3 hazard discipline must
# survive: the evictor must NOT free the DSA area while a reader can still
# dereference it.  The scan must complete with correct results regardless of
# when the evict lands; cache state never changes the answer (G4).

setup
{
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_acorn;

    CREATE TABLE ev_items (
        id        serial PRIMARY KEY,
        bucket    int,
        embedding vector(4)
    );

    INSERT INTO ev_items (bucket, embedding) SELECT
        (i % 10),
        ('[' || (i*0.1)::text || ',0.0,0.0,0.0]')::vector
    FROM generate_series(1, 400) i;

    SET pg_acorn.build_seed = 11;
    CREATE INDEX ev_acorn ON ev_items
        USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
        WITH (m = 8, ef_construction = 32, acorn_gamma = 2);
    RESET pg_acorn.build_seed;

    -- Warm the cache to READY.
    SET pg_acorn.scan_code_cache = on;
    SET enable_seqscan = off;
    SELECT count(*) FROM (
        SELECT id FROM ev_items WHERE bucket < 3
        ORDER BY embedding <-> '[1.0,0.0,0.0,0.0]' LIMIT 10) s;
    RESET enable_seqscan;
    RESET pg_acorn.scan_code_cache;
}

teardown
{
    DROP TABLE ev_items CASCADE;
}

# Scanner: a cursor keeps the index scan (and its active_scans ref) open
# across steps so the evict races a live reader.
session "scanner"
setup           { SET pg_acorn.scan_code_cache = on; SET enable_seqscan = off; }
step "s_begin"  { BEGIN; }
step "s_open"   {
    DECLARE cur CURSOR FOR
        SELECT id FROM ev_items
        WHERE bucket < 3
        ORDER BY embedding <-> '[1.0,0.0,0.0,0.0]';
}
step "s_fetch1" { FETCH 5 FROM cur; }
step "s_fetch2" { FETCH 5 FROM cur; }
step "s_close"  { CLOSE cur; COMMIT; }

# Manipulator: force-evict the slot the scanner is reading.
session "evictor"
step "e_evict"  { SELECT pg_acorn_code_cache_evict('ev_acorn'); }

# Rescan after eviction must reload and stay correct.
session "checker"
setup           { SET pg_acorn.scan_code_cache = on; SET enable_seqscan = off; }
step "c_scan"   {
    SELECT count(*) FROM (
        SELECT id FROM ev_items WHERE bucket < 3
        ORDER BY embedding <-> '[1.0,0.0,0.0,0.0]' LIMIT 10) s;
}

# Evict lands mid-scan (between the two fetches): the area must not be freed
# under the open cursor; the scan completes correctly.
permutation "s_begin" "s_open" "s_fetch1" "e_evict" "s_fetch2" "s_close" "c_scan"
# Evict before the scan opens its cursor: clean reload on the next scan.
permutation "e_evict" "s_begin" "s_open" "s_fetch1" "s_fetch2" "s_close" "c_scan"
# Evict after the cursor closes: slot already quiescent.
permutation "s_begin" "s_open" "s_fetch1" "s_fetch2" "s_close" "e_evict" "c_scan"
