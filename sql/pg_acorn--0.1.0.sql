\echo Use "CREATE EXTENSION pg_acorn" to load this file. \quit

-- acorn_hnsw index access method
CREATE FUNCTION acorn_hnsw_handler(internal)
    RETURNS index_am_handler
    AS 'MODULE_PATHNAME'
    LANGUAGE C;

CREATE ACCESS METHOD acorn_hnsw TYPE INDEX HANDLER acorn_hnsw_handler;

COMMENT ON ACCESS METHOD acorn_hnsw IS
    'ACORN-HNSW: filterable approximate nearest neighbor index';

-- Operator classes for acorn_hnsw.
--
-- These mirror pgvector's hnsw opclasses exactly (same operators and support
-- functions) but bind them to the acorn_hnsw AM.  Because acorn_hnsw writes
-- pages in pgvector's on-disk format, the shared traversal in acorn_scan.c
-- resolves the distance kernel via index->rd_support[0] = FUNCTION 1 here.
--
-- vector_negative_inner_product / vector_l2_squared_distance / vector_norm /
-- l1_distance are provided by the `vector` extension (required dependency).

CREATE OPERATOR CLASS vector_l2_ops
    DEFAULT FOR TYPE vector USING acorn_hnsw AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_l2_squared_distance(vector, vector);

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING acorn_hnsw AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_negative_inner_product(vector, vector);

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING acorn_hnsw AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_negative_inner_product(vector, vector),
    FUNCTION 2 vector_norm(vector);

CREATE OPERATOR CLASS vector_l1_ops
    FOR TYPE vector USING acorn_hnsw AS
    OPERATOR 1 <+> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 l1_distance(vector, vector);

-- Scalar filter opclasses for acorn_hnsw (Tier 2 in-filter ACORN).
--
-- A multi-column acorn_hnsw index (embedding vector_*_ops, attr <scalar>_acorn_ops)
-- lets the planner pass a scalar comparison (e.g. bucket < N) as an indexable
-- ScanKey on the second column; the AM evaluates it against the value stored
-- inline in each node during traversal (no heap fetch).  Strategy numbers follow
-- btree (1=<, 2=<=, 3==, 4=>=, 5=>); FUNCTION 1 is the btree-style 3-way compare.

CREATE OPERATOR CLASS int4_acorn_ops
    FOR TYPE int4 USING acorn_hnsw AS
    OPERATOR 1 < (int4, int4),
    OPERATOR 2 <= (int4, int4),
    OPERATOR 3 = (int4, int4),
    OPERATOR 4 >= (int4, int4),
    OPERATOR 5 > (int4, int4),
    FUNCTION 1 btint4cmp(int4, int4);

CREATE OPERATOR CLASS text_acorn_ops
    FOR TYPE text USING acorn_hnsw AS
    OPERATOR 1 < (text, text),
    OPERATOR 2 <= (text, text),
    OPERATOR 3 = (text, text),
    OPERATOR 4 >= (text, text),
    OPERATOR 5 > (text, text),
    FUNCTION 1 bttextcmp(text, text);

-- GUCs (loaded via _PG_init, declared here for documentation)
-- pg_acorn.enable_hook     boolean  default true   (Tier 1 hook)
-- pg_acorn.default_gamma   integer  default 1      (ACORN-1 by default)
