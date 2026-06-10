#ifndef PG_ACORN_H
#define PG_ACORN_H

/* Minimum pgvector version required for Tier 1 page layout access */
#define PG_ACORN_PGVECTOR_MIN_VERSION "0.8.0"
#define PG_ACORN_PGVECTOR_MAX_VERSION "0.8.99"

/* GUC variables declared in pg_acorn.c */
extern bool acorn_enable_hook;
extern int  acorn_default_gamma;
extern bool acorn_enable_2hop;
extern int  acorn_ef_search;
extern bool acorn_member_first;

#endif /* PG_ACORN_H */
