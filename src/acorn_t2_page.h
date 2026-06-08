/*
 * acorn_t2_page.h — Tier 2 (acorn_hnsw) own on-disk element tuple format
 *
 * Tier 2 diverges from pgvector's hnsw element format by storing an inline
 * filter column value (filter_val) in each element tuple.  This enables
 * in-filter ACORN traversal — the AM evaluates scalar predicates directly
 * against the stored value without any heap fetch.
 *
 * Neighbor tuples and all page / meta structures are inherited from
 * hnsw_compat.h unchanged.  Only the element tuple gains the extra field.
 *
 * Tier 1 (acorn_hook.c, acorn_scan.c batch path) reads pgvector's own HNSW
 * pages via hnsw_compat.h.  Do not mix these two element formats.
 */

#ifndef ACORN_T2_PAGE_H
#define ACORN_T2_PAGE_H

#include "hnsw_compat.h"	/* neighbor tuple, page opaque, meta, all macros */

/*
 * Tier 2 element tuple.
 *
 * The header fields through `unused` sit at the same byte offsets as
 * HnswElementTupleData (type, level, deleted, version, heaptids,
 * neighbortid, unused = bytes 0–71).  filter_val is appended at offset 72,
 * pushing the inline vector to offset 80.  For single-column indexes (no
 * filter) filter_val is stored as 0 and never evaluated.
 *
 * filter_val stores a by-value filter column datum: int4, int8, oid, bool.
 * For int4 it is stored as (int64)(Datum)values[1] and compared via
 * btint4cmp using the ScanKey's sk_func.
 */
typedef struct AcornT2ElementTupleData
{
	uint8			type;
	uint8			level;
	uint8			deleted;
	uint8			version;
	ItemPointerData heaptids[HNSW_HEAPTIDS];	/* heap TID(s) */
	ItemPointerData neighbortid;				/* index TID of neighbor tuple */
	uint16			unused;
	int64			filter_val;		/* inline filter column (0 = no filter) */
	/* Vector data follows immediately at sizeof(AcornT2ElementTupleData) */
} AcornT2ElementTupleData;
typedef AcornT2ElementTupleData *AcornT2ElementTuple;

StaticAssertDecl(sizeof(AcornT2ElementTupleData) == 80,
				 "AcornT2ElementTupleData must be 80 bytes (72 header + 8 filter_val)");

/* Access the inline vector — offset 80, not 72 as in pgvector. */
#define AcornT2ElementTupleGetVector(etup) \
	((void *) ((char *)(etup) + sizeof(AcornT2ElementTupleData)))

/* On-disk tuple size including the vector varlena. */
#define ACORN_T2_ELEMENT_TUPLE_SIZE(vsize) \
	MAXALIGN(sizeof(AcornT2ElementTupleData) + (vsize))

#endif /* ACORN_T2_PAGE_H */
