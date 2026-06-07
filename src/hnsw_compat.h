/*
 * hnsw_compat.h — pgvector 0.8.x internal page structure declarations
 *
 * Tier 1 reads pgvector's HNSW index pages directly.  These declarations
 * must match exactly with pgvector 0.8.x.  _PG_init() verifies the installed
 * pgvector version before enabling Tier 1.
 *
 * Source reference: https://github.com/pgvector/pgvector/tree/v0.8.0/src/hnsw.h
 *
 * NOTE: If you update pgvector, re-verify these structs against hnsw.h.
 */

#ifndef HNSW_COMPAT_H
#define HNSW_COMPAT_H

#include "postgres.h"
#include "storage/bufmgr.h"
#include "storage/itemptr.h"
#include "storage/off.h"

/* -----------------------------------------------------------------------
 * Constants (must match pgvector 0.8.x hnsw.h)
 * ----------------------------------------------------------------------- */

#define HNSW_PAGE_ID			0xFF80
#define HNSW_METAPAGE_BLKNO		0
#define HNSW_ELEMENT_TUPLE_TYPE	1
#define HNSW_NEIGHBOR_TUPLE_TYPE	2
#define HNSW_HEAPTIDS			10
#define HNSW_MAGIC_NUMBER		0xA953A953
#define HNSW_VERSION			1
#define HNSW_DEFAULT_M			16
#define HNSW_MAX_M				100
#define HNSW_DEFAULT_EF_CONSTRUCTION	64

/* -----------------------------------------------------------------------
 * Page opaque — at PageGetSpecialPointer(page)
 * ----------------------------------------------------------------------- */

typedef struct HnswPageOpaqueData
{
	BlockNumber nextblkno;
	uint16		unused;
	uint16		page_id;		/* HNSW_PAGE_ID */
} HnswPageOpaqueData;
typedef HnswPageOpaqueData *HnswPageOpaque;

#define HnswPageGetOpaque(page) \
	((HnswPageOpaque) PageGetSpecialPointer(page))

#define HnswPageIsValid(page) \
	(HnswPageGetOpaque(page)->page_id == HNSW_PAGE_ID)

/* -----------------------------------------------------------------------
 * Meta page (block 0)
 * ----------------------------------------------------------------------- */

typedef struct HnswMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	uint32		dimensions;
	uint16		m;					/* M parameter */
	uint16		efConstruction;
	BlockNumber entryBlkno;			/* entry point location */
	OffsetNumber entryOffno;
	int16		entryLevel;			/* highest level in graph */
	BlockNumber insertPage;			/* page where next insert goes */
} HnswMetaPageData;
typedef HnswMetaPageData *HnswMetaPage;

#define HnswPageGetMeta(page)	((HnswMetaPage) PageGetContents(page))

/* -----------------------------------------------------------------------
 * Element tuple — one per HNSW node (type = HNSW_ELEMENT_TUPLE_TYPE)
 *
 * Layout on page: HnswElementTupleData header followed immediately by
 * the vector bytes (FLEXIBLE_ARRAY_MEMBER in pgvector's actual definition).
 * The vector size is fixed per index: sizeof(Vector) + dim * sizeof(float).
 * ----------------------------------------------------------------------- */

typedef struct HnswElementTupleData
{
	uint8		type;			/* HNSW_ELEMENT_TUPLE_TYPE */
	uint8		level;			/* highest layer this node appears in */
	uint8		deleted;		/* 1 = logically deleted */
	uint8		unused;
	ItemPointerData heaptids[HNSW_HEAPTIDS]; /* heap TIDs (HOT chain) */
	/* Vector data follows immediately here */
} HnswElementTupleData;
typedef HnswElementTupleData *HnswElementTuple;

/* Access the inline vector data */
#define HnswElementTupleGetVector(etup) \
	((void *) ((char *)(etup) + sizeof(HnswElementTupleData)))

/* -----------------------------------------------------------------------
 * Neighbor tuple — one per HNSW node (type = HNSW_NEIGHBOR_TUPLE_TYPE)
 *
 * In pgvector 0.8.x, the neighbor tuple immediately follows the element
 * tuple on the same index page (offset = element_offno + 1).
 *
 * Layout: header followed by an array of ItemPointerData (index TIDs).
 * Total count = sum over each layer of (layer == 0 ? m : m0), where m0
 * is typically floor(m/2).  The layer 0 neighbors come first.
 * ----------------------------------------------------------------------- */

typedef struct HnswNeighborTupleData
{
	uint8		type;			/* HNSW_NEIGHBOR_TUPLE_TYPE */
	uint8		unused[3];
	/* Immediately followed by ItemPointerData array of neighbor index TIDs */
} HnswNeighborTupleData;
typedef HnswNeighborTupleData *HnswNeighborTuple;

#define HnswNeighborTupleGetTids(ntup) \
	((ItemPointerData *) ((char *)(ntup) + sizeof(HnswNeighborTupleData)))

/*
 * Number of layer-0 neighbors stored per node.
 * In pgvector, layer 0 uses m, layers 1+ use max(1, floor(m/2)).
 */
#define HnswM0(m)	(m)
#define HnswMl(m)	Max(1, (m) / 2)

/*
 * Byte offset to the start of layer l's neighbors within the neighbor TID
 * array.  Layer 0 neighbors start at index 0 (size m0), layer 1 at m0, etc.
 */
static inline int
HnswNeighborOffset(int m, int layer)
{
	int off = HnswM0(m);		/* layer 0 */
	for (int l = 1; l <= layer; l++)
		off += HnswMl(m);		/* layers 1..layer-1 already counted */
	/* Return the start of layer l, not the running total */
	return (layer == 0) ? 0 : HnswM0(m) + HnswMl(m) * (layer - 1);
}

static inline int
HnswNeighborCount(int m, int layer)
{
	return (layer == 0) ? HnswM0(m) : HnswMl(m);
}

#endif /* HNSW_COMPAT_H */
