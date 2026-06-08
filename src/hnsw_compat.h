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

/*
 * Max neighbors stored for any single layer.  pgvector's base layer (0) holds
 * HnswGetLayerM(m, 0) = 2*m neighbors, so a per-layer buffer must hold 2*m.
 */
#define HNSW_MAX_NEIGHBORS		(HNSW_MAX_M * 2)

/* Largest tuple that fits on a page (mirrors pgvector's HNSW_MAX_SIZE) */
#define HNSW_MAX_SIZE \
	(BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - \
	 MAXALIGN(sizeof(HnswPageOpaqueData)) - sizeof(ItemIdData))

/*
 * On-disk tuple sizes (MAXALIGNed), matching pgvector 0.8.0.  The element
 * header is sizeof(HnswElementTupleData) (= offsetof(data) = 72) followed by
 * the inline vector; the neighbor header is sizeof(HnswNeighborTupleData)
 * (= offsetof(indextids) = 4) followed by (level+2)*m ItemPointerData slots.
 */
#define HNSW_ELEMENT_TUPLE_SIZE(vsize) \
	MAXALIGN(sizeof(HnswElementTupleData) + (vsize))
#define HNSW_NEIGHBOR_TUPLE_SIZE(level, m) \
	MAXALIGN(sizeof(HnswNeighborTupleData) + \
			 ((level) + 2) * (m) * sizeof(ItemPointerData))

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
	uint8		version;		/* tuple format version */
	ItemPointerData heaptids[HNSW_HEAPTIDS]; /* heap TIDs (HOT chain) */
	ItemPointerData neighbortid; /* index TID of this node's neighbor tuple */
	uint16		unused;
	/* Vector data (pgvector inlines a `Vector` struct) follows here */
} HnswElementTupleData;
typedef HnswElementTupleData *HnswElementTuple;

/*
 * Access the inline vector data.
 *
 * In pgvector this is offsetof(HnswElementTupleData, data) where `data` is a
 * Vector.  Because all members above are <= 4-byte aligned and the running
 * offset (72 bytes) is already 4-aligned, sizeof(HnswElementTupleData) here
 * equals that offset exactly.
 */
#define HnswElementTupleGetVector(etup) \
	((void *) ((char *)(etup) + sizeof(HnswElementTupleData)))

/* -----------------------------------------------------------------------
 * Neighbor tuple — one per HNSW node (type = HNSW_NEIGHBOR_TUPLE_TYPE)
 *
 * IMPORTANT: the neighbor tuple lives on a SEPARATE index tuple referenced by
 * HnswElementTupleData.neighbortid (a possibly different page/offset) — NOT at
 * element_offno + 1.
 *
 * Layout: header followed by an ItemPointerData array of neighbor index TIDs.
 * Neighbors are stored highest-layer-first: for an element at `level`, the
 * array holds layer `level`, then `level-1`, ..., down to layer 0.  Each layer
 * lc occupies HnswGetLayerM(m, lc) slots (2*m for layer 0, m otherwise), so the
 * total slot count is (level + 2) * m.  Layer lc starts at index (level-lc)*m.
 * ----------------------------------------------------------------------- */

typedef struct HnswNeighborTupleData
{
	uint8		type;			/* HNSW_NEIGHBOR_TUPLE_TYPE */
	uint8		version;		/* must match element's version */
	uint16		count;			/* total slot count = (level + 2) * m */
	/* Immediately followed by ItemPointerData array of neighbor index TIDs */
} HnswNeighborTupleData;
typedef HnswNeighborTupleData *HnswNeighborTuple;

#define HnswNeighborTupleGetTids(ntup) \
	((ItemPointerData *) ((char *)(ntup) + sizeof(HnswNeighborTupleData)))

/*
 * Number of neighbor slots stored for a given layer.
 * pgvector: layer 0 (the base layer) uses 2*m, all upper layers use m.
 */
#define HnswGetLayerM(m, layer)	((layer) == 0 ? (m) * 2 : (m))

/*
 * Start index (in ItemPointerData units) of layer `layer`'s neighbors within
 * the neighbor TID array, for an element whose highest layer is `level`.
 * Matches pgvector's HnswLoadNeighborTids: start = (level - layer) * m.
 */
static inline int
HnswNeighborStart(int m, int level, int layer)
{
	return (level - layer) * m;
}

static inline int
HnswNeighborCount(int m, int layer)
{
	return HnswGetLayerM(m, layer);
}

/*
 * Layout guards — must match pgvector 0.8.0 on-disk format.  ItemPointerData
 * is 6 bytes on every PostgreSQL platform, so the element header (vector
 * offset) is 72 bytes and the neighbor header is 4 bytes.  If these fire,
 * pgvector's struct layout changed and the offsets above are wrong.
 */
StaticAssertDecl(sizeof(HnswElementTupleData) == 72,
				 "HnswElementTupleData layout mismatch vs pgvector 0.8.0");
StaticAssertDecl(sizeof(HnswNeighborTupleData) == 4,
				 "HnswNeighborTupleData layout mismatch vs pgvector 0.8.0");

#endif /* HNSW_COMPAT_H */
