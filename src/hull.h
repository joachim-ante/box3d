// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "platform.h"

#include "box3d/types.h"

#include <stddef.h>

// Big enough to handle AVX2.
#define B3_HULL_SOA_PADDING 8

static inline int b3GetHullSoaStride( int count )
{
	return ( count + B3_HULL_SOA_PADDING - 1 ) & ~( B3_HULL_SOA_PADDING - 1 );
}

uint64_t b3HashHullData( const b3HullData* hull );
bool b3CompareHullData( const b3HullData* hull1, const b3HullData* hull2 );

// Map keyed by hull content. The world hull database uses the value as a reference count,
// while compound baking stores uses the value as a byte offset. Implementation is in hull.c.
#define NAME b3HullMap
#define KEY_TY const b3HullData*
#define VAL_TY int
#define HEADER_MODE
#include "verstable.h"

// Pull in the face planes of a box hull alongside the header, they are the first thing a ray
// cast reads after it and would otherwise wait a second memory round trip. Other hulls keep
// their planes further out and only lose a harmless guess.
static inline void b3PrefetchHullPlanes( const b3HullData* hull )
{
	const char* base = (const char*)hull;
	b3Prefetch( base + offsetof( b3BoxHull, boxPlanes ) );
	b3Prefetch( base + offsetof( b3BoxHull, boxPlanes ) + 64 );
}

// Total map allocation in bytes, excluding the stored hull data
size_t b3HullMapByteCount( b3HullMap* map );
