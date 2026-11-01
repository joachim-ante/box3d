// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "simd.h"

#include "box3d/collision.h"
#include "box3d/math_functions.h"

#define B3_TREE_STACK_SIZE 512

// Used to mark a lane as being moved such that pairs need to be generated and the tree
// may need to be rebuilt. This mark comes from a few sources:
// - updating the transform and AABB in the solver from movement
// - a joint disabling collision
// - creating a proxy
// - setting the transform on a body
// The mark propagates up through the parent lanes to the root.

#define B3_MOVED_NODE ( 1u << 30 )
#define B3_LEAF_NODE ( 1u << 31 )
#define B3_NODE_INDEX_MASK ( 0xFFFFFFFFu & ~( B3_MOVED_NODE | B3_LEAF_NODE ) )

#define B3_ROOT_NODE 0

B3_FORCE_INLINE bool b3IsLeaf( const b3WideNode* node )
{
	return ( node->child[0] & B3_LEAF_NODE ) == B3_LEAF_NODE;
}

B3_FORCE_INLINE bool b3IsLaneEmpty( uint32_t child )
{
	return ( child & B3_NODE_INDEX_MASK ) == B3_NODE_INDEX_MASK;
}

B3_FORCE_INLINE bool b3IsLaneMoved( uint32_t child )
{
	return ( child & B3_MOVED_NODE ) == B3_MOVED_NODE;
}

B3_FORCE_INLINE int b3GetLaneIndex( uint32_t child )
{
	return (int)( child & B3_NODE_INDEX_MASK );
}

// Bit per lane that is moved
B3_FORCE_INLINE int b3GetMovedMask( const b3WideNode* node )
{
	return (int)( ( ( node->child[0] >> 30 ) & 1 ) | ( ( node->child[1] >> 29 ) & 2 ) | ( ( node->child[2] >> 28 ) & 4 ) |
				  ( ( node->child[3] >> 27 ) & 8 ) );
}

B3_FORCE_INLINE bool b3HasMovedLane( const b3WideNode* node )
{
	return ( ( node->child[0] | node->child[1] | node->child[2] | node->child[3] ) & B3_MOVED_NODE ) != 0;
}

B3_FORCE_INLINE b3AABB b3GetLaneAABB( const b3WideNode* node, int lane )
{
	b3AABB aabb = {
		{ node->lowerX[lane], node->lowerY[lane], node->lowerZ[lane] },
		{ node->upperX[lane], node->upperY[lane], node->upperZ[lane] },
	};
	return aabb;
}

// Four boxes transposed into SIMD registers. The node layout loads straight into this.
typedef struct b3BoxW4
{
	b3FloatW4 lowerX, upperX;
	b3FloatW4 lowerY, upperY;
	b3FloatW4 lowerZ, upperZ;
} b3BoxW4;

B3_FORCE_INLINE b3BoxW4 b3LoadBoxW4( const b3WideNode* node )
{
	b3BoxW4 boxes;
	boxes.lowerX = b3LoadW4( node->lowerX );
	boxes.upperX = b3LoadW4( node->upperX );
	boxes.lowerY = b3LoadW4( node->lowerY );
	boxes.upperY = b3LoadW4( node->upperY );
	boxes.lowerZ = b3LoadW4( node->lowerZ );
	boxes.upperZ = b3LoadW4( node->upperZ );
	return boxes;
}

// One lane of a node in every lane
B3_FORCE_INLINE b3BoxW4 b3LaneBoxW4( const b3WideNode* node, int lane )
{
	b3BoxW4 boxes;
	boxes.lowerX = b3SplatW4( node->lowerX[lane] );
	boxes.upperX = b3SplatW4( node->upperX[lane] );
	boxes.lowerY = b3SplatW4( node->lowerY[lane] );
	boxes.upperY = b3SplatW4( node->upperY[lane] );
	boxes.lowerZ = b3SplatW4( node->lowerZ[lane] );
	boxes.upperZ = b3SplatW4( node->upperZ[lane] );
	return boxes;
}

// Bit per lane where the query box overlaps the lane box. Empty lanes never overlap.
B3_FORCE_INLINE int b3OverlapMaskW4( const b3BoxW4* query, const b3BoxW4* boxes )
{
	b3FloatW4 mask = b3AndW4( b3LessEqW4( query->lowerX, boxes->upperX ), b3GreaterEqW4( query->upperX, boxes->lowerX ) );
	mask = b3AndW4( mask, b3AndW4( b3LessEqW4( query->lowerY, boxes->upperY ), b3GreaterEqW4( query->upperY, boxes->lowerY ) ) );
	mask = b3AndW4( mask, b3AndW4( b3LessEqW4( query->lowerZ, boxes->upperZ ), b3GreaterEqW4( query->upperZ, boxes->lowerZ ) ) );
	return b3MoveMaskW4( mask );
}

typedef struct b3TreeView
{
	const b3WideNode* nodes;
	const b3TreeProxy* proxies;
} b3TreeView;

b3TreeStats b3TreeView_Query( const b3TreeView* view, b3AABB aabb, uint64_t maskBits, bool requireAllBits,
							  b3TreeQueryCallbackFcn* callback, void* context );
b3TreeStats b3TreeView_QueryClosest( const b3TreeView* view, b3Vec3 point, uint64_t maskBits, bool requireAllBits,
									 b3TreeQueryClosestCallbackFcn* callback, void* context, float* minDistanceSqr );
b3TreeStats b3TreeView_RayCast( const b3TreeView* view, const b3RayCastInput* input, uint64_t maskBits, bool requireAllBits,
								b3TreeRayCastCallbackFcn* callback, void* context );
b3TreeStats b3TreeView_BoxCast( const b3TreeView* view, const b3BoxCastInput* input, uint64_t maskBits, bool requireAllBits,
								b3TreeBoxCastCallbackFcn* callback, void* context );

static inline bool b3HasTreeMoved( const b3DynamicTree* tree )
{
	return b3HasMovedLane( tree->nodes + B3_ROOT_NODE );
}

static inline bool b3NeedsRebuild( const b3DynamicTree* tree )
{
	return b3HasTreeMoved( tree ) || tree->dfsOrdered == false;
}

int b3CreateTreeProxyInternal( b3DynamicTree* tree, b3AABB aabb, uint64_t categoryBits, uint64_t userData, bool markMoved );
void b3DynamicTree_MoveProxyInternal( b3DynamicTree* tree, int proxyId, b3AABB aabb, bool markMoved );

void b3DynamicTree_MarkProxyMovedSerial( b3DynamicTree* tree, int proxyId );
void b3DynamicTree_MarkProxyMoved( b3DynamicTree* tree, int proxyId, b3AABB aabb );
void b3DynamicTree_ClearMoved( b3DynamicTree* tree );
void b3DynamicTree_Refit( b3DynamicTree* tree );
