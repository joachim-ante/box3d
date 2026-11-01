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
	return b3BitMaskU32W4( node->child, 30 );
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

// One lane of loaded boxes in every lane. The lane must be a compile time constant so the
// splats are register shuffles rather than reloads.
#define B3_LANE_BOX_W4( BOXES, LANE )                                                                                            \
	B3_LITERAL( b3BoxW4 )                                                                                                        \
	{                                                                                                                            \
		.lowerX = b3SplatLaneW4( ( BOXES ).lowerX, LANE ), .upperX = b3SplatLaneW4( ( BOXES ).upperX, LANE ),                    \
		.lowerY = b3SplatLaneW4( ( BOXES ).lowerY, LANE ), .upperY = b3SplatLaneW4( ( BOXES ).upperY, LANE ),                    \
		.lowerZ = b3SplatLaneW4( ( BOXES ).lowerZ, LANE ), .upperZ = b3SplatLaneW4( ( BOXES ).upperZ, LANE ),                    \
	}

// One lane of a node in every lane, reloaded from memory. For the cold paths where the lane
// is a runtime value.
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

// Push the lanes in the mask onto a node stack. Four slots are always written, so the caller
// reserves four. The child words keep their flags and are masked when popped.
B3_FORCE_INLINE int b3PushLanesW4( uint32_t* stack, int stackCount, const b3WideNode* node, int mask )
{
	return stackCount + b3CompressLanesW4( stack + stackCount, node->child, mask );
}

// Node index of a popped stack word
B3_FORCE_INLINE int b3PopNodeIndex( uint32_t word )
{
	return (int)( word & B3_NODE_INDEX_MASK );
}

// Bit per lane where the query box overlaps the lane box. Empty lanes never overlap.
B3_FORCE_INLINE int b3OverlapMaskW4( const b3BoxW4* query, const b3BoxW4* boxes )
{
	b3FloatW4 mask = b3AndW4( b3LessEqW4( query->lowerX, boxes->upperX ), b3GreaterEqW4( query->upperX, boxes->lowerX ) );
	mask = b3AndW4( mask, b3AndW4( b3LessEqW4( query->lowerY, boxes->upperY ), b3GreaterEqW4( query->upperY, boxes->lowerY ) ) );
	mask = b3AndW4( mask, b3AndW4( b3LessEqW4( query->lowerZ, boxes->upperZ ), b3GreaterEqW4( query->upperZ, boxes->lowerZ ) ) );
	return b3MoveMaskW4( mask );
}

// Overlap mask of every lane of A against the four lanes of B. Each lane of A is splatted from
// the loaded registers. An empty lane of A has an inverted box and gets a zero mask.
B3_FORCE_INLINE void b3OverlapMasks4x4W4( const b3BoxW4* boxesA, const b3BoxW4* boxesB, int* masks )
{
	b3BoxW4 lane0 = B3_LANE_BOX_W4( *boxesA, 0 );
	b3BoxW4 lane1 = B3_LANE_BOX_W4( *boxesA, 1 );
	b3BoxW4 lane2 = B3_LANE_BOX_W4( *boxesA, 2 );
	b3BoxW4 lane3 = B3_LANE_BOX_W4( *boxesA, 3 );
	masks[0] = b3OverlapMaskW4( &lane0, boxesB );
	masks[1] = b3OverlapMaskW4( &lane1, boxesB );
	masks[2] = b3OverlapMaskW4( &lane2, boxesB );
	masks[3] = b3OverlapMaskW4( &lane3, boxesB );
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

// Draw the lanes of the internal nodes inside the drawing bounds, colored by depth
void b3DrawDynamicTree( const b3DynamicTree* tree, b3DebugDraw* draw );
