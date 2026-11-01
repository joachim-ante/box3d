// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#include "dynamic_tree.h"

#include "aabb.h"
#include "algorithm.h"
#include "core.h"
#include "ctz.h"
#include "platform.h"
#include "simd.h"

#include "box3d/collision.h"
#include "box3d/constants.h"
#include "box3d/math_functions.h"

#include <float.h>
#include <stdio.h>
#include <string.h>

_Static_assert( B3_ROOT_NODE == 0, "bad root" );
_Static_assert( sizeof( b3WideNode ) == 128, "expected size" );
_Static_assert( sizeof( b3TreeProxy ) == 24, "expected size" );

// Ranges this small are sorted along one axis and chunked into full nodes instead of split.
#define B3_SMALL_RANGE_SIZE 32

// The box of the subtree below a node is the union of its four lane boxes. This is what the
// lane above the node stores, up to any enlargement. Empty lanes drop out of the min and max.
// A node with no valid lanes yields an inverted infinite box.
B3_FORCE_INLINE b3AABB b3GetSubtreeAABB( const b3WideNode* node )
{
	b3AABB aabb;
	aabb.lowerBound.x = b3HorizontalMinW4( b3LoadW4( node->lowerX ) );
	aabb.lowerBound.y = b3HorizontalMinW4( b3LoadW4( node->lowerY ) );
	aabb.lowerBound.z = b3HorizontalMinW4( b3LoadW4( node->lowerZ ) );
	aabb.upperBound.x = b3HorizontalMaxW4( b3LoadW4( node->upperX ) );
	aabb.upperBound.y = b3HorizontalMaxW4( b3LoadW4( node->upperY ) );
	aabb.upperBound.z = b3HorizontalMaxW4( b3LoadW4( node->upperZ ) );
	return aabb;
}


// Empty lanes keep the leaf bit of their node so the leaf test can read any lane.
#define B3_EMPTY_NODE_LANE B3_NODE_INDEX_MASK
#define B3_EMPTY_LEAF_LANE ( B3_NODE_INDEX_MASK | B3_LEAF_NODE )

// A lane reference packs a node index and a lane
#define B3_LANE_REF( NODE, LANE ) ( ( NODE ) * 4 + ( LANE ) )
#define B3_LANE_REF_NODE( REF ) ( ( REF ) >> 2 )
#define B3_LANE_REF_LANE( REF ) ( ( REF ) & 3 )

B3_FORCE_INLINE int b3GetLaneCount( const b3WideNode* node )
{
	int count = 0;
	for ( int i = 0; i < 4; ++i )
	{
		count += b3IsLaneEmpty( node->child[i] ) ? 0 : 1;
	}
	return count;
}

B3_FORCE_INLINE void b3SetLaneAABB( b3WideNode* node, int lane, b3AABB aabb )
{
	node->lowerX[lane] = aabb.lowerBound.x;
	node->lowerY[lane] = aabb.lowerBound.y;
	node->lowerZ[lane] = aabb.lowerBound.z;
	node->upperX[lane] = aabb.upperBound.x;
	node->upperY[lane] = aabb.upperBound.y;
	node->upperZ[lane] = aabb.upperBound.z;
}

// An empty lane has an inverted infinite box so it fails every overlap test
B3_FORCE_INLINE void b3SetLaneEmpty( b3WideNode* node, int lane, bool leaf )
{
	node->lowerX[lane] = INFINITY;
	node->lowerY[lane] = INFINITY;
	node->lowerZ[lane] = INFINITY;
	node->upperX[lane] = -INFINITY;
	node->upperY[lane] = -INFINITY;
	node->upperZ[lane] = -INFINITY;
	node->child[lane] = leaf ? B3_EMPTY_LEAF_LANE : B3_EMPTY_NODE_LANE;
	node->userData[lane] = 0;
}

B3_FORCE_INLINE b3WideNode b3MakeEmptyNode( bool leaf )
{
	b3WideNode node;
	for ( int i = 0; i < 4; ++i )
	{
		b3SetLaneEmpty( &node, i, leaf );
	}
	return node;
}

// One box in every lane
B3_FORCE_INLINE b3BoxW4 b3SplatBoxW4( b3AABB aabb )
{
	b3BoxW4 boxes;
	boxes.lowerX = b3SplatW4( aabb.lowerBound.x );
	boxes.upperX = b3SplatW4( aabb.upperBound.x );
	boxes.lowerY = b3SplatW4( aabb.lowerBound.y );
	boxes.upperY = b3SplatW4( aabb.upperBound.y );
	boxes.lowerZ = b3SplatW4( aabb.lowerBound.z );
	boxes.upperZ = b3SplatW4( aabb.upperBound.z );
	return boxes;
}

// Squared distance from a point to each lane box. Empty lanes are infinite.
B3_FORCE_INLINE b3FloatW4 b3DistanceSqrW4( b3FloatW4 px, b3FloatW4 py, b3FloatW4 pz, const b3BoxW4* boxes )
{
	b3FloatW4 dx = b3SubW4( b3MinW4( b3MaxW4( px, boxes->lowerX ), boxes->upperX ), px );
	b3FloatW4 dy = b3SubW4( b3MinW4( b3MaxW4( py, boxes->lowerY ), boxes->upperY ), py );
	b3FloatW4 dz = b3SubW4( b3MinW4( b3MaxW4( pz, boxes->lowerZ ), boxes->upperZ ), pz );
	return b3AddW4( b3AddW4( b3MulW4( dx, dx ), b3MulW4( dy, dy ) ), b3MulW4( dz, dz ) );
}

// Slab test of a ray against four boxes. The reciprocal direction must be finite so the products
// never produce NaN. Returns a bit per hit lane and the entry fraction per lane.
B3_FORCE_INLINE b3FloatW4 b3RayMaskW4( b3Vec3 origin, b3Vec3 invDirection, float maxFraction, const b3BoxW4* boxes,
									   b3FloatW4* nearFractions )
{
	b3FloatW4 ox = b3SplatW4( origin.x );
	b3FloatW4 oy = b3SplatW4( origin.y );
	b3FloatW4 oz = b3SplatW4( origin.z );
	b3FloatW4 ix = b3SplatW4( invDirection.x );
	b3FloatW4 iy = b3SplatW4( invDirection.y );
	b3FloatW4 iz = b3SplatW4( invDirection.z );

	b3FloatW4 t1x = b3MulW4( b3SubW4( boxes->lowerX, ox ), ix );
	b3FloatW4 t2x = b3MulW4( b3SubW4( boxes->upperX, ox ), ix );
	b3FloatW4 t1y = b3MulW4( b3SubW4( boxes->lowerY, oy ), iy );
	b3FloatW4 t2y = b3MulW4( b3SubW4( boxes->upperY, oy ), iy );
	b3FloatW4 t1z = b3MulW4( b3SubW4( boxes->lowerZ, oz ), iz );
	b3FloatW4 t2z = b3MulW4( b3SubW4( boxes->upperZ, oz ), iz );

	b3FloatW4 nearT = b3MaxW4( b3MaxW4( b3MinW4( t1x, t2x ), b3MinW4( t1y, t2y ) ), b3MaxW4( b3MinW4( t1z, t2z ), b3ZeroW4() ) );
	b3FloatW4 farT =
		b3MinW4( b3MinW4( b3MaxW4( t1x, t2x ), b3MaxW4( t1y, t2y ) ), b3MinW4( b3MaxW4( t1z, t2z ), b3SplatW4( maxFraction ) ) );

	*nearFractions = nearT;
	return b3AndW4( b3LessEqW4( nearT, farT ), b3LessEqW4( boxes->lowerX, boxes->upperX ) );
}

// Finite reciprocal that cannot make NaN with a zero numerator
B3_FORCE_INLINE b3Vec3 b3InverseDirection( b3Vec3 d )
{
	b3Vec3 inv;
	inv.x = b3AbsFloat( d.x ) > FLT_MIN ? 1.0f / d.x : ( d.x < 0.0f ? -FLT_MAX : FLT_MAX );
	inv.y = b3AbsFloat( d.y ) > FLT_MIN ? 1.0f / d.y : ( d.y < 0.0f ? -FLT_MAX : FLT_MAX );
	inv.z = b3AbsFloat( d.z ) > FLT_MIN ? 1.0f / d.z : ( d.z < 0.0f ? -FLT_MAX : FLT_MAX );
	return inv;
}

static inline b3TreeView b3MakeTreeView( const b3DynamicTree* tree )
{
	b3TreeView view = {
		.nodes = tree->nodes,
		.proxies = tree->proxies,
	};
	return view;
}

static void b3GrowNodes( b3DynamicTree* tree, int newCapacity )
{
	int oldCapacity = tree->nodeCapacity;
	B3_ASSERT( newCapacity > oldCapacity );
	tree->nodes = B3_GROW_ZERO( tree->nodes, oldCapacity, newCapacity );
	tree->parents = B3_GROW_ZERO( tree->parents, oldCapacity, newCapacity );
	tree->nodeCapacity = newCapacity;

	// The spare has to match, the rebuild allocates it again.
	b3Free( tree->swapNodes, oldCapacity * sizeof( b3WideNode ) );
	tree->swapNodes = NULL;
}

// The free list is chained through the parent index of free nodes. A node comes from the
// free list when there is one and from the bump pointer otherwise. A rebuild empties the free
// list, so allocations can just bump.
static int b3AllocateNode( b3DynamicTree* tree, bool leaf )
{
	int index;
	if ( tree->nodeFreeList != B3_NULL_INDEX )
	{
		index = tree->nodeFreeList;
		tree->nodeFreeList = tree->parents[index];
	}
	else
	{
		if ( tree->nodeEnd == tree->nodeCapacity )
		{
			int oldCapacity = tree->nodeCapacity;
			b3GrowNodes( tree, oldCapacity + b3MaxInt( oldCapacity >> 1, 1 ) );
		}

		index = tree->nodeEnd;
		tree->nodeEnd += 1;
	}

	tree->nodes[index] = b3MakeEmptyNode( leaf );
	tree->parents[index] = B3_NULL_INDEX;
	return index;
}

static void b3FreeNode( b3DynamicTree* tree, int index )
{
	B3_ASSERT( B3_ROOT_NODE < index && index < tree->nodeEnd );
	tree->nodes[index] = b3MakeEmptyNode( true );
	tree->parents[index] = tree->nodeFreeList;
	tree->nodeFreeList = index;
}

b3DynamicTree b3DynamicTree_Create( int proxyCapacity )
{
	int capacity = b3MaxInt( proxyCapacity, 16 );

	// Intentionally _not_ initialized with brace initialization, which can leave
	// uninitialized gaps.
	b3DynamicTree tree;

	// memset needed for deterministic serialization
	memset( &tree, 0, sizeof( b3DynamicTree ) );

	tree.version = B3_DYNAMIC_TREE_VERSION;

	// A node holds up to four proxies, so this is generous. Growth is handled anyway.
	tree.nodeCapacity = capacity;
	tree.nodes = (b3WideNode*)b3AllocZero( tree.nodeCapacity * sizeof( b3WideNode ) );
	tree.parents = (int32_t*)b3AllocZero( tree.nodeCapacity * sizeof( int32_t ) );
	tree.nodeFreeList = B3_NULL_INDEX;

	// The root always exists. An empty tree has a leaf root with no lanes.
	tree.nodes[B3_ROOT_NODE] = b3MakeEmptyNode( true );
	tree.parents[B3_ROOT_NODE] = B3_NULL_INDEX;

	// This is the bump index.
	tree.nodeEnd = 1;
	tree.dfsOrdered = true;

	tree.proxyCapacity = capacity;
	tree.proxyCount = 0;
	tree.proxies = (b3TreeProxy*)b3AllocZero( tree.proxyCapacity * sizeof( b3TreeProxy ) );

	// Build a linked list for the free list.
	for ( int i = 0; i < tree.proxyCapacity - 1; ++i )
	{
		tree.proxies[i].leaf = B3_NULL_INDEX;
		tree.proxies[i].next = i + 1;
	}

	tree.proxies[tree.proxyCapacity - 1].leaf = B3_NULL_INDEX;
	tree.proxies[tree.proxyCapacity - 1].next = B3_NULL_INDEX;
	tree.proxyFreeList = 0;

	tree.swapNodes = NULL;
	tree.leafIndices = NULL;
	tree.leafRefs = NULL;
	tree.leafBoxes = NULL;
	tree.leafCenters = NULL;
	tree.builtNodes = NULL;
	tree.rebuildCapacity = 0;

	return tree;
}

void b3DynamicTree_Destroy( b3DynamicTree* tree )
{
	b3Free( tree->nodes, tree->nodeCapacity * sizeof( b3WideNode ) );
	b3Free( tree->parents, tree->nodeCapacity * sizeof( int32_t ) );
	b3Free( tree->proxies, tree->proxyCapacity * sizeof( b3TreeProxy ) );
	b3Free( tree->swapNodes, tree->nodeCapacity * sizeof( b3WideNode ) );
	b3Free( tree->leafIndices, tree->rebuildCapacity * sizeof( int32_t ) );
	b3Free( tree->leafRefs, tree->rebuildCapacity * sizeof( int32_t ) );
	b3Free( tree->leafBoxes, tree->rebuildCapacity * sizeof( b3AABB ) );
	b3Free( tree->leafCenters, tree->rebuildCapacity * sizeof( b3Vec3 ) );
	b3Free( tree->builtNodes, tree->rebuildCapacity * sizeof( int32_t ) );

	memset( tree, 0, sizeof( b3DynamicTree ) );
}

// Allocate a proxy from the pool. Grow the pool if necessary.
static int b3AllocateProxy( b3DynamicTree* tree )
{
	// Expand the pool as needed.
	if ( tree->proxyFreeList == B3_NULL_INDEX )
	{
		B3_ASSERT( tree->proxyCount == tree->proxyCapacity );

		// The free list is empty. Rebuild a bigger pool.
		// A restored pool can have capacity 1, so guarantee strict growth.
		int oldCapacity = tree->proxyCapacity;
		tree->proxyCapacity += b3MaxInt( oldCapacity >> 1, 1 );
		tree->proxies = B3_GROW_ZERO( tree->proxies, oldCapacity, tree->proxyCapacity );

		// Build a linked list for the free list.
		for ( int i = oldCapacity; i < tree->proxyCapacity - 1; ++i )
		{
			tree->proxies[i].leaf = B3_NULL_INDEX;
			tree->proxies[i].next = i + 1;
		}

		tree->proxies[tree->proxyCapacity - 1].leaf = B3_NULL_INDEX;
		tree->proxies[tree->proxyCapacity - 1].next = B3_NULL_INDEX;
		tree->proxyFreeList = oldCapacity;
	}

	// Peel a proxy off the free list.
	int proxyIndex = tree->proxyFreeList;
	tree->proxyFreeList = tree->proxies[proxyIndex].next;
	memset( tree->proxies + proxyIndex, 0, sizeof( b3TreeProxy ) );
	tree->proxies[proxyIndex].leaf = B3_NULL_INDEX;
	tree->proxies[proxyIndex].next = B3_NULL_INDEX;
	++tree->proxyCount;
	return proxyIndex;
}

// Return a proxy to the pool.
static void b3FreeProxy( b3DynamicTree* tree, int proxyId )
{
	B3_ASSERT( 0 <= proxyId && proxyId < tree->proxyCapacity );
	B3_ASSERT( 0 < tree->proxyCount );
	tree->proxies[proxyId].leaf = B3_NULL_INDEX;
	tree->proxies[proxyId].next = tree->proxyFreeList;
	tree->proxyFreeList = proxyId;
	--tree->proxyCount;
}

static inline int b3FindFreeLane( const b3WideNode* node )
{
	for ( int i = 0; i < 4; ++i )
	{
		if ( b3IsLaneEmpty( node->child[i] ) )
		{
			return i;
		}
	}

	return B3_NULL_INDEX;
}

// The lane word for a child node: the index plus the moved flag gathered from its lanes
static inline uint32_t b3MakeNodeLane( const b3WideNode* nodes, int childIndex )
{
	return (uint32_t)childIndex | ( b3HasMovedLane( nodes + childIndex ) ? B3_MOVED_NODE : 0 );
}

// Write the lane of a parent that holds a child node: box and moved flag come from the child.
static inline void b3SetNodeLane( b3WideNode* nodes, int parent, int lane, int childIndex )
{
	b3WideNode* node = nodes + parent;
	b3SetLaneAABB( node, lane, b3GetSubtreeAABB( nodes + childIndex ) );
	node->child[lane] = b3MakeNodeLane( nodes, childIndex );
}

// The lanes of a node's ancestors are recomputed from the child below them. Starting at the
// given node, the lane holding it in its parent is refreshed, then the lane above and so on.
static void b3RefitAncestors( b3DynamicTree* tree, int index )
{
	b3WideNode* nodes = tree->nodes;
	const int32_t* parents = tree->parents;

	int ref = parents[index];
	while ( ref != B3_NULL_INDEX )
	{
		int parent = B3_LANE_REF_NODE( ref );
		b3SetNodeLane( nodes, parent, B3_LANE_REF_LANE( ref ), index );
		index = parent;
		ref = parents[index];
	}
}

// A lane as a free standing value for splitting nodes.
typedef struct b3LaneItem
{
	b3AABB aabb;
	uint32_t child;
	uint32_t userData;
} b3LaneItem;

static inline b3LaneItem b3GetLaneItem( const b3WideNode* node, int lane )
{
	b3LaneItem item = {
		.aabb = b3GetLaneAABB( node, lane ),
		.child = node->child[lane],
		.userData = node->userData[lane],
	};
	return item;
}

// Write items into an empty node and link whatever hangs below them.
static void b3FillNode( b3DynamicTree* tree, int index, const b3LaneItem* items, int count, bool leaf )
{
	B3_ASSERT( 0 < count && count <= 4 );
	b3WideNode* node = tree->nodes + index;
	*node = b3MakeEmptyNode( leaf );

	for ( int i = 0; i < count; ++i )
	{
		b3SetLaneAABB( node, i, items[i].aabb );
		node->child[i] = items[i].child;
		node->userData[i] = items[i].userData;

		int ref = B3_LANE_REF( index, i );
		if ( leaf )
		{
			B3_ASSERT( ( items[i].child & B3_LEAF_NODE ) != 0 );
			tree->proxies[b3GetLaneIndex( items[i].child )].leaf = ref;
		}
		else
		{
			B3_ASSERT( ( items[i].child & B3_LEAF_NODE ) == 0 );
			int child = b3GetLaneIndex( items[i].child );
			tree->parents[child] = ref;

			// The sweep refit visits indices from high to low, so it needs every child above its
			// parent
			if ( child < index )
			{
				tree->dfsOrdered = false;
			}
		}
	}
}

// Partition five items into two groups by the surface area heuristic. The items are sorted
// along each axis and the two balanced cuts are costed. The items are reordered so the first
// group leads and its count is returned.
static int b3SplitItems( b3LaneItem* items )
{
	const int count = 5;
	int bestOrder[5] = { 0, 1, 2, 3, 4 };
	int bestSplit = 2;
	float bestCost = FLT_MAX;

	for ( int axis = 0; axis < 3; ++axis )
	{
		float keys[5];
		int order[5];
		for ( int i = 0; i < count; ++i )
		{
			b3Vec3 center = b3AABB_Center( items[i].aabb );
			keys[i] = axis == 0 ? center.x : ( axis == 1 ? center.y : center.z );
			order[i] = i;
		}

		// Insertion sort
		for ( int i = 1; i < count; ++i )
		{
			int value = order[i];
			float key = keys[value];
			int j = i;
			while ( j > 0 && key < keys[order[j - 1]] )
			{
				order[j] = order[j - 1];
				j -= 1;
			}
			order[j] = value;
		}

		b3AABB prefix[5];
		prefix[0] = items[order[0]].aabb;
		for ( int i = 1; i < count; ++i )
		{
			prefix[i] = b3AABB_Union( prefix[i - 1], items[order[i]].aabb );
		}

		b3AABB suffix[5];
		suffix[count - 1] = items[order[count - 1]].aabb;
		for ( int i = count - 2; i >= 0; --i )
		{
			suffix[i] = b3AABB_Union( suffix[i + 1], items[order[i]].aabb );
		}

		for ( int split = 2; split <= 3; ++split )
		{
			float cost =
				(float)split * b3Perimeter( prefix[split - 1] ) + (float)( count - split ) * b3Perimeter( suffix[split] );
			if ( cost < bestCost )
			{
				bestCost = cost;
				bestSplit = split;
				memcpy( bestOrder, order, sizeof( order ) );
			}
		}
	}

	b3LaneItem sorted[5];
	for ( int i = 0; i < count; ++i )
	{
		sorted[i] = items[bestOrder[i]];
	}
	memcpy( items, sorted, sizeof( sorted ) );

	return bestSplit;
}

// Node index became full and one more item has to go in. The five items are split into two
// groups. The node keeps the first group and a new node takes the second, then the new node
// is inserted into the parent, splitting it in turn when it is full too. A split root moves
// both groups down into new nodes so the root stays at index zero.
static void b3SplitNode( b3DynamicTree* tree, int index, b3LaneItem* items, bool leaf )
{
	for ( ;; )
	{
		int countA = b3SplitItems( items );
		int countB = 5 - countA;

		if ( index == B3_ROOT_NODE )
		{
			int a = b3AllocateNode( tree, leaf );
			int b = b3AllocateNode( tree, leaf );
			b3FillNode( tree, a, items, countA, leaf );
			b3FillNode( tree, b, items + countA, countB, leaf );

			b3WideNode* nodes = tree->nodes;
			nodes[B3_ROOT_NODE] = b3MakeEmptyNode( false );
			b3SetNodeLane( nodes, B3_ROOT_NODE, 0, a );
			b3SetNodeLane( nodes, B3_ROOT_NODE, 1, b );
			tree->parents[a] = B3_LANE_REF( B3_ROOT_NODE, 0 );
			tree->parents[b] = B3_LANE_REF( B3_ROOT_NODE, 1 );
			return;
		}

		int b = b3AllocateNode( tree, leaf );
		b3FillNode( tree, index, items, countA, leaf );
		b3FillNode( tree, b, items + countA, countB, leaf );

		b3WideNode* nodes = tree->nodes;
		int ref = tree->parents[index];
		int parent = B3_LANE_REF_NODE( ref );

		// The node shrank, so its lane is refreshed before the parent is considered.
		b3SetNodeLane( nodes, parent, B3_LANE_REF_LANE( ref ), index );

		int lane = b3FindFreeLane( nodes + parent );
		if ( lane != B3_NULL_INDEX )
		{
			b3SetNodeLane( nodes, parent, lane, b );
			tree->parents[b] = B3_LANE_REF( parent, lane );
			if ( b < parent )
			{
				tree->dfsOrdered = false;
			}

			b3RefitAncestors( tree, parent );
			return;
		}

		// The parent is full. Its four lanes and the new node are the next five items.
		for ( int i = 0; i < 4; ++i )
		{
			items[i] = b3GetLaneItem( nodes + parent, i );
		}

		items[4].aabb = b3GetSubtreeAABB( nodes + b );
		items[4].child = b3MakeNodeLane( nodes, b );
		items[4].userData = 0;

		index = parent;
		leaf = false;
	}
}

// Greedy descent choosing the lane with the least area increase, R-tree style.
static int b3ChooseLane( const b3WideNode* node, b3AABB aabb )
{
	int bestLane = B3_NULL_INDEX;
	float bestCost = FLT_MAX;
	float bestArea = FLT_MAX;

	for ( int i = 0; i < 4; ++i )
	{
		if ( b3IsLaneEmpty( node->child[i] ) )
		{
			continue;
		}

		b3AABB box = b3GetLaneAABB( node, i );
		float area = b3Perimeter( box );
		float cost = b3Perimeter( b3AABB_Union( box, aabb ) ) - area;
		if ( cost < bestCost || ( cost == bestCost && area < bestArea ) )
		{
			bestLane = i;
			bestCost = cost;
			bestArea = area;
		}
	}

	B3_ASSERT( bestLane != B3_NULL_INDEX );
	return bestLane;
}

static void b3InsertLeaf( b3DynamicTree* tree, b3AABB aabb, int proxyId, bool moved )
{
	b3TreeProxy* proxy = tree->proxies + proxyId;
	b3LaneItem item = {
		.aabb = aabb,
		.child = (uint32_t)proxyId | B3_LEAF_NODE | ( moved ? B3_MOVED_NODE : 0 ),
		.userData = (uint32_t)proxy->userData,
	};

	// Descend to a leaf node
	b3WideNode* nodes = tree->nodes;
	int index = B3_ROOT_NODE;
	while ( b3IsLeaf( nodes + index ) == false )
	{
		int lane = b3ChooseLane( nodes + index, aabb );
		index = b3GetLaneIndex( nodes[index].child[lane] );
	}

	b3WideNode* node = nodes + index;
	int lane = b3FindFreeLane( node );
	if ( lane != B3_NULL_INDEX )
	{
		b3SetLaneAABB( node, lane, item.aabb );
		node->child[lane] = item.child;
		node->userData[lane] = item.userData;
		proxy->leaf = B3_LANE_REF( index, lane );
		b3RefitAncestors( tree, index );
		return;
	}

	b3LaneItem items[5];
	for ( int i = 0; i < 4; ++i )
	{
		items[i] = b3GetLaneItem( node, i );
	}
	items[4] = item;

	b3SplitNode( tree, index, items, true );
}

static void b3RemoveLeaf( b3DynamicTree* tree, int proxyId )
{
	b3WideNode* nodes = tree->nodes;
	int32_t* parents = tree->parents;

	int ref = tree->proxies[proxyId].leaf;
	B3_ASSERT( ref != B3_NULL_INDEX );
	int index = B3_LANE_REF_NODE( ref );
	int lane = B3_LANE_REF_LANE( ref );
	B3_ASSERT( 0 <= index && index < tree->nodeEnd );
	B3_ASSERT( b3IsLeaf( nodes + index ) && b3GetLaneIndex( nodes[index].child[lane] ) == proxyId );

	b3SetLaneEmpty( nodes + index, lane, true );

	// Nodes left empty are freed and their lanes cleared, up to the first node with something left.
	while ( index != B3_ROOT_NODE && b3GetLaneCount( nodes + index ) == 0 )
	{
		int parentRef = parents[index];
		int parent = B3_LANE_REF_NODE( parentRef );
		b3SetLaneEmpty( nodes + parent, B3_LANE_REF_LANE( parentRef ), false );
		b3FreeNode( tree, index );
		index = parent;
	}

	if ( index == B3_ROOT_NODE && b3GetLaneCount( nodes + index ) == 0 )
	{
		// The tree is empty
		nodes[B3_ROOT_NODE] = b3MakeEmptyNode( true );
		return;
	}

	// An internal node with one child left is spliced out so the child takes its lane.
	if ( index != B3_ROOT_NODE && b3IsLeaf( nodes + index ) == false && b3GetLaneCount( nodes + index ) == 1 )
	{
		int childLane = 0;
		while ( b3IsLaneEmpty( nodes[index].child[childLane] ) )
		{
			childLane += 1;
		}

		int child = b3GetLaneIndex( nodes[index].child[childLane] );
		int parentRef = parents[index];
		int parent = B3_LANE_REF_NODE( parentRef );
		int parentLane = B3_LANE_REF_LANE( parentRef );

		b3SetLaneAABB( nodes + parent, parentLane, b3GetLaneAABB( nodes + index, childLane ) );
		nodes[parent].child[parentLane] = nodes[index].child[childLane];
		parents[child] = parentRef;
		if ( child < parent )
		{
			tree->dfsOrdered = false;
		}

		b3FreeNode( tree, index );
		index = parent;
	}

	b3RefitAncestors( tree, index );

	// A root with one child collapses so the height stays low.
	while ( b3IsLeaf( nodes + B3_ROOT_NODE ) == false && b3GetLaneCount( nodes + B3_ROOT_NODE ) == 1 )
	{
		int childLane = 0;
		while ( b3IsLaneEmpty( nodes[B3_ROOT_NODE].child[childLane] ) )
		{
			childLane += 1;
		}

		int child = b3GetLaneIndex( nodes[B3_ROOT_NODE].child[childLane] );
		nodes[B3_ROOT_NODE] = nodes[child];

		b3WideNode* root = nodes + B3_ROOT_NODE;
		bool leaf = b3IsLeaf( root );
		for ( int i = 0; i < 4; ++i )
		{
			if ( b3IsLaneEmpty( root->child[i] ) )
			{
				continue;
			}

			int laneIndex = b3GetLaneIndex( root->child[i] );
			if ( leaf )
			{
				tree->proxies[laneIndex].leaf = B3_LANE_REF( B3_ROOT_NODE, i );
			}
			else
			{
				parents[laneIndex] = B3_LANE_REF( B3_ROOT_NODE, i );
			}
		}

		b3FreeNode( tree, child );
	}
}

// Create a proxy in the tree as a leaf lane. We return the proxy id instead of a pointer so that we
// can grow the node pool.
int b3CreateTreeProxyInternal( b3DynamicTree* tree, b3AABB aabb, uint64_t categoryBits, uint64_t userData, bool markMoved )
{
	B3_VALIDATE( b3IsValidAABB( aabb ) );

	int proxyId = b3AllocateProxy( tree );

	b3TreeProxy* proxy = tree->proxies + proxyId;
	proxy->categoryBits = categoryBits;
	proxy->userData = userData;

	b3InsertLeaf( tree, aabb, proxyId, markMoved );

	return proxyId;
}

int b3DynamicTree_CreateProxy( b3DynamicTree* tree, b3AABB aabb, uint64_t categoryBits, uint64_t userData )
{
	return b3CreateTreeProxyInternal( tree, aabb, categoryBits, userData, false );
}

void b3DynamicTree_DestroyProxy( b3DynamicTree* tree, int proxyId )
{
	B3_ASSERT( 0 <= proxyId && proxyId < tree->proxyCapacity );

	b3RemoveLeaf( tree, proxyId );
	b3FreeProxy( tree, proxyId );
}

int b3DynamicTree_GetProxyCount( const b3DynamicTree* tree )
{
	return tree->proxyCount;
}

void b3DynamicTree_MoveProxyInternal( b3DynamicTree* tree, int proxyId, b3AABB aabb, bool markMoved )
{
	B3_VALIDATE( b3IsValidAABB( aabb ) );
	B3_VALIDATE( aabb.upperBound.x - aabb.lowerBound.x < B3_HUGE );
	B3_VALIDATE( aabb.upperBound.y - aabb.lowerBound.y < B3_HUGE );
	B3_VALIDATE( aabb.upperBound.z - aabb.lowerBound.z < B3_HUGE );
	B3_ASSERT( 0 <= proxyId && proxyId < tree->proxyCapacity );

	b3RemoveLeaf( tree, proxyId );
	b3InsertLeaf( tree, aabb, proxyId, markMoved );
}

void b3DynamicTree_MoveProxy( b3DynamicTree* tree, int proxyId, b3AABB aabb )
{
	b3DynamicTree_MoveProxyInternal( tree, proxyId, aabb, false );
}

void b3DynamicTree_EnlargeProxy( b3DynamicTree* tree, int proxyId, b3AABB aabb )
{
	B3_VALIDATE( b3IsValidAABB( aabb ) );
	B3_VALIDATE( aabb.upperBound.x - aabb.lowerBound.x < B3_HUGE );
	B3_VALIDATE( aabb.upperBound.y - aabb.lowerBound.y < B3_HUGE );
	B3_VALIDATE( aabb.upperBound.z - aabb.lowerBound.z < B3_HUGE );
	B3_ASSERT( 0 <= proxyId && proxyId < tree->proxyCapacity );

	b3WideNode* nodes = tree->nodes;
	const int32_t* parents = tree->parents;

	int ref = tree->proxies[proxyId].leaf;
	int index = B3_LANE_REF_NODE( ref );
	int lane = B3_LANE_REF_LANE( ref );
	b3WideNode* node = nodes + index;
	B3_ASSERT( b3IsLeaf( node ) );

	// Caller must ensure this
	B3_VALIDATE( b3AABB_Contains( b3GetLaneAABB( node, lane ), aabb ) == false );

	b3SetLaneAABB( node, lane, aabb );
	node->child[lane] |= B3_MOVED_NODE;

	ref = parents[index];
	while ( ref != B3_NULL_INDEX )
	{
		index = B3_LANE_REF_NODE( ref );
		lane = B3_LANE_REF_LANE( ref );
		node = nodes + index;

		b3AABB laneBox = b3GetLaneAABB( node, lane );
		bool changed = b3EnlargeAABB( &laneBox, aabb );
		b3SetLaneAABB( node, lane, laneBox );

		// This is marked to ensure the root is marked in this loop or the one below.
		node->child[lane] |= B3_MOVED_NODE;

		ref = parents[index];

		if ( changed == false )
		{
			break;
		}
	}

	// Mark all the way up to the root.
	while ( ref != B3_NULL_INDEX )
	{
		index = B3_LANE_REF_NODE( ref );
		lane = B3_LANE_REF_LANE( ref );
		node = nodes + index;
		if ( node->child[lane] & B3_MOVED_NODE )
		{
			// Early out because this ancestor was previously ascended and marked as moved.
			break;
		}

		node->child[lane] |= B3_MOVED_NODE;
		ref = parents[index];
	}
}

void b3DynamicTree_SetCategoryBits( b3DynamicTree* tree, int proxyId, uint64_t categoryBits )
{
	B3_ASSERT( 0 <= proxyId && proxyId < tree->proxyCapacity );
	tree->proxies[proxyId].categoryBits = categoryBits;
}

uint64_t b3DynamicTree_GetCategoryBits( b3DynamicTree* tree, int proxyId )
{
	B3_ASSERT( 0 <= proxyId && proxyId < tree->proxyCapacity );
	return tree->proxies[proxyId].categoryBits;
}

// A leaf node has zero height. The tree is walked, there is no stored height.
int b3DynamicTree_GetHeight( const b3DynamicTree* tree )
{
	const b3WideNode* nodes = tree->nodes;

	struct
	{
		int index;
		int depth;
	} stack[B3_TREE_STACK_SIZE];

	int stackCount = 0;
	stack[stackCount].index = B3_ROOT_NODE;
	stack[stackCount].depth = 0;
	stackCount += 1;

	int height = 0;
	while ( stackCount > 0 )
	{
		stackCount -= 1;
		int index = stack[stackCount].index;
		int depth = stack[stackCount].depth;
		const b3WideNode* node = nodes + index;

		height = b3MaxInt( height, depth );

		if ( b3IsLeaf( node ) )
		{
			continue;
		}

		for ( int i = 0; i < 4; ++i )
		{
			if ( b3IsLaneEmpty( node->child[i] ) )
			{
				continue;
			}

			if ( stackCount < B3_TREE_STACK_SIZE )
			{
				stack[stackCount].index = b3GetLaneIndex( node->child[i] );
				stack[stackCount].depth = depth + 1;
				stackCount += 1;
			}
			else
			{
				B3_ASSERT( stackCount < B3_TREE_STACK_SIZE );
			}
		}
	}

	return height;
}

// The area ratio is the thing that SAH seeks to minimize. SAH cannot do anything about leaf
// boxes or the root box. It seeks to minimize the area of all the node boxes below the root,
// which are the lanes of the internal nodes. Divide this by the root area to make the metric
// non-dimensional. So this becomes a meaningful measure of tree quality.
float b3DynamicTree_GetAreaRatio( const b3DynamicTree* tree )
{
	if ( tree->proxyCount == 0 )
	{
		return 0.0f;
	}

	const b3WideNode* nodes = tree->nodes;
	float rootArea = b3Perimeter( b3GetSubtreeAABB( nodes + B3_ROOT_NODE ) );
	if ( rootArea <= 0.0f )
	{
		return 0.0f;
	}

	// Free nodes are leaf tagged.
	float internalArea = 0.0f;
	int nodeEnd = tree->nodeEnd;
	for ( int i = 0; i < nodeEnd; ++i )
	{
		const b3WideNode* node = nodes + i;
		if ( b3IsLeaf( node ) )
		{
			continue;
		}

		for ( int lane = 0; lane < 4; ++lane )
		{
			if ( b3IsLaneEmpty( node->child[lane] ) == false )
			{
				internalArea += b3Perimeter( b3GetLaneAABB( node, lane ) );
			}
		}
	}

	return internalArea / rootArea;
}

b3AABB b3DynamicTree_GetRootBounds( const b3DynamicTree* tree )
{
	if ( tree->proxyCount == 0 )
	{
		return B3_LITERAL( b3AABB ){ b3Vec3_zero, b3Vec3_zero };
	}

	return b3GetSubtreeAABB( tree->nodes + B3_ROOT_NODE );
}

#if B3_ENABLE_VALIDATION

static int b3ValidateSubtree( const b3DynamicTree* tree, int nodeIndex, int* leafCount, int* nodeCount )
{
	B3_ASSERT( 0 <= nodeIndex && nodeIndex < tree->nodeEnd );
	const b3WideNode* node = tree->nodes + nodeIndex;
	*nodeCount += 1;

	bool leaf = b3IsLeaf( node );
	int laneCount = 0;
	int height = 0;
	for ( int lane = 0; lane < 4; ++lane )
	{
		uint32_t child = node->child[lane];
		B3_ASSERT( ( ( child & B3_LEAF_NODE ) != 0 ) == leaf );

		if ( b3IsLaneEmpty( child ) )
		{
			B3_ASSERT( b3IsLaneMoved( child ) == false );
			B3_ASSERT( node->lowerX[lane] == INFINITY && node->upperX[lane] == -INFINITY );
			B3_ASSERT( node->lowerY[lane] == INFINITY && node->upperY[lane] == -INFINITY );
			B3_ASSERT( node->lowerZ[lane] == INFINITY && node->upperZ[lane] == -INFINITY );
			continue;
		}

		laneCount += 1;
		b3AABB laneBox = b3GetLaneAABB( node, lane );
		B3_ASSERT( b3IsValidAABB( laneBox ) );

		int laneIndex = b3GetLaneIndex( child );
		if ( leaf )
		{
			B3_ASSERT( 0 <= laneIndex && laneIndex < tree->proxyCapacity );
			B3_ASSERT( tree->proxies[laneIndex].leaf == B3_LANE_REF( nodeIndex, lane ) );
			B3_ASSERT( (uint32_t)tree->proxies[laneIndex].userData == node->userData[lane] );
			*leafCount += 1;
			continue;
		}

		B3_ASSERT( B3_ROOT_NODE < laneIndex && laneIndex < tree->nodeEnd );
		B3_ASSERT( tree->parents[laneIndex] == B3_LANE_REF( nodeIndex, lane ) );
		B3_ASSERT( tree->dfsOrdered == false || nodeIndex < laneIndex );

		const b3WideNode* childNode = tree->nodes + laneIndex;
		B3_ASSERT( b3AABB_Contains( laneBox, b3GetSubtreeAABB( childNode ) ) );
		B3_ASSERT( b3IsLaneMoved( child ) == b3HasMovedLane( childNode ) );

		// A bad tree can stack overflow, but that is validation on its own.
		int childHeight = b3ValidateSubtree( tree, laneIndex, leafCount, nodeCount );
		height = b3MaxInt( height, childHeight + 1 );
	}

	// Only the root may be an empty leaf. An internal node always has at least two children.
	B3_ASSERT( leaf ? ( laneCount > 0 || nodeIndex == B3_ROOT_NODE ) : laneCount >= 2 );
	return height;
}

#endif

void b3DynamicTree_Validate( const b3DynamicTree* tree )
{
#if B3_ENABLE_VALIDATION
	B3_ASSERT( 1 <= tree->nodeEnd && tree->nodeEnd <= tree->nodeCapacity );
	B3_ASSERT( tree->parents[B3_ROOT_NODE] == B3_NULL_INDEX );

	int freeNodeCount = 0;
	int freeIndex = tree->nodeFreeList;
	while ( freeIndex != B3_NULL_INDEX )
	{
		B3_ASSERT( B3_ROOT_NODE < freeIndex && freeIndex < tree->nodeEnd );
		B3_ASSERT( b3IsLeaf( tree->nodes + freeIndex ) && b3GetLaneCount( tree->nodes + freeIndex ) == 0 );
		freeIndex = tree->parents[freeIndex];
		++freeNodeCount;
		B3_ASSERT( freeNodeCount < tree->nodeEnd );
	}

	// Validate proxy free list.
	int freeProxyCount = 0;
	freeIndex = tree->proxyFreeList;
	while ( freeIndex != B3_NULL_INDEX )
	{
		B3_ASSERT( 0 <= freeIndex && freeIndex < tree->proxyCapacity );
		B3_ASSERT( tree->proxies[freeIndex].leaf == B3_NULL_INDEX );
		freeIndex = tree->proxies[freeIndex].next;
		++freeProxyCount;
	}
	B3_ASSERT( tree->proxyCount + freeProxyCount == tree->proxyCapacity );

	int leafCount = 0;
	int nodeCount = 0;
	b3ValidateSubtree( tree, B3_ROOT_NODE, &leafCount, &nodeCount );
	B3_ASSERT( leafCount == tree->proxyCount );
	B3_ASSERT( nodeCount + freeNodeCount == tree->nodeEnd );

#else
	B3_UNUSED( tree );
#endif
}

void b3DynamicTree_ValidateNoMoved( const b3DynamicTree* tree )
{
#if B3_ENABLE_VALIDATION == 1
	const b3WideNode* nodes = tree->nodes;
	int nodeEnd = tree->nodeEnd;
	for ( int i = 0; i < nodeEnd; ++i )
	{
		B3_ASSERT( b3HasMovedLane( nodes + i ) == false );
	}
#else
	B3_UNUSED( tree );
#endif
}

int b3DynamicTree_GetByteCount( const b3DynamicTree* tree )
{
	size_t size = sizeof( b3DynamicTree );
	size += tree->nodeCapacity * sizeof( b3WideNode );
	size += tree->nodeCapacity * sizeof( int32_t );
	size += tree->proxyCapacity * sizeof( b3TreeProxy );
	size += tree->swapNodes == NULL ? 0 : tree->nodeCapacity * sizeof( b3WideNode );

	// leafIndices, leafRefs, builtNodes
	size += 3 * tree->rebuildCapacity * sizeof( int32_t );
	// leafBoxes
	size += tree->rebuildCapacity * sizeof( b3AABB );
	// leafCenters
	size += tree->rebuildCapacity * sizeof( b3Vec3 );

	return (int)size;
}

B3_FORCE_INLINE bool b3TestCategory( uint64_t categoryBits, uint64_t maskBits, bool requireAllBits )
{
	return requireAllBits ? ( categoryBits & maskBits ) == maskBits : ( categoryBits & maskBits ) != 0;
}

B3_FORCE_INLINE b3TreeStats b3QueryTreeView( const b3TreeView* view, b3AABB aabb, uint64_t maskBits, bool requireAllBits,
											 b3TreeQueryCallbackFcn* callback, void* context, bool hasProxies )
{
	b3TreeStats result = { 0 };

	const b3WideNode* nodes = view->nodes;

	uint32_t stack[B3_TREE_STACK_SIZE];
	int stackCount = 0;
	stack[stackCount++] = B3_ROOT_NODE;

	b3BoxW4 query = b3SplatBoxW4( aabb );

	while ( stackCount > 0 )
	{
		const b3WideNode* node = nodes + b3PopNodeIndex( stack[--stackCount] );
		result.nodeVisits += 1;

		b3BoxW4 boxes = b3LoadBoxW4( node );
		int mask = b3OverlapMaskW4( &query, &boxes );

		if ( b3IsLeaf( node ) )
		{
			// Left-pack the hit lanes so the callback loop is a plain counted loop
			uint32_t children[4];
			uint32_t userDatas[4];
			int hitCount = b3CompressLanesW4( children, node->child, mask );
			b3CompressLanesW4( userDatas, node->userData, mask );

			for ( int i = 0; i < hitCount; ++i )
			{
				int proxyId = b3GetLaneIndex( children[i] );
				uint64_t userData = userDatas[i];
				if ( hasProxies )
				{
					const b3TreeProxy* proxy = view->proxies + proxyId;
					if ( b3TestCategory( proxy->categoryBits, maskBits, requireAllBits ) == false )
					{
						continue;
					}
					userData = proxy->userData;
				}

				bool proceed = callback( proxyId, userData, context );
				result.leafVisits += 1;

				if ( proceed == false )
				{
					return result;
				}
			}
		}
		else
		{
			if ( stackCount > B3_TREE_STACK_SIZE - 4 )
			{
				B3_ASSERT( stackCount <= B3_TREE_STACK_SIZE - 4 );
				continue;
			}

			stackCount = b3PushLanesW4( stack, stackCount, node, mask );
		}
	}

	return result;
}

typedef struct b3TreeHit
{
	int lane;
	float value;
} b3TreeHit;

// Branch free compare and exchange of two unsigned keys
#define B3_SORT_KEYS( A, B )                                                                                                     \
	{                                                                                                                            \
		uint32_t lo = ( A ) < ( B ) ? ( A ) : ( B );                                                                             \
		uint32_t hi = ( A ) < ( B ) ? ( B ) : ( A );                                                                             \
		( A ) = lo;                                                                                                              \
		( B ) = hi;                                                                                                              \
	}

// Gather the lanes in a mask ordered by ascending value, branch free. The values must be
// non-negative so their bit patterns order as unsigned integers. The lane index rides in the
// two low mantissa bits through a five exchange sorting network and the misses sort last. All
// four hits are written, the count says how many are real. The values are read back from the
// unmodified lanes so the hit values are exact.
#if defined( B3_SIMD_NEON )

// NEON version of the sort below. The keys never leave the vector registers: the lane index
// is embedded, the misses are forced to all ones from the vector mask, the five exchange
// network runs on permuted copies with unsigned min and max, and the sorted values are
// gathered back from the unmodified lanes with a byte shuffle so they stay exact.
B3_FORCE_INLINE int b3SortHits( b3FloatW4 mask, b3FloatW4 values, b3TreeHit* hits )
{
	static const uint32_t laneIndex[4] = { 0, 1, 2, 3 };
	static const uint8_t swapMiddle[16] = { 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15 };
	static const uint32_t selectLane2[4] = { 0, 0, 0xFFFFFFFFu, 0 };

	uint32x4_t hit = vreinterpretq_u32_f32( mask );
	uint32x4_t keys = vorrq_u32( vbicq_u32( vreinterpretq_u32_f32( values ), vdupq_n_u32( 3 ) ), vld1q_u32( laneIndex ) );
	keys = vornq_u32( keys, hit );

	// The nearest lane is all the descent needs, and a horizontal min gets it in one step
	// instead of waiting for the three stages of the network
	uint32_t nearest = vminvq_u32( keys );

	// (0,1) (2,3)
	uint32x4_t partner = vrev64q_u32( keys );
	uint32x4_t lo = vminq_u32( keys, partner );
	uint32x4_t hi = vmaxq_u32( keys, partner );
	keys = vtrn1q_u32( lo, hi );

	// (0,2) (1,3)
	partner = vextq_u32( keys, keys, 2 );
	lo = vminq_u32( keys, partner );
	hi = vmaxq_u32( keys, partner );
	keys = vextq_u32( lo, hi, 2 );

	// (1,2)
	partner = vreinterpretq_u32_u8( vqtbl1q_u8( vreinterpretq_u8_u32( keys ), vld1q_u8( swapMiddle ) ) );
	lo = vminq_u32( keys, partner );
	hi = vmaxq_u32( keys, partner );
	keys = vbslq_u32( vld1q_u32( selectLane2 ), hi, lo );

	// Gather the exact values by sorted lane
	uint32x4_t lanes = vandq_u32( keys, vdupq_n_u32( 3 ) );
	uint32x4_t control =
		vaddq_u32( vmulq_u32( vshlq_n_u32( lanes, 2 ), vdupq_n_u32( 0x01010101u ) ), vdupq_n_u32( 0x03020100u ) );
	uint32x4_t sorted = vreinterpretq_u32_u8( vqtbl1q_u8( vreinterpretq_u8_f32( values ), vreinterpretq_u8_u32( control ) ) );

	uint32_t* out = (uint32_t*)hits;
	vst1q_u32( out, vzip1q_u32( lanes, sorted ) );
	vst1q_u32( out + 4, vzip2q_u32( lanes, sorted ) );

	hits[0].lane = (int)( nearest & 3 );

	return (int)vaddvq_u32( vshrq_n_u32( hit, 31 ) );
}

#else

B3_FORCE_INLINE int b3SortHits( b3FloatW4 maskW, b3FloatW4 values, b3TreeHit* hits )
{
	int mask = b3MoveMaskW4( maskW );
	float lanes[4];
	b3StoreW4( lanes, values );

	float keyed[4];
	b3StoreW4( keyed, b3EmbedIndexW4( values, 0, 2 ) );

	uint32_t keys[4];
	memcpy( keys, keyed, sizeof( keys ) );

	// A clear mask bit sets every key bit, so the lane sorts behind every hit
	keys[0] |= 0u - (uint32_t)( ( mask & 1 ) ^ 1 );
	keys[1] |= 0u - (uint32_t)( ( ( mask >> 1 ) & 1 ) ^ 1 );
	keys[2] |= 0u - (uint32_t)( ( ( mask >> 2 ) & 1 ) ^ 1 );
	keys[3] |= 0u - (uint32_t)( ( ( mask >> 3 ) & 1 ) ^ 1 );

	B3_SORT_KEYS( keys[0], keys[1] );
	B3_SORT_KEYS( keys[2], keys[3] );
	B3_SORT_KEYS( keys[0], keys[2] );
	B3_SORT_KEYS( keys[1], keys[3] );
	B3_SORT_KEYS( keys[1], keys[2] );

	for ( int i = 0; i < 4; ++i )
	{
		int lane = (int)( keys[i] & 3 );
		hits[i].lane = lane;
		hits[i].value = lanes[lane];
	}

	return b3LaneCountW4( mask );
}

#endif

typedef struct b3QueryClosestItem
{
	int nodeIndex;
	float distanceToNodeSqr;
} b3QueryClosestItem;

B3_FORCE_INLINE b3TreeStats b3QueryClosestTreeView( const b3TreeView* view, b3Vec3 point, uint64_t maskBits, bool requireAllBits,
													b3TreeQueryClosestCallbackFcn* callback, void* context, float* minDistanceSqr,
													bool hasProxies )
{
	b3TreeStats result = { 0 };

	const b3WideNode* nodes = view->nodes;

	float minSqr = *minDistanceSqr;
	b3QueryClosestItem stack[B3_TREE_STACK_SIZE];
	int stackCount = 0;
	stack[stackCount++] = B3_LITERAL( b3QueryClosestItem ){ B3_ROOT_NODE, 0.0f };

	b3FloatW4 px = b3SplatW4( point.x );
	b3FloatW4 py = b3SplatW4( point.y );
	b3FloatW4 pz = b3SplatW4( point.z );

	while ( stackCount > 0 )
	{
		b3QueryClosestItem item = stack[--stackCount];
		if ( item.distanceToNodeSqr >= minSqr )
		{
			continue;
		}

		const b3WideNode* node = nodes + item.nodeIndex;
		result.nodeVisits += 1;

		b3BoxW4 boxes = b3LoadBoxW4( node );
		b3FloatW4 distances = b3DistanceSqrW4( px, py, pz, &boxes );
		b3FloatW4 mask = b3LessThanW4( distances, b3SplatW4( minSqr ) );

		// Closest first so the minimum narrows quickly
		b3TreeHit hits[4];
		int hitCount = b3SortHits( mask, distances, hits );

		if ( b3IsLeaf( node ) )
		{
			for ( int i = 0; i < hitCount; ++i )
			{
				if ( hits[i].value >= minSqr )
				{
					break;
				}

				int lane = hits[i].lane;
				int proxyId = b3GetLaneIndex( node->child[lane] );
				uint64_t userData = node->userData[lane];
				if ( hasProxies )
				{
					const b3TreeProxy* proxy = view->proxies + proxyId;
					if ( b3TestCategory( proxy->categoryBits, maskBits, requireAllBits ) == false )
					{
						continue;
					}
					userData = proxy->userData;
				}

				// callback to user code with minimum distance squared so far and proxy id
				float dd = callback( minSqr, proxyId, userData, context );

				if ( dd < minSqr )
				{
					minSqr = dd;
				}

				result.leafVisits += 1;
			}
		}
		else
		{
			if ( stackCount + hitCount > B3_TREE_STACK_SIZE )
			{
				B3_ASSERT( stackCount + hitCount <= B3_TREE_STACK_SIZE );
				continue;
			}

			// Push the farthest first so the closest pops first
			for ( int i = hitCount - 1; i >= 0; --i )
			{
				stack[stackCount++] =
					B3_LITERAL( b3QueryClosestItem ){ b3GetLaneIndex( node->child[hits[i].lane] ), hits[i].value };
			}
		}
	}

	*minDistanceSqr = minSqr;

	return result;
}

typedef struct b3CastItem
{
	int nodeIndex;
	float fraction;
} b3CastItem;

// Four lane slab test. The survivors are ordered so the closer one narrows first.
B3_FORCE_INLINE b3TreeStats b3RayCastTreeView( const b3TreeView* view, const b3RayCastInput* input, uint64_t maskBits,
											   bool requireAllBits, b3TreeRayCastCallbackFcn* callback, void* context,
											   bool hasProxies )
{
	b3TreeStats result = { 0 };

	b3Vec3 origin = input->origin;
	b3Vec3 invDirection = b3InverseDirection( input->translation );
	float maxFraction = input->maxFraction;

	const b3WideNode* nodes = view->nodes;

	b3CastItem stack[B3_TREE_STACK_SIZE];
	int stackCount = 0;
	stack[stackCount++] = B3_LITERAL( b3CastItem ){ B3_ROOT_NODE, 0.0f };

	b3RayCastInput subInput = *input;

	while ( stackCount > 0 )
	{
		b3CastItem item = stack[--stackCount];
		if ( item.fraction > maxFraction )
		{
			continue;
		}

		// Descend straight into the nearest child at each level. Only the farther children go
		// through the stack, so the next node load does not wait on a store and a pop.
		int nodeIndex = item.nodeIndex;
		for ( ;; )
		{
			const b3WideNode* node = nodes + nodeIndex;
			result.nodeVisits += 1;

			b3BoxW4 boxes = b3LoadBoxW4( node );
			b3FloatW4 fractions;
			b3FloatW4 mask = b3RayMaskW4( origin, invDirection, maxFraction, &boxes, &fractions );

			b3TreeHit hits[4];
			int hitCount = b3SortHits( mask, fractions, hits );

			if ( b3IsLeaf( node ) )
			{
				for ( int i = 0; i < hitCount; ++i )
				{
					if ( hits[i].value > maxFraction )
					{
						break;
					}

					int lane = hits[i].lane;
					int proxyId = b3GetLaneIndex( node->child[lane] );
					uint64_t userData = node->userData[lane];
					if ( hasProxies )
					{
						const b3TreeProxy* proxy = view->proxies + proxyId;
						if ( b3TestCategory( proxy->categoryBits, maskBits, requireAllBits ) == false )
						{
							continue;
						}
						userData = proxy->userData;
					}

					subInput.maxFraction = maxFraction;

					float value = callback( &subInput, proxyId, userData, context );
					result.leafVisits += 1;

					// The user may return -1 to indicate this shape should be skipped

					if ( value == 0.0f )
					{
						// The client has terminated the ray cast.
						return result;
					}

					if ( 0.0f < value && value <= maxFraction )
					{
						maxFraction = value;
					}
				}

				break;
			}

			if ( hitCount == 0 )
			{
				break;
			}

			if ( stackCount + hitCount - 1 > B3_TREE_STACK_SIZE )
			{
				B3_ASSERT( stackCount + hitCount - 1 <= B3_TREE_STACK_SIZE );
				break;
			}

			// Push the farther children, farthest first. The pushed nodes are prefetched so the
			// sibling loads overlap the test of the nearest.
			for ( int i = hitCount - 1; i >= 1; --i )
			{
				int childIndex = b3GetLaneIndex( node->child[hits[i].lane] );
				b3PrefetchNode( nodes + childIndex );
				stack[stackCount++] = B3_LITERAL( b3CastItem ){ childIndex, hits[i].value };
			}

			nodeIndex = b3GetLaneIndex( node->child[hits[0].lane] );
		}
	}

	return result;
}

// Follows structure of ray cast with small tweaks to handle a swept box. The lane boxes are
// grown by the half extents and the center is cast as a ray.
B3_FORCE_INLINE b3TreeStats b3BoxCastTreeView( const b3TreeView* view, const b3BoxCastInput* input, uint64_t maskBits,
											   bool requireAllBits, b3TreeBoxCastCallbackFcn* callback, void* context,
											   bool hasProxies )
{
	b3TreeStats result = { 0 };

	// The caller folds the shape radius and the world origin into the box. The ray starts at
	// the lower corner of the box, so only the lane lower bounds need to grow, by the full box
	// size. Same hits as casting the center against boxes grown by the half extents on both
	// sides, three fewer ops per node.
	b3AABB originAABB = input->box;

	b3Vec3 origin = originAABB.lowerBound;
	b3Vec3 size = b3Sub( originAABB.upperBound, originAABB.lowerBound );
	b3Vec3 invDirection = b3InverseDirection( input->translation );
	float maxFraction = input->maxFraction;

	b3FloatW4 sx = b3SplatW4( size.x );
	b3FloatW4 sy = b3SplatW4( size.y );
	b3FloatW4 sz = b3SplatW4( size.z );

	const b3WideNode* nodes = view->nodes;

	b3CastItem stack[B3_TREE_STACK_SIZE];
	int stackCount = 0;
	stack[stackCount++] = B3_LITERAL( b3CastItem ){ B3_ROOT_NODE, 0.0f };

	b3BoxCastInput subInput = *input;

	while ( stackCount > 0 )
	{
		b3CastItem item = stack[--stackCount];
		if ( item.fraction > maxFraction )
		{
			continue;
		}

		const b3WideNode* node = nodes + item.nodeIndex;
		result.nodeVisits += 1;

		// Empty lanes stay inverted because infinity absorbs the extension
		b3BoxW4 boxes = b3LoadBoxW4( node );
		boxes.lowerX = b3SubW4( boxes.lowerX, sx );
		boxes.lowerY = b3SubW4( boxes.lowerY, sy );
		boxes.lowerZ = b3SubW4( boxes.lowerZ, sz );

		b3FloatW4 fractions;
		b3FloatW4 mask = b3RayMaskW4( origin, invDirection, maxFraction, &boxes, &fractions );

		b3TreeHit hits[4];
		int hitCount = b3SortHits( mask, fractions, hits );

		if ( b3IsLeaf( node ) )
		{
			for ( int i = 0; i < hitCount; ++i )
			{
				if ( hits[i].value > maxFraction )
				{
					break;
				}

				int lane = hits[i].lane;
				int proxyId = b3GetLaneIndex( node->child[lane] );
				uint64_t userData = node->userData[lane];
				if ( hasProxies )
				{
					const b3TreeProxy* proxy = view->proxies + proxyId;
					if ( b3TestCategory( proxy->categoryBits, maskBits, requireAllBits ) == false )
					{
						continue;
					}
					userData = proxy->userData;
				}

				subInput.maxFraction = maxFraction;

				float value = callback( &subInput, proxyId, userData, context );
				result.leafVisits += 1;

				if ( value == 0.0f )
				{
					// The client has terminated the cast.
					return result;
				}

				if ( 0.0f < value && value < maxFraction )
				{
					maxFraction = value;
				}
			}
		}
		else
		{
			if ( stackCount + hitCount > B3_TREE_STACK_SIZE )
			{
				B3_ASSERT( stackCount + hitCount <= B3_TREE_STACK_SIZE );
				continue;
			}

			for ( int i = hitCount - 1; i >= 0; --i )
			{
				stack[stackCount++] = B3_LITERAL( b3CastItem ){ b3GetLaneIndex( node->child[hits[i].lane] ), hits[i].value };
			}
		}
	}

	return result;
}

b3TreeStats b3DynamicTree_Query( const b3DynamicTree* tree, b3AABB aabb, uint64_t maskBits, bool requireAllBits,
								 b3TreeQueryCallbackFcn* callback, void* context )
{
	if ( tree->proxyCount == 0 )
	{
		return B3_LITERAL( b3TreeStats ){ 0 };
	}

	b3TreeView view = b3MakeTreeView( tree );
	return b3QueryTreeView( &view, aabb, maskBits, requireAllBits, callback, context, true );
}

b3TreeStats b3DynamicTree_QueryClosest( const b3DynamicTree* tree, b3Vec3 point, uint64_t maskBits, bool requireAllBits,
										b3TreeQueryClosestCallbackFcn* callback, void* context, float* minDistanceSqr )
{
	if ( tree->proxyCount == 0 )
	{
		return B3_LITERAL( b3TreeStats ){ 0 };
	}

	b3TreeView view = b3MakeTreeView( tree );
	return b3QueryClosestTreeView( &view, point, maskBits, requireAllBits, callback, context, minDistanceSqr, true );
}

b3TreeStats b3DynamicTree_RayCast( const b3DynamicTree* tree, const b3RayCastInput* input, uint64_t maskBits, bool requireAllBits,
								   b3TreeRayCastCallbackFcn* callback, void* context )
{
	if ( tree->proxyCount == 0 )
	{
		return B3_LITERAL( b3TreeStats ){ 0 };
	}

	b3TreeView view = b3MakeTreeView( tree );
	return b3RayCastTreeView( &view, input, maskBits, requireAllBits, callback, context, true );
}

b3TreeStats b3DynamicTree_BoxCast( const b3DynamicTree* tree, const b3BoxCastInput* input, uint64_t maskBits, bool requireAllBits,
								   b3TreeBoxCastCallbackFcn* callback, void* context )
{
	if ( tree->proxyCount == 0 )
	{
		return B3_LITERAL( b3TreeStats ){ 0 };
	}

	b3TreeView view = b3MakeTreeView( tree );
	return b3BoxCastTreeView( &view, input, maskBits, requireAllBits, callback, context, true );
}

b3TreeStats b3TreeView_Query( const b3TreeView* view, b3AABB aabb, uint64_t maskBits, bool requireAllBits,
							  b3TreeQueryCallbackFcn* callback, void* context )
{
	if ( view->proxies == NULL )
	{
		return b3QueryTreeView( view, aabb, maskBits, requireAllBits, callback, context, false );
	}

	return b3QueryTreeView( view, aabb, maskBits, requireAllBits, callback, context, true );
}

b3TreeStats b3TreeView_QueryClosest( const b3TreeView* view, b3Vec3 point, uint64_t maskBits, bool requireAllBits,
									 b3TreeQueryClosestCallbackFcn* callback, void* context, float* minDistanceSqr )
{
	if ( view->proxies == NULL )
	{
		return b3QueryClosestTreeView( view, point, maskBits, requireAllBits, callback, context, minDistanceSqr, false );
	}

	return b3QueryClosestTreeView( view, point, maskBits, requireAllBits, callback, context, minDistanceSqr, true );
}

b3TreeStats b3TreeView_RayCast( const b3TreeView* view, const b3RayCastInput* input, uint64_t maskBits, bool requireAllBits,
								b3TreeRayCastCallbackFcn* callback, void* context )
{
	if ( view->proxies == NULL )
	{
		return b3RayCastTreeView( view, input, maskBits, requireAllBits, callback, context, false );
	}

	return b3RayCastTreeView( view, input, maskBits, requireAllBits, callback, context, true );
}

b3TreeStats b3TreeView_BoxCast( const b3TreeView* view, const b3BoxCastInput* input, uint64_t maskBits, bool requireAllBits,
								b3TreeBoxCastCallbackFcn* callback, void* context )
{
	if ( view->proxies == NULL )
	{
		return b3BoxCastTreeView( view, input, maskBits, requireAllBits, callback, context, false );
	}

	return b3BoxCastTreeView( view, input, maskBits, requireAllBits, callback, context, true );
}

// Bounds of a set of leaf centers. Min and max are exact, so the result does not depend on
// the visiting order and matches bounds accumulated during a partition bit for bit.
typedef struct b3CenterBounds
{
	b3Vec3 lower, upper;
} b3CenterBounds;

static b3CenterBounds b3ComputeCenterBounds( const b3Vec3* centers, int count )
{
	B3_ASSERT( count > 0 );
	b3CenterBounds bounds = { centers[0], centers[0] };
	for ( int i = 1; i < count; ++i )
	{
		bounds.lower = b3Min( bounds.lower, centers[i] );
		bounds.upper = b3Max( bounds.upper, centers[i] );
	}
	return bounds;
}

// Longest axis of the center bounds
static inline int b3LongestAxis( b3CenterBounds bounds )
{
	b3Vec3 d = b3Sub( bounds.upper, bounds.lower );
	return ( d.x >= d.y && d.x >= d.z ) ? 0 : ( d.y >= d.z ? 1 : 2 );
}

// Median split heuristic. Partitions the indices and centers in place on the longest axis of
// the given center bounds and returns the left count, which is strictly inside the range.
// The center bounds of both sides are accumulated while partitioning, so a split is one pass
// over the range rather than a bounds scan followed by the partition.
static int b3PartitionMid( int* indices, b3Vec3* centers, int count, b3CenterBounds bounds, b3CenterBounds* boundsA,
						   b3CenterBounds* boundsB )
{
	B3_ASSERT( count >= 2 );

	// Handle trivial case
	if ( count == 2 )
	{
		*boundsA = B3_LITERAL( b3CenterBounds ){ centers[0], centers[0] };
		*boundsB = B3_LITERAL( b3CenterBounds ){ centers[1], centers[1] };
		return 1;
	}

	int axis = b3LongestAxis( bounds );
	b3Vec3 c = b3MulSV( 0.5f, b3Add( bounds.lower, bounds.upper ) );
	float pivot = axis == 0 ? c.x : ( axis == 1 ? c.y : c.z );

	b3Vec3 lowerA = { FLT_MAX, FLT_MAX, FLT_MAX }, upperA = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	b3Vec3 lowerB = { FLT_MAX, FLT_MAX, FLT_MAX }, upperB = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

	// Partition longest axis using the Hoare partition scheme
	// https://en.wikipedia.org/wiki/Quicksort
	// https://nicholasvadivelu.com/2021/01/11/array-partition/
	const float* keys = &centers[0].x;
	int i1 = 0, i2 = count;
	while ( i1 < i2 )
	{
		while ( i1 < i2 && keys[3 * i1 + axis] < pivot )
		{
			lowerA = b3Min( lowerA, centers[i1] );
			upperA = b3Max( upperA, centers[i1] );
			i1 += 1;
		};

		while ( i1 < i2 && keys[3 * ( i2 - 1 ) + axis] >= pivot )
		{
			lowerB = b3Min( lowerB, centers[i2 - 1] );
			upperB = b3Max( upperB, centers[i2 - 1] );
			i2 -= 1;
		};

		if ( i1 < i2 )
		{
			// The item at i1 moves to the right side and the item at i2 - 1 to the left side
			lowerB = b3Min( lowerB, centers[i1] );
			upperB = b3Max( upperB, centers[i1] );
			lowerA = b3Min( lowerA, centers[i2 - 1] );
			upperA = b3Max( upperA, centers[i2 - 1] );

			// Swap indices
			{
				int temp = indices[i1];
				indices[i1] = indices[i2 - 1];
				indices[i2 - 1] = temp;
			}

			// Swap centers
			{
				b3Vec3 temp = centers[i1];
				centers[i1] = centers[i2 - 1];
				centers[i2 - 1] = temp;
			}

			i1 += 1;
			i2 -= 1;
		}
	}
	B3_ASSERT( i1 == i2 );

	if ( i1 > 0 && i1 < count )
	{
		*boundsA = B3_LITERAL( b3CenterBounds ){ lowerA, upperA };
		*boundsB = B3_LITERAL( b3CenterBounds ){ lowerB, upperB };
		return i1;
	}

	// Every center fell on one side of the pivot. Split in half and scan the halves, rare.
	int split = count / 2;
	*boundsA = b3ComputeCenterBounds( centers, split );
	*boundsB = b3ComputeCenterBounds( centers + split, count - split );
	return split;
}

// Insertion sort of a range along the longest axis of the center bounds. Ranges are small.
static void b3SortRange( int* indices, b3Vec3* centers, int count, b3CenterBounds bounds )
{
	int axis = b3LongestAxis( bounds );
	float* keys = &centers[0].x;

	for ( int i = 1; i < count; ++i )
	{
		int index = indices[i];
		b3Vec3 center = centers[i];
		float key = keys[3 * i + axis];
		int j = i;
		while ( j > 0 && key < keys[3 * ( j - 1 ) + axis] )
		{
			indices[j] = indices[j - 1];
			centers[j] = centers[j - 1];
			j -= 1;
		}
		indices[j] = index;
		centers[j] = center;
	}
}

// A range of build leaves to be placed below a node of the new array
typedef struct b3BuildRange
{
	int start;
	int count;
	int nodeIndex;

	// Bounds of the leaf centers in the range. Only read when the range is split or sorted.
	b3CenterBounds bounds;
} b3BuildRange;

// Bump allocate a node in the new array. The new array grows on its own, the old array is
// still being read and keeps its capacity until the swap. The parents are rebuilt from scratch.
static int b3BumpNode( b3DynamicTree* tree, bool leaf )
{
	int index = tree->nodeEnd;
	if ( index == tree->nodeCapacity )
	{
		int oldCapacity = tree->nodeCapacity;
		int newCapacity = oldCapacity + ( oldCapacity >> 1 );
		tree->swapNodes = B3_GROW_ZERO( tree->swapNodes, oldCapacity, newCapacity );
		tree->parents = B3_GROW_ZERO( tree->parents, oldCapacity, newCapacity );
		tree->nodeCapacity = newCapacity;
	}

	tree->nodeEnd += 1;
	tree->swapNodes[index] = b3MakeEmptyNode( leaf );
	tree->parents[index] = B3_NULL_INDEX;
	return index;
}

typedef struct b3CopyItem
{
	int oldIndex;
	int newIndex;
} b3CopyItem;

// Copy a retained subtree from the old array into the rebuilt array. Children are bump
// allocated as their parent is copied, so they land above it.
static void b3CopySubtree( b3DynamicTree* tree, int oldIndex, int newIndex )
{
	const b3WideNode* oldNodes = tree->nodes;
	b3TreeProxy* proxies = tree->proxies;

	b3CopyItem stack[B3_TREE_STACK_SIZE];
	int stackCount = 0;
	stack[stackCount++] = B3_LITERAL( b3CopyItem ){ oldIndex, newIndex };

	while ( stackCount > 0 )
	{
		b3CopyItem item = stack[--stackCount];
		const b3WideNode* src = oldNodes + item.oldIndex;
		tree->swapNodes[item.newIndex] = *src;

		bool leaf = b3IsLeaf( src );
		for ( int lane = 0; lane < 4; ++lane )
		{
			if ( b3IsLaneEmpty( src->child[lane] ) )
			{
				continue;
			}

			int ref = B3_LANE_REF( item.newIndex, lane );
			if ( leaf )
			{
				proxies[b3GetLaneIndex( src->child[lane] )].leaf = ref;
				continue;
			}

			// The bump can move the new array
			int child = b3BumpNode( tree, false );
			tree->swapNodes[item.newIndex].child[lane] = ( src->child[lane] & ~B3_NODE_INDEX_MASK ) | (uint32_t)child;
			tree->parents[child] = ref;

			if ( stackCount < B3_TREE_STACK_SIZE )
			{
				stack[stackCount++] = B3_LITERAL( b3CopyItem ){ b3GetLaneIndex( src->child[lane] ), child };
			}
			else
			{
				// todo fail gracefully if the stack runs out in release
				B3_ASSERT( stackCount < B3_TREE_STACK_SIZE );
			}
		}
	}
}

static inline bool b3IsProxyLeaf( const b3DynamicTree* tree, int leafIndex )
{
	return ( tree->leafRefs[leafIndex] & B3_LEAF_NODE ) != 0;
}

// Fill a leaf node of the new array with proxy build leaves
static void b3FillLeafNode( b3DynamicTree* tree, int nodeIndex, const int* leafIndices, int count )
{
	B3_ASSERT( 0 < count && count <= 4 );
	b3WideNode* node = tree->swapNodes + nodeIndex;
	*node = b3MakeEmptyNode( true );

	for ( int i = 0; i < count; ++i )
	{
		int leafIndex = leafIndices[i];
		uint32_t ref = (uint32_t)tree->leafRefs[leafIndex];
		B3_ASSERT( ( ref & B3_LEAF_NODE ) != 0 );
		int proxyId = b3GetLaneIndex( ref );

		b3SetLaneAABB( node, i, tree->leafBoxes[leafIndex] );
		node->child[i] = ref;
		node->userData[i] = (uint32_t)tree->proxies[proxyId].userData;
		tree->proxies[proxyId].leaf = B3_LANE_REF( nodeIndex, i );
	}
}

// Write a lane of an internal node that holds a retained subtree root. The subtree is copied now.
// The lane box is filled by the bottom up pass over the built nodes.
static void b3PlaceRetainedNode( b3DynamicTree* tree, int nodeIndex, int lane, int leafIndex )
{
	int oldIndex = b3GetLaneIndex( (uint32_t)tree->leafRefs[leafIndex] );
	int newIndex = b3BumpNode( tree, false );
	b3CopySubtree( tree, oldIndex, newIndex );

	tree->swapNodes[nodeIndex].child[lane] = (uint32_t)newIndex;
	tree->parents[newIndex] = B3_LANE_REF( nodeIndex, lane );
}

// Fill a node of the new array with at most four build leaves. All proxies make a leaf node.
// Otherwise the retained subtrees get their own lanes and the proxies share one new leaf node.
static void b3FillSmallGroup( b3DynamicTree* tree, b3BuildRange range, int* builtCount )
{
	B3_ASSERT( 0 < range.count && range.count <= 4 );
	const int* leafIndices = tree->leafIndices + range.start;

	int proxyIndices[4];
	int proxyCount = 0;
	int nodeIndices[4];
	int nodeCount = 0;
	for ( int i = 0; i < range.count; ++i )
	{
		if ( b3IsProxyLeaf( tree, leafIndices[i] ) )
		{
			proxyIndices[proxyCount++] = leafIndices[i];
		}
		else
		{
			nodeIndices[nodeCount++] = leafIndices[i];
		}
	}

	if ( nodeCount == 0 )
	{
		b3FillLeafNode( tree, range.nodeIndex, proxyIndices, proxyCount );
		return;
	}

	if ( nodeCount == 1 && proxyCount == 0 )
	{
		// A lone retained subtree is copied straight over this node
		int oldIndex = b3GetLaneIndex( (uint32_t)tree->leafRefs[nodeIndices[0]] );
		b3CopySubtree( tree, oldIndex, range.nodeIndex );
		return;
	}

	tree->swapNodes[range.nodeIndex] = b3MakeEmptyNode( false );
	tree->builtNodes[*builtCount] = range.nodeIndex;
	*builtCount += 1;

	int lane = 0;
	for ( int i = 0; i < nodeCount; ++i )
	{
		b3PlaceRetainedNode( tree, range.nodeIndex, lane, nodeIndices[i] );
		lane += 1;
	}

	if ( proxyCount > 0 )
	{
		int child = b3BumpNode( tree, true );
		b3FillLeafNode( tree, child, proxyIndices, proxyCount );
		tree->swapNodes[range.nodeIndex].child[lane] = (uint32_t)child;
		tree->parents[child] = B3_LANE_REF( range.nodeIndex, lane );
	}
}

// Give an internal node one child per sub-range. A sub-range holding a single retained subtree
// is placed directly, everything else gets a fresh node and goes on the stack.
static void b3CreateChildren( b3DynamicTree* tree, int nodeIndex, const b3BuildRange* subRanges, int subRangeCount,
							  b3BuildRange* stack, int* stackCount, int* builtCount )
{
	B3_ASSERT( 2 <= subRangeCount && subRangeCount <= 4 );
	tree->swapNodes[nodeIndex] = b3MakeEmptyNode( false );
	tree->builtNodes[*builtCount] = nodeIndex;
	*builtCount += 1;

	for ( int i = 0; i < subRangeCount; ++i )
	{
		b3BuildRange sub = subRanges[i];
		B3_ASSERT( sub.count > 0 );

		if ( sub.count == 1 && b3IsProxyLeaf( tree, tree->leafIndices[sub.start] ) == false )
		{
			b3PlaceRetainedNode( tree, nodeIndex, i, tree->leafIndices[sub.start] );
			continue;
		}

		int child = b3BumpNode( tree, true );
		tree->swapNodes[nodeIndex].child[i] = (uint32_t)child;
		tree->parents[child] = B3_LANE_REF( nodeIndex, i );

		if ( *stackCount < B3_TREE_STACK_SIZE )
		{
			sub.nodeIndex = child;
			stack[*stackCount] = sub;
			*stackCount += 1;
		}
		else
		{
			// todo fail gracefully if the stack runs out in release
			B3_ASSERT( *stackCount < B3_TREE_STACK_SIZE );
		}
	}
}

// Build the new array over the build leaves. Nodes are allocated parent first so the sweep
// refit works on the result. Ranges above the small size are split in four by two levels of
// median split. Small ranges are sorted along their longest axis and chunked into full nodes.
static void b3BuildTree( b3DynamicTree* tree, int leafCount, b3CenterBounds leafBounds )
{
	int* leafIndices = tree->leafIndices;
	b3Vec3* leafCenters = tree->leafCenters;

	// Bump allocation into the swap nodes.
	tree->nodeEnd = 0;
	int root = b3BumpNode( tree, true );
	B3_ASSERT( root == B3_ROOT_NODE );

	int builtCount = 0;

	b3BuildRange stack[B3_TREE_STACK_SIZE];
	int stackCount = 0;
	stack[stackCount++] = B3_LITERAL( b3BuildRange ){ 0, leafCount, B3_ROOT_NODE, leafBounds };

	while ( stackCount > 0 )
	{
		b3BuildRange range = stack[--stackCount];

		if ( range.count <= 4 )
		{
			b3FillSmallGroup( tree, range, &builtCount );
			continue;
		}

		b3BuildRange subRanges[4];

		if ( range.count <= B3_SMALL_RANGE_SIZE )
		{
			b3SortRange( leafIndices + range.start, leafCenters + range.start, range.count, range.bounds );

			// Three full chunks and whatever remains. Full chunks become leaf nodes and never read
			// their bounds. The remainder can be sorted again, so it gets a scan, at most 20 items.
			int subRangeCount = 0;
			int start = range.start;
			int remaining = range.count;
			while ( remaining > 4 && subRangeCount < 3 )
			{
				subRanges[subRangeCount++] = B3_LITERAL( b3BuildRange ){ start, 4, B3_NULL_INDEX, range.bounds };
				start += 4;
				remaining -= 4;
			}

			if ( remaining > 0 )
			{
				b3CenterBounds remainderBounds =
					remaining > 4 ? b3ComputeCenterBounds( leafCenters + start, remaining ) : range.bounds;
				subRanges[subRangeCount++] = B3_LITERAL( b3BuildRange ){ start, remaining, B3_NULL_INDEX, remainderBounds };
			}

			b3CreateChildren( tree, range.nodeIndex, subRanges, subRangeCount, stack, &stackCount, &builtCount );
			continue;
		}

		// Two levels of median split. The first keeps two on each side so the second has something
		// to cut. Each split hands its sides their center bounds, so no range is scanned twice.
		int* indices = leafIndices + range.start;
		b3Vec3* centers = leafCenters + range.start;
		int count = range.count;

		b3CenterBounds bounds1A, bounds1B;
		int split1 = b3PartitionMid( indices, centers, count, range.bounds, &bounds1A, &bounds1B );
		if ( split1 < 2 || count - split1 < 2 )
		{
			split1 = count / 2;
			bounds1A = b3ComputeCenterBounds( centers, split1 );
			bounds1B = b3ComputeCenterBounds( centers + split1, count - split1 );
		}

		b3CenterBounds bounds[4];
		int split0 = b3PartitionMid( indices, centers, split1, bounds1A, bounds + 0, bounds + 1 );
		int split2 =
			split1 + b3PartitionMid( indices + split1, centers + split1, count - split1, bounds1B, bounds + 2, bounds + 3 );

		subRanges[0] = B3_LITERAL( b3BuildRange ){ range.start, split0, B3_NULL_INDEX, bounds[0] };
		subRanges[1] = B3_LITERAL( b3BuildRange ){ range.start + split0, split1 - split0, B3_NULL_INDEX, bounds[1] };
		subRanges[2] = B3_LITERAL( b3BuildRange ){ range.start + split1, split2 - split1, B3_NULL_INDEX, bounds[2] };
		subRanges[3] = B3_LITERAL( b3BuildRange ){ range.start + split2, count - split2, B3_NULL_INDEX, bounds[3] };

		b3CreateChildren( tree, range.nodeIndex, subRanges, 4, stack, &stackCount, &builtCount );
	}

	// The built internal nodes were listed parent first, so walking back fills every lane from
	// a finished child. Lanes holding copied subtrees are refreshed too, which is harmless.
	b3WideNode* nodes = tree->swapNodes;
	for ( int i = builtCount - 1; i >= 0; --i )
	{
		int nodeIndex = tree->builtNodes[i];
		b3WideNode* node = nodes + nodeIndex;
		B3_ASSERT( b3IsLeaf( node ) == false );
		for ( int lane = 0; lane < 4; ++lane )
		{
			if ( b3IsLaneEmpty( node->child[lane] ) )
			{
				continue;
			}

			b3SetNodeLane( nodes, nodeIndex, lane, b3GetLaneIndex( node->child[lane] ) );
		}
	}
}

// Rebuild the stale parts of the tree. The entire tree is put into parent first order. This
// makes refitting much faster. This is done async with threading, so the cost is hidden.
// Not safe to access tree during this operation.
int b3DynamicTree_Rebuild( b3DynamicTree* tree, bool fullBuild )
{
	int proxyCount = tree->proxyCount;
	if ( proxyCount == 0 )
	{
		return 0;
	}

	// An unordered tree is rebuilt even when nothing moved, the sweep refit needs the order
	if ( fullBuild == false && b3HasTreeMoved( tree ) == false && tree->dfsOrdered )
	{
		return 0;
	}

	// Every build leaf is a proxy or a retained subtree root, so the lanes of the old tree bound
	// the count.
	int leafCapacity = b3MaxInt( 4 * tree->nodeEnd, 4 );
	if ( leafCapacity > tree->rebuildCapacity )
	{
		int oldCapacity = tree->rebuildCapacity;
		int newCapacity = leafCapacity + leafCapacity / 2;

		b3Free( tree->leafIndices, oldCapacity * sizeof( int32_t ) );
		tree->leafIndices = (int32_t*)b3Alloc( newCapacity * sizeof( int32_t ) );

		b3Free( tree->leafRefs, oldCapacity * sizeof( int32_t ) );
		tree->leafRefs = (int32_t*)b3Alloc( newCapacity * sizeof( int32_t ) );

		b3Free( tree->leafBoxes, oldCapacity * sizeof( b3AABB ) );
		tree->leafBoxes = (b3AABB*)b3Alloc( newCapacity * sizeof( b3AABB ) );

		b3Free( tree->leafCenters, oldCapacity * sizeof( b3Vec3 ) );
		tree->leafCenters = (b3Vec3*)b3Alloc( newCapacity * sizeof( b3Vec3 ) );

		b3Free( tree->builtNodes, oldCapacity * sizeof( int32_t ) );
		tree->builtNodes = (int32_t*)b3Alloc( newCapacity * sizeof( int32_t ) );

		tree->rebuildCapacity = newCapacity;
	}

	const b3WideNode* nodes = tree->nodes;
	int* leafIndices = tree->leafIndices;
	int* leafRefs = tree->leafRefs;
	b3AABB* leafBoxes = tree->leafBoxes;
	b3Vec3* leafCenters = tree->leafCenters;

	// Gather build leaves. A marked lane holding a node is descended and abandoned in the old
	// array. Every other lane is a build leaf, kept subtrees included.
	int leafCount = 0;
	int stack[B3_TREE_STACK_SIZE];
	int stackCount = 0;
	stack[stackCount++] = B3_ROOT_NODE;

	while ( stackCount > 0 )
	{
		const b3WideNode* node = nodes + stack[--stackCount];
		bool leaf = b3IsLeaf( node );
		for ( int lane = 0; lane < 4; ++lane )
		{
			uint32_t child = node->child[lane];
			if ( b3IsLaneEmpty( child ) )
			{
				continue;
			}

			if ( leaf == false && ( fullBuild || b3IsLaneMoved( child ) ) )
			{
				// todo fail gracefully if the stack runs out in release
				B3_ASSERT( stackCount < B3_TREE_STACK_SIZE );
				stack[stackCount++] = b3GetLaneIndex( child );
				continue;
			}

			if ( fullBuild == false )
			{
				child &= ~B3_MOVED_NODE;
			}

			B3_ASSERT( leafCount < tree->rebuildCapacity );
			leafIndices[leafCount] = leafCount;
			leafRefs[leafCount] = (int32_t)child;
			leafBoxes[leafCount] = b3GetLaneAABB( node, lane );
			leafCenters[leafCount] = b3AABB_Center( leafBoxes[leafCount] );
			leafCount += 1;
		}
	}

	B3_ASSERT( 0 < leafCount && leafCount <= 4 * tree->nodeEnd );

	// The root range bounds seed the build, every split below derives its own from the parent.
	b3CenterBounds leafBounds = b3ComputeCenterBounds( leafCenters, leafCount );

	int oldCapacity = tree->nodeCapacity;
	if ( tree->swapNodes == NULL )
	{
		tree->swapNodes = b3AllocZero( oldCapacity * sizeof( b3WideNode ) );
	}

	b3BuildTree( tree, leafCount, leafBounds );

	// The spare is now the tree and the old array is the spare. The build is dense, so the
	// stale tail of the old array sits above the end and is never read.
	B3_SWAP( tree->nodes, tree->swapNodes );
	tree->nodeFreeList = B3_NULL_INDEX;
	tree->dfsOrdered = true;

	if ( tree->nodeCapacity != oldCapacity )
	{
		// The build outgrew the old array, which is now the spare and too small to swap back in.
		b3Free( tree->swapNodes, oldCapacity * sizeof( b3WideNode ) );
		tree->swapNodes = NULL;
	}

	b3DynamicTree_Validate( tree );
	if ( fullBuild == false )
	{
		b3DynamicTree_ValidateNoMoved( tree );
	}

	return leafCount;
}

// Set the moved flag on the lanes above the proxy. Serial use case.
void b3DynamicTree_MarkProxyMovedSerial( b3DynamicTree* tree, int proxyId )
{
	B3_VALIDATE( 0 <= proxyId && proxyId < tree->proxyCapacity );

	b3WideNode* nodes = tree->nodes;
	const int32_t* parents = tree->parents;

	int ref = tree->proxies[proxyId].leaf;
	B3_VALIDATE( 0 <= ref && B3_LANE_REF_NODE( ref ) < tree->nodeEnd );
	B3_VALIDATE( b3IsLeaf( nodes + B3_LANE_REF_NODE( ref ) ) );

	while ( ref != B3_NULL_INDEX )
	{
		int index = B3_LANE_REF_NODE( ref );
		nodes[index].child[B3_LANE_REF_LANE( ref )] |= B3_MOVED_NODE;
		ref = parents[index];
	}
}

// Update a proxy AABB and flag the lanes above as moved. Thread-safe using atomics. The leaf
// lane is owned by the proxy, the lanes above are shared with the proxies beside it.
void b3DynamicTree_MarkProxyMoved( b3DynamicTree* tree, int proxyId, b3AABB aabb )
{
	B3_VALIDATE( 0 <= proxyId && proxyId < tree->proxyCapacity );

	b3WideNode* nodes = tree->nodes;
	const int32_t* parents = tree->parents;

	int ref = tree->proxies[proxyId].leaf;
	B3_VALIDATE( 0 <= ref && B3_LANE_REF_NODE( ref ) < tree->nodeEnd );

	int index = B3_LANE_REF_NODE( ref );
	int lane = B3_LANE_REF_LANE( ref );
	b3WideNode* node = nodes + index;
	B3_VALIDATE( b3IsLeaf( node ) );
	B3_VALIDATE( b3AABB_Contains( b3GetLaneAABB( node, lane ), aabb ) == false );

	// This is not raced since it is the leaf lane.
	b3SetLaneAABB( node, lane, aabb );
	node->child[lane] |= B3_MOVED_NODE;

	ref = parents[index];
	while ( ref != B3_NULL_INDEX )
	{
		index = B3_LANE_REF_NODE( ref );
		uint32_t* word = nodes[index].child + B3_LANE_REF_LANE( ref );

		// Read first to avoid the FetchOr if possible.
		if ( b3AtomicLoadU32Raw( word ) & B3_MOVED_NODE )
		{
			break;
		}

		uint32_t previousFlags = b3AtomicFetchOrU32( word, B3_MOVED_NODE );
		if ( previousFlags & B3_MOVED_NODE )
		{
			// Ancestor already visited.
			break;
		}

		ref = parents[index];
	}
}

// Clear the moved flags from the entire tree.
void b3DynamicTree_ClearMoved( b3DynamicTree* tree )
{
	b3WideNode* nodes = tree->nodes;

	if ( b3HasMovedLane( nodes + B3_ROOT_NODE ) == false )
	{
		return;
	}

	uint32_t stack[B3_TREE_STACK_SIZE];
	int stackCount = 0;
	stack[stackCount++] = B3_ROOT_NODE;

	while ( stackCount > 0 )
	{
		b3WideNode* node = nodes + b3PopNodeIndex( stack[--stackCount] );
		int moved = b3GetMovedMask( node );

		node->child[0] &= ~B3_MOVED_NODE;
		node->child[1] &= ~B3_MOVED_NODE;
		node->child[2] &= ~B3_MOVED_NODE;
		node->child[3] &= ~B3_MOVED_NODE;

		if ( b3IsLeaf( node ) )
		{
			continue;
		}

		if ( stackCount > B3_TREE_STACK_SIZE - 4 )
		{
			// Bad stuff will happen if the moved flags don't get cleared.
			B3_ASSERT( stackCount <= B3_TREE_STACK_SIZE - 4 );
			continue;
		}

		stackCount = b3PushLanesW4( stack, stackCount, node, moved );
	}
}

// Refresh the moved lanes of an internal node from the children below them
static inline void b3RefitNode( b3WideNode* nodes, int index )
{
	b3WideNode* node = nodes + index;
	for ( int lane = 0; lane < 4; ++lane )
	{
		uint32_t child = node->child[lane];
		if ( ( child & B3_MOVED_NODE ) == 0 )
		{
			continue;
		}

		b3SetLaneAABB( node, lane, b3GetSubtreeAABB( nodes + b3GetLaneIndex( child ) ) );
	}
}

// Slow refit for unit tests.
static void b3RefitSubtree( b3WideNode* nodes, int rootIndex )
{
	if ( b3IsLeaf( nodes + rootIndex ) || b3HasMovedLane( nodes + rootIndex ) == false )
	{
		return;
	}

	// Negative entries indicate deferred refits. A parent is pushed before its children so the
	// LIFO order guarantees the children are finished when the refit runs.
	int stack[B3_TREE_STACK_SIZE];
	int stackCount = 0;
	stack[stackCount++] = rootIndex;

	while ( stackCount > 0 )
	{
		int item = stack[--stackCount];
		if ( item < 0 )
		{
			b3RefitNode( nodes, ~item );
			continue;
		}

		b3WideNode* node = nodes + item;
		if ( b3IsLeaf( node ) || b3HasMovedLane( node ) == false )
		{
			continue;
		}

		// ~0 == -1
		stack[stackCount++] = ~item;
		for ( int lane = 0; lane < 4; ++lane )
		{
			if ( ( node->child[lane] & B3_MOVED_NODE ) == 0 )
			{
				continue;
			}

			if ( stackCount < B3_TREE_STACK_SIZE )
			{
				stack[stackCount++] = b3GetLaneIndex( node->child[lane] );
			}
			else
			{
				B3_ASSERT( stackCount < B3_TREE_STACK_SIZE );
			}
		}
	}
}

void b3DynamicTree_Refit( b3DynamicTree* tree )
{
	if ( b3HasTreeMoved( tree ) == false )
	{
		return;
	}

	b3WideNode* nodes = tree->nodes;

	if ( tree->dfsOrdered == false )
	{
		// This only happens in unit tests.
		b3RefitSubtree( nodes, B3_ROOT_NODE );
		return;
	}

	// Children sit above their parents, so a backward sweep finishes every child first. Free
	// nodes and leaf nodes carry the leaf bit and are skipped.
	for ( int index = tree->nodeEnd - 1; index >= 0; --index )
	{
		b3WideNode* node = nodes + index;
		uint32_t any = node->child[0] | node->child[1] | node->child[2] | node->child[3];
		if ( ( any & B3_MOVED_NODE ) == 0 || ( any & B3_LEAF_NODE ) != 0 )
		{
			continue;
		}

		b3RefitNode( nodes, index );
	}
}

// Every lane of an internal node is a box in the tree, colored by depth so the levels stand
// apart. The lanes of a leaf node are the shape boxes. They take a color from the leaf node
// index and get a spoke to the center of their node box, so the shapes sharing a node are easy
// to see.
void b3DrawDynamicTree( const b3DynamicTree* tree, b3DebugDraw* draw )
{
	static const b3HexColor depthColors[] = {
		b3_colorRed,	b3_colorOrange,	 b3_colorYellow, b3_colorLime,	   b3_colorCyan,	  b3_colorDodgerBlue,
		b3_colorViolet, b3_colorMagenta, b3_colorCoral,	 b3_colorGoldenRod, b3_colorSeaGreen, b3_colorPlum,
	};
	const int depthColorCount = sizeof( depthColors ) / sizeof( depthColors[0] );

	static const b3HexColor leafColors[] = {
		b3_colorHotPink,	b3_colorSpringGreen, b3_colorDeepSkyBlue, b3_colorTomato,	 b3_colorGold,	  b3_colorAquamarine,
		b3_colorLightSalmon, b3_colorChartreuse, b3_colorOrchid,	  b3_colorKhaki,	 b3_colorTurquoise, b3_colorPeachPuff,
		b3_colorPaleGreen,	b3_colorLightSkyBlue, b3_colorThistle,	  b3_colorSandyBrown,
	};
	const int leafColorCount = sizeof( leafColors ) / sizeof( leafColors[0] );

	if ( tree->proxyCount == 0 )
	{
		return;
	}

	const b3WideNode* nodes = tree->nodes;
	b3BoxW4 query = b3SplatBoxW4( draw->drawingBounds );

	struct
	{
		int index;
		int depth;
	} stack[B3_TREE_STACK_SIZE];

	int stackCount = 0;
	stack[stackCount].index = B3_ROOT_NODE;
	stack[stackCount].depth = 0;
	stackCount += 1;

	while ( stackCount > 0 )
	{
		stackCount -= 1;
		int index = stack[stackCount].index;
		int depth = stack[stackCount].depth;
		const b3WideNode* node = nodes + index;

		b3BoxW4 boxes = b3LoadBoxW4( node );
		int mask = b3OverlapMaskW4( &query, &boxes );

		if ( b3IsLeaf( node ) )
		{
			b3HexColor color = leafColors[index % leafColorCount];
			b3Pos center = b3ToPos( b3AABB_Center( b3GetSubtreeAABB( node ) ) );
			while ( mask != 0 )
			{
				int lane = (int)b3CTZ32( (uint32_t)mask );
				mask &= mask - 1;

				b3AABB laneBox = b3GetLaneAABB( node, lane );
				draw->DrawBoundsFcn( laneBox, color, draw->context );
				draw->DrawSegmentFcn( b3ToPos( b3AABB_Center( laneBox ) ), center, color, draw->context );
			}

			draw->DrawPointFcn( center, 5.0f, color, draw->context );
			continue;
		}

		b3HexColor color = depthColors[depth % depthColorCount];

		while ( mask != 0 )
		{
			int lane = (int)b3CTZ32( (uint32_t)mask );
			mask &= mask - 1;

			draw->DrawBoundsFcn( b3GetLaneAABB( node, lane ), color, draw->context );

			if ( stackCount < B3_TREE_STACK_SIZE )
			{
				stack[stackCount].index = b3GetLaneIndex( node->child[lane] );
				stack[stackCount].depth = depth + 1;
				stackCount += 1;
			}
			else
			{
				B3_ASSERT( stackCount < B3_TREE_STACK_SIZE );
			}
		}
	}
}

static FILE* b3OpenTreeFile( const char* fileName, const char* mode )
{
	FILE* file = NULL;

#if defined( _MSC_VER )
	errno_t e = fopen_s( &file, fileName, mode );
	if ( e != 0 )
	{
		return NULL;
	}
#else
	file = fopen( fileName, mode );
	if ( file == NULL )
	{
		return NULL;
	}
#endif

	return file;
}

void b3DynamicTree_Save( const b3DynamicTree* tree, const char* fileName )
{
	if ( tree->nodes == NULL || tree->parents == NULL || tree->proxies == NULL )
	{
		return;
	}

	FILE* file = b3OpenTreeFile( fileName, "wb" );
	if ( file == NULL )
	{
		return;
	}

	// Copy to allow setting some fields to zero
	b3DynamicTree temp = *tree;

	// Zero pointers and temp data
	temp.nodes = NULL;
	temp.parents = NULL;
	temp.proxies = NULL;
	temp.swapNodes = NULL;
	temp.leafIndices = NULL;
	temp.leafRefs = NULL;
	temp.leafBoxes = NULL;
	temp.leafCenters = NULL;
	temp.builtNodes = NULL;
	temp.rebuildCapacity = 0;

	temp.nodeCapacity = tree->nodeEnd;

	fwrite( &temp, sizeof( b3DynamicTree ), 1, file );

	if ( tree->nodeEnd > 0 )
	{
		fwrite( tree->nodes, sizeof( b3WideNode ), tree->nodeEnd, file );
		fwrite( tree->parents, sizeof( int32_t ), tree->nodeEnd, file );
	}

	if ( tree->proxyCapacity > 0 )
	{
		fwrite( tree->proxies, sizeof( b3TreeProxy ), tree->proxyCapacity, file );
	}

	fclose( file );
}

b3DynamicTree b3DynamicTree_Load( const char* fileName, float scale )
{
	b3DynamicTree tree = { 0 };

	FILE* file = b3OpenTreeFile( fileName, "rb" );
	if ( file == NULL )
	{
		return tree;
	}

	int readCount = (int)fread( &tree, sizeof( b3DynamicTree ), 1, file );
	if ( readCount != 1 )
	{
		fclose( file );
		memset( &tree, 0, sizeof( b3DynamicTree ) );
		return tree;
	}

	if ( tree.version != B3_DYNAMIC_TREE_VERSION )
	{
		fclose( file );
		memset( &tree, 0, sizeof( b3DynamicTree ) );
		return tree;
	}

	tree.nodes = NULL;
	tree.parents = NULL;
	tree.proxies = NULL;
	tree.swapNodes = NULL;
	tree.leafIndices = NULL;
	tree.leafRefs = NULL;
	tree.leafBoxes = NULL;
	tree.leafCenters = NULL;
	tree.builtNodes = NULL;
	tree.rebuildCapacity = 0;

	bool ok = tree.nodeCapacity >= 1 && tree.proxyCapacity >= 1 && tree.nodeEnd >= 1 && tree.nodeEnd <= tree.nodeCapacity;

	if ( ok )
	{
		tree.nodes = (b3WideNode*)b3Alloc( tree.nodeCapacity * sizeof( b3WideNode ) );
		tree.parents = (int32_t*)b3Alloc( tree.nodeCapacity * sizeof( int32_t ) );
		tree.proxies = (b3TreeProxy*)b3Alloc( tree.proxyCapacity * sizeof( b3TreeProxy ) );

		ok = (int)fread( tree.nodes, sizeof( b3WideNode ), tree.nodeCapacity, file ) == tree.nodeCapacity;
		ok = ok && (int)fread( tree.parents, sizeof( int32_t ), tree.nodeCapacity, file ) == tree.nodeCapacity;
		ok = ok && (int)fread( tree.proxies, sizeof( b3TreeProxy ), tree.proxyCapacity, file ) == tree.proxyCapacity;
	}

	if ( ok == false )
	{
		b3Free( tree.nodes, tree.nodeCapacity * sizeof( b3WideNode ) );
		b3Free( tree.parents, tree.nodeCapacity * sizeof( int32_t ) );
		b3Free( tree.proxies, tree.proxyCapacity * sizeof( b3TreeProxy ) );
		fclose( file );
		memset( &tree, 0, sizeof( b3DynamicTree ) );
		return tree;
	}

	// Empty lanes are infinite and stay that way under a positive scale.
	for ( int i = 0; i < tree.nodeEnd; ++i )
	{
		b3WideNode* node = tree.nodes + i;
		for ( int lane = 0; lane < 4; ++lane )
		{
			node->lowerX[lane] *= scale;
			node->lowerY[lane] *= scale;
			node->lowerZ[lane] *= scale;
			node->upperX[lane] *= scale;
			node->upperY[lane] *= scale;
			node->upperZ[lane] *= scale;
		}
	}

	fclose( file );

	return tree;
}
