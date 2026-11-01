// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#include "broad_phase.h"

#include "aabb.h"
#include "arena_allocator.h"
#include "body.h"
#include "compound.h"
#include "contact.h"
#include "core.h"
#include "ctz.h"
#include "dynamic_tree.h"
#include "parallel_for.h"
#include "physics_world.h"
#include "platform.h"
#include "qsort.h"
#include "shape.h"
#include "simd.h"

#include <stdbool.h>
#include <string.h>

void b3CreateBroadPhase( b3BroadPhase* bp, const b3Capacity* capacity )
{
	_Static_assert( b3_bodyTypeCount == 3, "must be three body types" );

	bp->pairSet = b3CreateSet( b3MaxInt( 32, 2 * capacity->contactCount ) );

	int staticCapacity = b3MaxInt( 16, capacity->staticShapeCount );
	bp->trees[b3_staticBody] = b3DynamicTree_Create( staticCapacity );

	int kinematicCapacity = 16;
	bp->trees[b3_kinematicBody] = b3DynamicTree_Create( kinematicCapacity );

	int dynamicCapacity = b3MaxInt( 16, capacity->dynamicShapeCount );
	bp->trees[b3_dynamicBody] = b3DynamicTree_Create( dynamicCapacity );

	bp->movedNodes = NULL;
}

void b3DestroyBroadPhase( b3BroadPhase* bp )
{
	for ( int i = 0; i < b3_bodyTypeCount; ++i )
	{
		b3DynamicTree_Destroy( bp->trees + i );
	}

	b3DestroySet( &bp->pairSet );

	memset( bp, 0, sizeof( b3BroadPhase ) );
}

int b3BroadPhase_CreateProxy( b3BroadPhase* bp, b3BodyType proxyType, b3AABB aabb, uint64_t categoryBits, int shapeIndex,
							  bool forcePairCreation )
{
	B3_ASSERT( 0 <= proxyType && proxyType < b3_bodyTypeCount );

	bool mark = ( proxyType != b3_staticBody || forcePairCreation );

	int proxyId = b3CreateTreeProxyInternal( bp->trees + proxyType, aabb, categoryBits, shapeIndex, mark );
	int proxyKey = B3_PROXY_KEY( proxyId, proxyType );
	return proxyKey;
}

void b3BroadPhase_DestroyProxy( b3BroadPhase* bp, int proxyKey )
{
	b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );

	B3_ASSERT( 0 <= proxyType && proxyType < b3_bodyTypeCount );
	b3DynamicTree_DestroyProxy( bp->trees + proxyType, proxyId );
}

void b3BroadPhase_MoveProxy( b3BroadPhase* bp, int proxyKey, b3AABB aabb )
{
	b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );

	bool mark = true;
	b3DynamicTree_MoveProxyInternal( bp->trees + proxyType, proxyId, aabb, mark );
}

// Gather the nodes with a moved lane. This is done serially, it is cache friendly. Free nodes
// are empty leaf nodes without moved lanes, so they drop out.
static int b3GatherMovedNodes( const b3DynamicTree* tree, int* nodeIndices )
{
	const b3WideNode* nodes = tree->nodes;
	int nodeEnd = tree->nodeEnd;

	int count = 0;
	for ( int index = 0; index < nodeEnd; ++index )
	{
		if ( b3HasMovedLane( nodes + index ) )
		{
			nodeIndices[count++] = index;
		}
	}

	return count;
}

#define B3_CANDIDATE_BATCH 32

typedef struct b3CandidatePair
{
	int shapeIdA;
	int shapeIdB;
} b3CandidatePair;

typedef struct b3PairContext
{
	b3World* world;
	b3Array( uint64_t ) * pairKeys;
	b3CandidatePair batch[B3_CANDIDATE_BATCH];
	int batchCount;

	// Compound shapes are static only, so only the static cross pass can meet one. Elsewhere
	// this stays false and the batch never reads a shape type.
	bool checkCompounds;
} b3PairContext;

typedef struct b3IndexPair
{
	int a, b;
} b3IndexPair;

static inline void b3PrefetchHash( b3HashSet* set, uint32_t hash )
{
	uint32_t capacity = set->capacity;
	uint32_t index = hash & ( capacity - 1 );
	b3Prefetch( set->items + index );
}

// The shape pair gauntlet. Everything here is a property of the pair, not of a compound child,
// so a compound runs it once for the whole pair.
static bool b3ShouldCreatePair( b3World* world, int shapeIdA, int shapeIdB )
{
	b3Shape* shapeA = b3Array_Get( world->shapes, shapeIdA );
	b3Shape* shapeB = b3Array_Get( world->shapes, shapeIdB );

	int bodyIdA = shapeA->bodyId;
	int bodyIdB = shapeB->bodyId;

	// Are the shapes on the same body?
	if ( bodyIdA == bodyIdB )
	{
		return false;
	}

	// Sensors are handled elsewhere
	if ( shapeA->sensorIndex != B3_NULL_INDEX || shapeB->sensorIndex != B3_NULL_INDEX )
	{
		return false;
	}

	if ( b3ShouldShapesCollide( shapeA->filter, shapeB->filter ) == false )
	{
		return false;
	}

	// Does a joint override collision?
	b3Body* bodyA = b3Array_Get( world->bodies, bodyIdA );
	b3Body* bodyB = b3Array_Get( world->bodies, bodyIdB );
	if ( b3ShouldBodiesCollide( world, bodyA, bodyB ) == false )
	{
		return false;
	}

	// Custom user filter
	if ( ( shapeA->flags & b3_enableCustomFiltering ) || ( shapeB->flags & b3_enableCustomFiltering ) )
	{
		b3CustomFilterFcn* customFilterFcn = world->customFilterFcn;
		if ( customFilterFcn != NULL )
		{
			b3ShapeId idA = { shapeIdA + 1, world->worldId, shapeA->generation };
			b3ShapeId idB = { shapeIdB + 1, world->worldId, shapeB->generation };
			if ( customFilterFcn( idA, idB, world->customFilterContext ) == false )
			{
				return false;
			}
		}
	}

	return true;
}

typedef struct b3CompoundPairContext
{
	b3PairContext* pairContext;
	b3HashSet* pairSet;
	int compoundShapeId;
	int otherShapeId;
} b3CompoundPairContext;

static bool b3CompoundChildCallback( int proxyId, uint64_t userData, void* context )
{
	B3_UNUSED( proxyId );

	b3CompoundPairContext* compoundContext = context;
	int childIndex = (int)userData;

	uint64_t key = b3ShapePairKey( compoundContext->compoundShapeId, compoundContext->otherShapeId, childIndex );
	if ( b3ContainsKey( compoundContext->pairSet, key ) == false )
	{
		b3Array_Push( *compoundContext->pairContext->pairKeys, key );
	}

	return true;
}

// One of the two shapes is a compound, so the pair becomes one pair per overlapping child. The
// child boxes live in the compound frame, so the other shape's box has to be demoted and pulled
// into that frame first.
static void b3EmitCompoundPairs( b3PairContext* context, int compoundShapeId, int otherShapeId )
{
	b3World* world = context->world;

	b3Shape* compoundShape = b3Array_Get( world->shapes, compoundShapeId );
	b3Shape* otherShape = b3Array_Get( world->shapes, otherShapeId );

	int otherProxyKey = otherShape->proxyKey;
	B3_ASSERT( otherProxyKey != B3_NULL_INDEX );
	b3AABB otherAABB =
		b3DynamicTree_GetAABB( world->broadPhase.trees + B3_PROXY_TYPE( otherProxyKey ), B3_PROXY_ID( otherProxyKey ) );

	// Query bounds are float world space, so the demoted transform is the matching float frame
	b3Transform compoundTransform = b3ToRelativeTransform( b3GetBodyTransform( world, compoundShape->bodyId ), b3Pos_zero );
	b3AABB localAABB = b3AABB_Transform( b3InvertTransform( compoundTransform ), otherAABB );

	b3CompoundPairContext compoundContext = {
		.pairContext = context,
		.pairSet = &world->broadPhase.pairSet,
		.compoundShapeId = compoundShapeId,
		.otherShapeId = otherShapeId,
	};

	int startCount = context->pairKeys->count;

	b3TreeView view = b3GetCompoundTreeView( compoundShape->compound );
	b3TreeView_Query( &view, localAABB, B3_DEFAULT_MASK_BITS, false, b3CompoundChildCallback, &compoundContext );

	if ( context->pairKeys->count > startCount && b3ShouldCreatePair( world, compoundShapeId, otherShapeId ) == false )
	{
		b3Array_Resize( *context->pairKeys, startCount );
	}
}

static void b3FlushCandidatePairs( b3PairContext* context )
{
	b3World* world = context->world;
	b3BroadPhase* bp = &world->broadPhase;

	int count1 = context->batchCount;
	context->batchCount = 0;

	b3CandidatePair plain[B3_CANDIDATE_BATCH];
	int plainCount = 0;

	if ( context->checkCompounds )
	{
		// A compound expands into one pair per child, so its key is not known yet and it cannot
		// ride the batched cull below.
		const b3Shape* shapes = world->shapes.data;
		for ( int i = 0; i < count1; ++i )
		{
			b3CandidatePair candidate = context->batch[i];
			bool compoundA = shapes[candidate.shapeIdA].type == b3_compoundShape;
			bool compoundB = shapes[candidate.shapeIdB].type == b3_compoundShape;

			if ( compoundA )
			{
				// Compound versus compound is not supported
				B3_ASSERT( compoundB == false );
				b3EmitCompoundPairs( context, candidate.shapeIdA, candidate.shapeIdB );
			}
			else if ( compoundB )
			{
				b3EmitCompoundPairs( context, candidate.shapeIdB, candidate.shapeIdA );
			}
			else
			{
				plain[plainCount] = candidate;
				plainCount += 1;
			}
		}
	}
	else
	{
		memcpy( plain, context->batch, count1 * sizeof( b3CandidatePair ) );
		plainCount = count1;
	}

	// Prefetch hash set entries. Less than 1% gain in Box2D but the 3D shape is fatter.
	uint64_t keys[B3_CANDIDATE_BATCH];
	uint32_t hashes[B3_CANDIDATE_BATCH];
	for ( int i = 0; i < plainCount; ++i )
	{
		keys[i] = b3ShapePairKey( plain[i].shapeIdA, plain[i].shapeIdB, 0 );
		hashes[i] = b3KeyHash( keys[i] );
		b3PrefetchHash( &bp->pairSet, hashes[i] );
	}

	// Cull existing pairs.
	b3CandidatePair candidates[B3_CANDIDATE_BATCH];
	int count2 = 0;
	for ( int i = 0; i < plainCount; ++i )
	{
		if ( b3ContainsHashedKey( &bp->pairSet, keys[i], hashes[i] ) == false )
		{
			candidates[count2] = plain[i];
			keys[count2] = keys[i];
			count2 += 1;
		}
	}

	// Prefetch shapes.
	const b3Shape* shapes = world->shapes.data;
	for ( int i = 0; i < count2; ++i )
	{
		b3Prefetch( shapes + candidates[i].shapeIdA );
		b3Prefetch( shapes + candidates[i].shapeIdB );
	}

	// Filter candidates.
	for ( int i = 0; i < count2; ++i )
	{
		if ( b3ShouldCreatePair( world, candidates[i].shapeIdA, candidates[i].shapeIdB ) )
		{
			// The pair passed the gauntlet. A new contact will be created.
			b3Array_Push( *context->pairKeys, keys[i] );
		}
	}
}

B3_FORCE_INLINE void b3AddCandidatePair( int shapeIdA, int shapeIdB, b3PairContext* context )
{
	// Follow shape index order.
	b3CandidatePair* candidate = context->batch + context->batchCount;
	candidate->shapeIdA = b3MinInt( shapeIdA, shapeIdB );
	candidate->shapeIdB = b3MaxInt( shapeIdA, shapeIdB );
	context->batchCount += 1;
	if ( context->batchCount == B3_CANDIDATE_BATCH )
	{
		b3FlushCandidatePairs( context );
	}
}

// A lane box against a subtree in the other array. The lane is a leaf, so every hit is a pair.
// Only pairs with a moved side are reported.
static void b3CollideLaneAndSubtree( const b3BoxW4* laneBox, bool laneMoved, int shapeId, const b3WideNode* nodes, int rootIndex,
									 b3PairContext* context )
{
	int stack[B3_TREE_STACK_SIZE];
	int stackCount = 0;
	stack[stackCount++] = rootIndex;

	while ( stackCount > 0 )
	{
		const b3WideNode* node = nodes + stack[--stackCount];

		b3BoxW4 boxes = b3LoadBoxW4( node );
		int mask = b3OverlapMaskW4( laneBox, &boxes );
		if ( laneMoved == false )
		{
			mask &= b3GetMovedMask( node );
		}

		if ( b3IsLeaf( node ) )
		{
			while ( mask != 0 )
			{
				int lane = (int)b3CTZ32( (uint32_t)mask );
				mask &= mask - 1;
				b3AddCandidatePair( shapeId, (int)node->userData[lane], context );
			}
		}
		else
		{
			while ( mask != 0 )
			{
				int lane = (int)b3CTZ32( (uint32_t)mask );
				mask &= mask - 1;

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
}

// This collides two sub-trees against each other lane by lane. They can live in the same
// dynamic tree. This can only generate pairs cross sub-tree, but not within a sub-tree. This
// fact means this does not generate duplicate pairs.
// For example consider the tree A (B (D E) C (F G))
// Colliding children of A (B and C) can give pairs (D,F) (D,G) (E,F) and (E,G).
// Then colliding children of B can give the pair (D,E) and for C (F,G).
// So no duplicates even when used for self-collision.
// Whenever a proxy is moved, the flag is propagated up the lanes to the root. So
// self collision gathers all those moved nodes and collides their lanes together.
// See Real-time collision detection section 6.3.2. This is faster than querying every moved
// proxy against the whole tree. Scaling is linear instead of linear * log.
// Many other physics engines do this (Bepu, Rapier, etc). So nothing new here.
// Each lane of A is tested against the four lanes of B at once. When only one side is a leaf
// it is made A and each of its lanes descends the subtrees of B on a local stack.
static void b3CollideCrossPairs( const b3WideNode* arrayA, const b3WideNode* arrayB, int rootA, int rootB,
								 b3PairContext* context )
{
	b3IndexPair stack[B3_TREE_STACK_SIZE];
	int stackCount = 0;
	stack[stackCount++] = B3_LITERAL( b3IndexPair ){ .a = rootA, .b = rootB };

	while ( stackCount > 0 )
	{
		b3IndexPair pair = stack[--stackCount];
		const b3WideNode* nodeA = arrayA + pair.a;
		const b3WideNode* nodeB = arrayB + pair.b;
		const b3WideNode* arrayOfB = arrayB;

		bool leafA = b3IsLeaf( nodeA );
		bool leafB = b3IsLeaf( nodeB );
		if ( leafA == false && leafB )
		{
			// Keep the leaf on the A side. Pairs are order free.
			const b3WideNode* temp = nodeA;
			nodeA = nodeB;
			nodeB = temp;
			arrayOfB = arrayA;
			leafA = true;
			leafB = false;
		}

		b3BoxW4 boxesB = b3LoadBoxW4( nodeB );
		int movedB = b3GetMovedMask( nodeB );

		for ( int i = 0; i < 4; ++i )
		{
			uint32_t childA = nodeA->child[i];
			if ( b3IsLaneEmpty( childA ) )
			{
				continue;
			}

			b3BoxW4 laneBox = b3LaneBoxW4( nodeA, i );
			int mask = b3OverlapMaskW4( &laneBox, &boxesB );
			bool movedA = ( childA & B3_MOVED_NODE ) != 0;
			if ( movedA == false )
			{
				mask &= movedB;
			}

			if ( leafA && leafB )
			{
				while ( mask != 0 )
				{
					int j = (int)b3CTZ32( (uint32_t)mask );
					mask &= mask - 1;
					b3AddCandidatePair( (int)nodeA->userData[i], (int)nodeB->userData[j], context );
				}
			}
			else if ( leafA == false )
			{
				while ( mask != 0 )
				{
					int j = (int)b3CTZ32( (uint32_t)mask );
					mask &= mask - 1;

					if ( stackCount < B3_TREE_STACK_SIZE )
					{
						stack[stackCount++] = B3_LITERAL( b3IndexPair ){
							.a = b3GetLaneIndex( childA ),
							.b = b3GetLaneIndex( nodeB->child[j] ),
						};
					}
					else
					{
						B3_ASSERT( stackCount < B3_TREE_STACK_SIZE );
					}
				}
			}
			else
			{
				while ( mask != 0 )
				{
					int j = (int)b3CTZ32( (uint32_t)mask );
					mask &= mask - 1;
					int childB = b3GetLaneIndex( nodeB->child[j] );
					b3CollideLaneAndSubtree( &laneBox, movedA, (int)nodeA->userData[i], arrayOfB, childB, context );
				}
			}
		}
	}
}

// Collide the lanes of one node against each other. Every lane pair with a moved side and an
// overlap is a candidate pair for a leaf node or a pair of subtrees for an internal node. The
// descendants handle their own lanes, so nothing is descended here.
static void b3CollideSelfNode( const b3WideNode* nodes, int nodeIndex, b3PairContext* context )
{
	const b3WideNode* node = nodes + nodeIndex;
	bool leaf = b3IsLeaf( node );
	b3BoxW4 boxes = b3LoadBoxW4( node );
	int moved = b3GetMovedMask( node );

	for ( int i = 0; i < 3; ++i )
	{
		uint32_t childI = node->child[i];
		if ( b3IsLaneEmpty( childI ) )
		{
			continue;
		}

		// Lanes above i only, so each pair is tested once
		b3BoxW4 laneBox = b3LaneBoxW4( node, i );
		int mask = b3OverlapMaskW4( &laneBox, &boxes ) & ~( ( 1 << ( i + 1 ) ) - 1 );
		if ( ( childI & B3_MOVED_NODE ) == 0 )
		{
			mask &= moved;
		}

		while ( mask != 0 )
		{
			int j = (int)b3CTZ32( (uint32_t)mask );
			mask &= mask - 1;

			if ( leaf )
			{
				b3AddCandidatePair( (int)node->userData[i], (int)node->userData[j], context );
			}
			else
			{
				b3CollideCrossPairs( nodes, nodes, b3GetLaneIndex( childI ), b3GetLaneIndex( node->child[j] ), context );
			}
		}
	}
}

// This takes moved nodes and collides their lanes against each other.
static void b3SelfPairsTask( int startIndex, int endIndex, int workerIndex, void* context )
{
	b3TracyCZoneNC( self_pairs, "Self", b3_colorCoral, true );

	b3World* world = context;
	b3BroadPhase* bp = &world->broadPhase;
	const b3DynamicTree* tree = bp->trees + b3_dynamicBody;
	const b3WideNode* nodes = tree->nodes;
	const int* nodeIndices = bp->movedNodes;

	b3PairContext pairContext = {
		.world = world,
		.pairKeys = &world->taskContexts.data[workerIndex].pairKeys,
	};

	for ( int i = startIndex; i < endIndex; ++i )
	{
		b3CollideSelfNode( nodes, nodeIndices[i], &pairContext );
	}

	b3FlushCandidatePairs( &pairContext );

	b3TracyCZoneEnd( self_pairs );
}

#define B3_CROSS_SEED_COUNT 128
_Static_assert( ( B3_CROSS_SEED_COUNT & ( B3_CROSS_SEED_COUNT - 1 ) ) == 0, "must be power of 2" );

// This does a serial cross-tree breadth first search until the queue is full. Then it returns
// the queue pairs as seeds for a parallel search.
static int b3GatherCrossSeeds( const b3DynamicTree* treeA, const b3DynamicTree* treeB, b3IndexPair* seeds )
{
	const b3WideNode* nodesA = treeA->nodes;
	const b3WideNode* nodesB = treeB->nodes;

	// Nothing moved on either side means no pairs. The roots have no lane of their own, so
	// the first pop tests the root lanes against each other. An empty tree has inverted lanes
	// that overlap nothing.
	if ( b3HasMovedLane( nodesA + B3_ROOT_NODE ) == false && b3HasMovedLane( nodesB + B3_ROOT_NODE ) == false )
	{
		return 0;
	}

	// Breadth-first search from the two roots as one pair.
	b3IndexPair queue[2 * B3_CROSS_SEED_COUNT];
	int mask = 2 * B3_CROSS_SEED_COUNT - 1;
	int head = 0;
	int tail = 0;

	queue[tail & mask] = B3_LITERAL( b3IndexPair ){ .a = B3_ROOT_NODE, .b = B3_ROOT_NODE };
	tail += 1;

	// A popped pair can push up to sixteen lane pairs
	int seedCount = 0;
	while ( head < tail && seedCount + ( tail - head ) + 15 < B3_CROSS_SEED_COUNT )
	{
		b3IndexPair pair = queue[head & mask];
		head += 1;

		const b3WideNode* nodeA = nodesA + pair.a;
		const b3WideNode* nodeB = nodesB + pair.b;
		if ( b3IsLeaf( nodeA ) || b3IsLeaf( nodeB ) )
		{
			seeds[seedCount++] = pair;
			continue;
		}

		b3BoxW4 boxesB = b3LoadBoxW4( nodeB );
		int movedB = b3GetMovedMask( nodeB );

		for ( int i = 0; i < 4; ++i )
		{
			uint32_t childA = nodeA->child[i];
			if ( b3IsLaneEmpty( childA ) )
			{
				continue;
			}

			b3BoxW4 laneBox = b3LaneBoxW4( nodeA, i );
			int hitMask = b3OverlapMaskW4( &laneBox, &boxesB );
			if ( ( childA & B3_MOVED_NODE ) == 0 )
			{
				hitMask &= movedB;
			}

			while ( hitMask != 0 )
			{
				int j = (int)b3CTZ32( (uint32_t)hitMask );
				hitMask &= hitMask - 1;
				queue[tail & mask] = B3_LITERAL( b3IndexPair ){
					.a = b3GetLaneIndex( childA ),
					.b = b3GetLaneIndex( nodeB->child[j] ),
				};
				tail += 1;
			}
		}
	}

	while ( head < tail )
	{
		seeds[seedCount] = queue[head & mask];
		seedCount += 1;
		head += 1;
	}

	return seedCount;
}

typedef struct b3CrossContext
{
	b3World* world;
	const b3IndexPair* seeds;
	int staticSeedCount;
	bool checkCompounds;
} b3CrossContext;

static void b3CrossPairsTask( int startIndex, int endIndex, int workerIndex, void* context )
{
	b3TracyCZoneNC( cross_pairs, "Cross", b3_colorCoral, true );

	b3CrossContext* crossContext = context;
	b3World* world = crossContext->world;
	b3BroadPhase* bp = &world->broadPhase;
	const b3WideNode* staticNodes = bp->trees[b3_staticBody].nodes;
	const b3WideNode* kinematicNodes = bp->trees[b3_kinematicBody].nodes;
	const b3WideNode* dynamicNodes = bp->trees[b3_dynamicBody].nodes;

	b3PairContext pairContext = {
		.world = world,
		.pairKeys = &world->taskContexts.data[workerIndex].pairKeys,
	};

	// Static seeds come first, so the two passes are contiguous sub-ranges. The batch is drained
	// between them because only the static pass can meet a compound.
	int staticEnd = b3MinInt( endIndex, crossContext->staticSeedCount );

	pairContext.checkCompounds = crossContext->checkCompounds;
	for ( int i = startIndex; i < staticEnd; ++i )
	{
		b3IndexPair seed = crossContext->seeds[i];
		b3CollideCrossPairs( dynamicNodes, staticNodes, seed.a, seed.b, &pairContext );
	}
	b3FlushCandidatePairs( &pairContext );

	pairContext.checkCompounds = false;
	for ( int i = b3MaxInt( startIndex, crossContext->staticSeedCount ); i < endIndex; ++i )
	{
		b3IndexPair seed = crossContext->seeds[i];
		b3CollideCrossPairs( dynamicNodes, kinematicNodes, seed.a, seed.b, &pairContext );
	}
	b3FlushCandidatePairs( &pairContext );

	b3TracyCZoneEnd( cross_pairs );
}

static void b3UpdateTreesTask( void* context )
{
	b3TracyCZoneNC( tree_task, "Rebuild Trees", b3_colorFireBrick, true );

	b3World* world = (b3World*)context;
	b3DynamicTree_Rebuild( world->broadPhase.trees + b3_dynamicBody, false );
	b3DynamicTree_Rebuild( world->broadPhase.trees + b3_kinematicBody, false );

	b3TracyCZoneEnd( tree_task );
}

// Task that can be done in parallel with the narrow-phase
// - rebuild the collision tree for dynamic and kinematic bodies to keep their query performance good
static void b3EnqueueTreeUpdate( b3World* world )
{
	if ( world->taskCount < B3_MAX_TASKS )
	{
		world->userTreeTask = world->enqueueTaskFcn( &b3UpdateTreesTask, world, world->userTaskContext, "rebuild tree" );
		world->taskCount += 1;
		world->activeTaskCount += world->userTreeTask == NULL ? 0 : 1;
	}
	else
	{
		world->userTreeTask = NULL;
		b3UpdateTreesTask( world );
	}
}

void b3UpdateBroadPhasePairs( b3World* world )
{
	b3BroadPhase* bp = &world->broadPhase;

	bool needUpdate = b3HasTreeMoved( bp->trees + b3_staticBody );
	needUpdate = needUpdate || b3NeedsRebuild( bp->trees + b3_kinematicBody );
	needUpdate = needUpdate || b3NeedsRebuild( bp->trees + b3_dynamicBody );

	if ( needUpdate == false )
	{
		return;
	}

	b3TracyCZoneNC( update_pairs, "Find Pairs", b3_colorMediumSlateBlue, true );

	b3Stack* alloc = &world->stack;

	for ( int i = 0; i < world->workerCount; ++i )
	{
		b3Array_Clear( world->taskContexts.data[i].pairKeys );
	}

	// Generate pairs by querying the dynamic body tree against itself and against
	// the kinematic and static trees.
	{
		// Get the nodes of the dynamic body tree that have a moved lane.
		const b3DynamicTree* dynamicTree = bp->trees + b3_dynamicBody;
		int nodeCapacity = b3MaxInt( dynamicTree->nodeEnd, 1 );
		bp->movedNodes = b3StackAlloc( alloc, nodeCapacity * sizeof( int ), "moved nodes" );
		int dynamicMoveCount = b3GatherMovedNodes( dynamicTree, bp->movedNodes );

		// Get seeds for colliding against the static and kinematic trees.
		b3IndexPair crossSeeds[2 * B3_CROSS_SEED_COUNT];
		int staticSeedCount = b3GatherCrossSeeds( dynamicTree, bp->trees + b3_staticBody, crossSeeds );
		B3_ASSERT( staticSeedCount <= B3_CROSS_SEED_COUNT );
		int kinematicSeedCount = b3GatherCrossSeeds( dynamicTree, bp->trees + b3_kinematicBody, crossSeeds + staticSeedCount );
		B3_ASSERT( kinematicSeedCount <= B3_CROSS_SEED_COUNT );
		int crossMoveCount = staticSeedCount + kinematicSeedCount;

		// Collide the dynamic body tree against the static and kinematic trees.
		b3CrossContext crossContext = {
			.world = world,
			.seeds = crossSeeds,
			.staticSeedCount = staticSeedCount,
			.checkCompounds = world->compoundShapeCount > 0,
		};
		b3ParallelFor( world, &b3CrossPairsTask, crossMoveCount, 1, &crossContext, "cross pairs" );

		// Collide the dynamic body tree against itself.
		b3ParallelFor( world, &b3SelfPairsTask, dynamicMoveCount, 64, world, "self pairs" );
	}

	b3DynamicTree_ClearMoved( bp->trees + b3_staticBody );

	b3TracyCZoneEnd( update_pairs );

	b3TracyCZoneNC( create_contacts, "Create Contacts", b3_colorCoral, true );

	// Update stale trees.
	b3EnqueueTreeUpdate( world );

	// Pairs arrive in deterministic order but scrambled relative to body and shape order.
	// Sorting them here improves solver performance and makes contact order independent of
	// tree structure.
	int pairCount = 0;
	for ( int i = 0; i < world->workerCount; ++i )
	{
		pairCount += world->taskContexts.data[i].pairKeys.count;
	}

	uint64_t* pairKeys = b3StackAlloc( alloc, b3MaxInt( pairCount, 1 ) * sizeof( uint64_t ), "pair keys" );
	int keyCount = 0;
	for ( int i = 0; i < world->workerCount; ++i )
	{
		const b3Array( uint64_t )* workerKeys = &world->taskContexts.data[i].pairKeys;
		if ( workerKeys->count > 0 )
		{
			memcpy( pairKeys + keyCount, workerKeys->data, workerKeys->count * sizeof( uint64_t ) );
			keyCount += workerKeys->count;
		}
	}

	B3_ASSERT( keyCount == pairCount );

	{
#define LESS( i, j ) ( pairKeys[(int)( i )] < pairKeys[(int)( j )] )
#define SWAP( i, j )                                                                                                             \
	do                                                                                                                           \
	{                                                                                                                            \
		uint64_t tmp_ = pairKeys[(int)( i )];                                                                                    \
		pairKeys[(int)( i )] = pairKeys[(int)( j )];                                                                             \
		pairKeys[(int)( j )] = tmp_;                                                                                             \
	}                                                                                                                            \
	while ( 0 )

		QSORT( pairCount, LESS, SWAP );

#undef LESS
#undef SWAP
	}

	for ( int i = 0; i < keyCount; ++i )
	{
		uint64_t key = pairKeys[i];

		// The traversal is cross subtree and the seeds are disjoint, so a duplicate key means a
		// real bug. Trap it in a validation build, and skip rather than double add to the pair
		// set, which a later destroy would then remove once.
		bool duplicate = i > 0 && key == pairKeys[i - 1];
		B3_VALIDATE( duplicate == false );
		if ( duplicate )
		{
			continue;
		}

		int shapeIdA = (int)( ( key >> ( 64 - B3_SHAPE_POWER ) ) & B3_SHAPE_MASK );
		int shapeIdB = (int)( ( key >> ( 64 - 2 * B3_SHAPE_POWER ) ) & B3_SHAPE_MASK );
		int childIndex = (int)( key & B3_CHILD_MASK );

		b3Shape* shapeA = b3Array_Get( world->shapes, shapeIdA );
		b3Shape* shapeB = b3Array_Get( world->shapes, shapeIdB );

		b3CreateContact( world, shapeA, shapeB, childIndex );
	}

	b3StackFree( alloc, pairKeys );

	b3StackFree( alloc, bp->movedNodes );
	bp->movedNodes = NULL;

	b3ValidateSolverSets( world );

	b3TracyCZoneEnd( create_contacts );
}

bool b3BroadPhase_TestOverlap( const b3BroadPhase* bp, int proxyKeyA, int proxyKeyB )
{
	int typeIndexA = B3_PROXY_TYPE( proxyKeyA );
	int proxyIdA = B3_PROXY_ID( proxyKeyA );
	int typeIndexB = B3_PROXY_TYPE( proxyKeyB );
	int proxyIdB = B3_PROXY_ID( proxyKeyB );

	b3AABB aabbA = b3DynamicTree_GetAABB( bp->trees + typeIndexA, proxyIdA );
	b3AABB aabbB = b3DynamicTree_GetAABB( bp->trees + typeIndexB, proxyIdB );
	return b3AABB_Overlaps( aabbA, aabbB );
}

int b3BroadPhase_GetShapeIndex( b3BroadPhase* bp, int proxyKey )
{
	int typeIndex = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );

	return (int)b3DynamicTree_GetUserData( bp->trees + typeIndex, proxyId );
}

void b3ValidateBroadPhase( const b3BroadPhase* bp )
{
	b3DynamicTree_Validate( bp->trees + b3_dynamicBody );
	b3DynamicTree_Validate( bp->trees + b3_kinematicBody );

	// todo validate every shape AABB is contained in tree AABB
}

void b3ValidateNoMoved( const b3BroadPhase* bp )
{
#if B3_ENABLE_VALIDATION == 1
	for ( int j = 0; j < b3_bodyTypeCount; ++j )
	{
		const b3DynamicTree* tree = bp->trees + j;
		b3DynamicTree_ValidateNoMoved( tree );
	}
#else
	B3_UNUSED( bp );
#endif
}
