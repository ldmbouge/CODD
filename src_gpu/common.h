#pragma once

template<typename State, typename Labels>
struct LightNode
{
    State state;
    Labels labels;
    gfl::f64 boundSrcToNode;
    gfl::u8 nEdgesSrcToNode;
    gfl::u8 labelsSrcToNode[128];
};

struct NodeInfo
{
    gfl::u64 hash;
    gfl::f64 cost;
    gfl::f64 boundSrcToNode;
    gfl::i64 id;
    gfl::i32 idx;
    gfl::u32 isRepresented;
};