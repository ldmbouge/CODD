#include "pool.hpp"

void LPool::release(ANode::Ptr n)
{
   _free.push(n);
}
