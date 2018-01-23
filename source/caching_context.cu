#include "caching_context.hxx"

cub::CachingDeviceAllocator caching_context_t::allocator(8, 3, 13);