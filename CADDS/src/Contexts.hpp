#pragma once

#include <GFL.hpp>

enum HContext : gfl::i8 { BBCtx, DDCtx, DDInit };
enum DDContext : gfl::i8 { DDRelaxed, DDRestricted, DDExact };