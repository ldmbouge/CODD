#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdio>
#include "util.hpp"

int main() {
   NatSet<2> s {0,32,33,40,45,55,74};
   NatSet<2> x = 91 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "X= " << x << "\n";
   return 0;
}
