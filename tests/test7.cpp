#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdio>
#include "util.hpp"

int t0() {
   GNSet s {0,32,33,40,45,55,74,80,110};
   std::cout << "S1= " << s << "\n";
   GNSet s2 = s;
   s2.removeBelow(41);  
   std::cout << "S2 " << s2 << "\n";
   return 0;
}

int t1() {
   GNSet s {0,32,33,40,45,55,74,80,110};
   std::cout << "S1= " << s << "\n";
   GNSet s2 = s;
   s2.removeBelow(40);  
   std::cout << "S2 " << s2 << "\n";
   return 0;
}

int t2() {
   GNSet s {0,32,33,40,45,55,74,80,110};
   std::cout << "S1= " << s << "\n";
   GNSet s2 = s;
   s2.removeBelow(74);  
   std::cout << "S2 " << s2 << "\n";
   return 0;
}

int t3() {
   GNSet s {0,32,33,40,45,55,74,80,110};
   std::cout << "S1= " << s << "\n";
   GNSet s2 = s;
   s2.removeBelow(75);  
   std::cout << "S2 " << s2 << "\n";
   return 0;
}

int t4() {
   GNSet s {0,32,33,40,45,55,74,80,110};
   std::cout << "S1= " << s << "\n";
   GNSet s2 = s;
   s2.removeBelow(0);  
   std::cout << "S2= " << s2 << "\n";
   return 0;
}

int main()
{
   t0();
   t1();
   t2();
   t3();
   t4();
}
