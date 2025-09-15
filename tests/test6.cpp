#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdio>
#include "util.hpp"

int t0() {
   NatSet<2> s {0,32,33,40,45,55,74};
   NatSet<2> g {17,36,46,51,58,59,91};
   NatSet<2> x = 91 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "G= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t1() {
   NatSet<2> s {0,1,3,5};
   NatSet<2> g {115,117,119,120};
   NatSet<2> x = 120 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t2() {
   NatSet<2> s {115,117,119};
   NatSet<2> g {1,3,5};
   NatSet<2> x = 120 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}


int t3() {
   NatSet<4> s {0,1,3,5};
   NatSet<4> g {195,197,199,200};
   NatSet<4> x = 200 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t4() {
   NatSet<4> s {60,61,63};
   NatSet<4> g {140,139,137};
   NatSet<4> x = 200 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t5() {
   NatSet<2> s {0};
   NatSet<2> g {1};
   NatSet<2> x = 1 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t6() {
   NatSet<2> s {0,50};
   NatSet<2> g {14,64};
   NatSet<2> x = 64 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t7() {
   NatSet<3> s {0,98};
   NatSet<3> g {30,128};
   NatSet<3> x = 128 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t8() {
   NatSet<3> s {0,98};
   NatSet<3> g {64,162};
   NatSet<3> x = 162 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t9() {
   NatSet<3> s {0,98};
   NatSet<3> g {29,127};
   NatSet<3> x = 127 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t10() {
   NatSet<3> s {0,63};
   NatSet<3> g {64,127};
   NatSet<3> x = 127 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t11() {
   NatSet<3> s {0,127};
   NatSet<3> g {64,191};
   NatSet<3> x = 191 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t12() {
   NatSet<4> s {0,127};
   NatSet<4> g {65,192};
   NatSet<4> x = 192 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}

int t13() {
   NatSet<3> s {0,1,3,7,12,20,30,45,61,85,107,128};
   NatSet<3> g {21,42,64,88,104,119,129,137,142,146,148,149};
   NatSet<3> x = 149 - s;
   std::cout << "S= " << s << "\n";
   std::cout << "g= " << g << "\n";
   std::cout << "X= " << x << "\n";
   if (g != x) abort();
   return 0;
}


int main()
{
   t0();
   t1();
   t2();
   t3();
   t4();
   t5();
   t6();
   t7();
   t8();
   t9();
   t10();
   t11();
   t12();
   t13();
}
