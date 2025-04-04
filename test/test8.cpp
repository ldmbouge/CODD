#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdio>
#include "util.hpp"

void f1(Matrix<bool,2>& m)
{
   std::cout << m[2][2] << "\n";
}

int t0() {
   int n = 5;
   Matrix<bool,2> m(n,n);
   for(int i=0;i < n;i++) {
      for(int j=0;j < n;j++) {
         m[i][j] = false;
      }
   }
   f1(m);
   return 0;   
}


int main()
{
   t0();
}
