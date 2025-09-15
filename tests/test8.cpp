#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdio>
#include "util.hpp"

void f1(Matrix<bool,2>& m)
{
   std::cout << m[2][2] << "\n";
}

void f2(const Matrix<bool,2>& m)
{
   std::cout << m[2][3] << "\n";
   //m[2][4] = true;
}

int t0() {
   int n = 5;
   Matrix<bool,2> m(n,n);
   for(int i=0;i < n;i++) {
      for(int j=0;j < n;j++) {
         m[i][j] = false;
      }
   }
   m[2][2] = true;
   m[2][3] = true;
   f1(m);
   f2(m);
   std::cout << m << "\n";
   return 0;   
}


int main()
{
   t0();
}
