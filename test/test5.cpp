#include "msort.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdio>

int main() {
   /*
   int uv[] = {1, 30, -4, 3, 5, -4, 1, 6, -8, 2, -5, 64, 1, 92};
   mergeSort<int>(uv,sizeof(uv)/sizeof(int),[](int a,int b) {
      return a < b;
   });
   for(auto i=0;i < sizeof(uv)/sizeof(int);i++)
      printf("%d ",uv[i]);
   printf("\n");
   */
   int ov[] = {1,2,3};
   mergeSort<int>(ov,sizeof(ov)/sizeof(int),[](int a,int b) {
      return a < b;
   });
   for(auto i=0;i < sizeof(ov)/sizeof(int);i++)
      printf("%d ",ov[i]);
   return 0;
}
