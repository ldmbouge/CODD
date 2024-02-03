#ifndef __MSORT_HPP__
#define __MSORT_HPP__

#include <stdlib.h>

template <typename T,typename F> void mergeSortAux(T* v,
                                                   std::size_t from,
                                                   std::size_t to,
                                                   const F& cmp,
                                                   T* aux)
{
   // [from, to )  
   if (from + 1 == to) {
      return;
   } else {
      auto d = from + (to - from)/2;
      mergeSortAux(v,from,d,cmp,aux);
      mergeSortAux(v,d,to,cmp,aux);
      auto i = from,j = d;
      auto a1 = aux - from,a2 = aux;
      for(;i < d;i++) a1[i] = v[i];
      for(;j < to;j++) a2[j] = v[j];
      i = from,j = d;
      T* k = v + from;
      while (i != d && j != to) {
         if (cmp(a1[i],a2[j]))
            *k++ = a1[i++];
         else *k++ = a2[j++];
      }
      while (i != d)  *k++ = a1[i++];
      while (j != to) *k++ = a2[j++];
   }
}

template <typename T,typename F>
void mergeSort(T* c,std::size_t sz,const F& cmp)
{
   T* aux = new T[sz];
   mergeSortAux(c,0,sz,cmp,aux);
   delete[] aux;
}

#endif
