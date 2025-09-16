#ifndef __MSORT_HPP__
#define __MSORT_HPP__

#include <stdlib.h>
#include <string.h>
#include <cstdio>

#include <iostream>

template <typename T,typename F> void mergeSortAux(T* v,
                                                   const std::size_t from,
                                                   const std::size_t to,
                                                   const F& cmp,
                                                   T* aux)
{
   // [from, to )  
   if (from + 1 >= to) {
      return;
   } else {
      auto d = from + (to - from)/2;
      mergeSortAux(aux,from,d,cmp,v);
      mergeSortAux(aux,d,to,cmp,v);
      auto i = from,j = d;
      T* k = v + from;
      while (i != d && j != to) {
         if (cmp(aux[i],aux[j]))
            *k++ = aux[i++];
         else *k++ = aux[j++];
      }
      while (i != d)  *k++ = aux[i++];
      while (j != to) *k++ = aux[j++];
   }
}

template <typename T,typename F> void mergeSortPermAux(T* v,
                                                       int* perm,
                                                       const std::size_t from,
                                                       const std::size_t to,
                                                       const F& cmp,
                                                       T* aux,
                                                       int* paux)
{
   // [from, to )  
   if (from + 1 >= to) {
      return;
   } else {
      auto d = from + (to - from)/2;
      mergeSortPermAux(aux,paux,from,d,cmp,v,perm);
      mergeSortPermAux(aux,paux,d,to,cmp,v,perm);
      auto i = from,j = d;
      T* k = v + from;
      int* pk = perm + from;
      while (i != d && j != to) {
         if (cmp(aux[i],aux[j])) {
            *k++  = aux[i];
            *pk++ = paux[i++];
         } else {
            *k++  = aux[j];
            *pk++ = paux[j++];
         }
      }
      while (i != d) {
         *k++  = aux[i];
         *pk++ = paux[i++];
      }
      while (j != to) {
         *k++  = aux[j];
         *pk++ = paux[j++];
      }
   }
}

template <typename T,typename F>
void mergeSort(T* c,std::size_t sz,const F& cmp)
{
   T* aux = (T*)alloca(sizeof(T)*sz);
   for(auto i=0u;i < sz;i++) aux[i] = c[i];
   mergeSortAux(c,0,sz,cmp,aux);
}

template <typename T,typename F>
void mergeSortPerm(T* c,int* perm,std::size_t sz,const F& cmp)
{
   T* aux = (T*)alloca(sizeof(T)*sz);
   int* paux = (int*)alloca(sizeof(int)*sz);
   for(auto i=0u;i < sz;i++) {
      aux[i] = c[i];
      paux[i] = perm[i];
   }
   mergeSortPermAux(c,perm,0,sz,cmp,aux,paux);
   // for(auto i=0u;i < sz;i++) {
   //    std::cout << i << " " << aux[i] << " " << perm[i] << " " << c[i] << "\n";
   // }
}

#endif
