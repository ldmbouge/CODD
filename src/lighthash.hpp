/*
 * ddOpt is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * ddOpt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2023. by Laurent Michel.
 *
 * Contributions by Waldemar Cruz, Rebecca Gentzel, Willem Jan Van Hoeve
 */

#ifndef __DDOPT_LIGHTHASHTABLE_H
#define __DDOPT_LIGHTHASHTABLE_H

#include <algorithm>
#include <iostream>
#include <functional>
#include <string>
#include <limits>
#include <string.h>
#include "store.hpp"
#include "node.hpp"

template <class ST,class Hash = std::hash<ST>,class Equal = std::equal_to<ST>> class LHashtable {
   struct HTNode {
      Node<ST>* _data;
      HTNode*   _next;
   };
   static constexpr const std::size_t _primes[] = {
      2,547,1229,1993,2749,3581,4421,5281,6143,7001,7927,8837,9739,10663,11677,12569,13513,14533,15413,16411,
      17393,18329,19427,20359,21391,22343,23327,24317,25409,26407,27457,28513,29453,30577,31607,32611,33617,        
      34651,35771,36787,37831,38923,39979,41113,42083,43063,44203,45317,46451,47533,48619,49667,50767,51817,
      52937,54001,55109,56197,57193,58243,59369,60509,61637,62791,63823,65071,66107,67247,68389,69497,70663,
      71719,72859,73999,75083,76213,77359,78487,79627,80737,81817,82903,84131,85243,86381,87557,88807,89867,
      90989,92177,93187,94351,95443,96587,97829,98953,100129,101287,102329,103613,104743,105953,107053,108271,
      109397,110603,111791,112997,114143,115327,116461,117673,118831,119993,121139,122263,123449,124601,125753,
      126989,128201,129287,130457,131707,132857,134059,135281,136393,137483,138647,139907,141067,142183,143503,
      144737,145933,147137,148361,149423,150589,151717,152897,154127,155333,156601,157747,159023,160183,161407,
      162671,163847,165083,166357,167483,168869,170047,171179,172399,173651,174799,176087,177211,178489,179581,
      180751,182089,183289,184477,185651,186761,187973,189253,190471,191677,192853,194017,195311,196561,197759,
      198943,200191,201511,202717,203971,205213,206383,207569,208739,209939,
      467237,467239,467261,467293,467297,467317,467329,467333,467353,467371,467399,467417,467431,467437,467447,
      467471,467473,467477,467479,467491,467497,467503,467507,467527,467531,467543,467549,467557,467587,467591,
      467611,467617,467627,467629,467633,467641,467651,467657,467669,467671,467681,467689,467699,467713,467729,
      467737,467743,467749,467773,467783,467813,467827,467833,467867,467869,467879,467881,467893,467897,467899,
      467903,467927,467941,467953,467963,467977,468001,468011,
      609373,609379,609391,609397,609403,609407,609421,609437,609443,609461,609487,609503,609509,609517,609527,
      609533,609541,609571,609589,609593,609599,609601,609607,609613,609617,609619,609641,609673,609683,609701,
      609709,609743,609751,609757,609779,609781,609803,609809,609821,609859,609877,609887,609907,609911,609913,
      609923,609929,609979,609989,609991,609997,610031,610063,610081,610123,610157,610163,610187,610193,610199,
      610217,610219,610229,610243,610271,610279,610289,610301,610327,610331,610339,610391,610409,610417,610429,
      610439,610447,610457,610469,610501,610523,610541,610543,610553,610559,610567,610579,610583,610619,610633,
      610639,610651,610661,610667,610681,610699,610703,610721,610733,610739,610741,610763,610781,610783,610787,
      610801,610817,610823,610829,610837,610843,610847,610849,610867,610877,610879,610891,610913,610919,610921,
      610933,610957,610969,610993,611011,611027,611033,611057,611069,611071,611081,611101,611111,611113,611131,
      611137,611147,611189,611207,611213,611257,611263,611279,611293,611297,611323,611333,611389,611393,611411,
      611419,611441,611449,611453,611459,611467,611483,611497,611531,611543,611549,611551,611557,611561,611587,
      611603,611621,611641,611657,611671,611693,611707,611729,611753,611791,611801,611803,611827,611833,611837,
      611839,611873,611879,611887,611903,611921,611927,611939,611951,611953
   };
   Pool::Ptr _pool;
   HTNode**  _tab;
   unsigned* _mgc;
   std::size_t  _mxs;
   unsigned _magic;
   unsigned _nbp;   // number of pairs
   std::size_t primeSize(std::size_t sz) {
      constexpr const std::size_t tsz = sizeof(_primes)/sizeof(std::size_t);
      std::size_t low=0,up = tsz - 1;
      while (low <= up) {
         int m = (low + up)/2;
         if (sz < _primes[m])
            up = m - 1;
         else if (sz > _primes[m])
            low = m + 1;
         else {
            low = up = m;
            break;
         }
      }
      if (low < tsz)
         return _primes[low];
      else {
         auto last = _primes[tsz-1];
         while (last < sz)
            last *= 3;
         return last;
      }
      //return low >= tsz ? _primes[tsz-1] :  _primes[low];
   }
public:
   LHashtable(Pool::Ptr p,std::size_t sz) : _pool(p) {
      _mxs = primeSize(sz);
      std::cout << "sz(LightHashtable):" << _mxs << '\n';
      _tab = new (_pool) HTNode*[_mxs];
      _mgc = new (_pool) unsigned[_mxs];
      memset(_tab,0,sizeof(HTNode*)*_mxs);
      memset(_mgc,0,sizeof(unsigned)*_mxs);
      _magic = 1;
      _nbp = 0;
   }
   void resize(std::size_t sz) {
      auto newSZ = primeSize(sz);
      std::cout << "sz(LightHashtable):" << newSZ << '\n';
      auto newtab = new (_pool) HTNode*[newSZ];
      auto newmgc = new (_pool) unsigned[newSZ];
      memset(_tab,0,sizeof(HTNode*)*newSZ);
      memset(_mgc,0,sizeof(unsigned)*newSZ);
      if (_nbp > 0) {
         for(auto i=0u;i <_mxs;i++) {
            HTNode* cur = (_mgc[i]==_magic) ? _tab[i] : nullptr;
            while (cur) {
               auto nextOne = cur->_next;
               std::size_t at = Hash{}(cur->_data->get()) % newSZ;
               cur->_next = newtab[at];
               newtab[at] = cur;
               newmgc[at] = _magic;
               cur = nextOne;
            }
         }
      }
      _tab = newtab;
      _mgc = newmgc;
      _mxs = newSZ;          
   }
   class HTAt {
      friend class LHashtable<ST,Hash,Equal>;
      std::size_t _at;
      bool       _inc; // true if query is in the hashtable
      HTAt(std::size_t at,bool inc) : _at(at),_inc(inc) {}
   public:
      operator bool() const noexcept { return _inc;}
   };
   HTAt getLoc(const ST& key,Node<ST>*& val) const noexcept {
      std::size_t at = Hash{}(key) % _mxs;
      assert(at >= 0);
      assert(at < _mxs);
      HTNode* cur =  (_mgc[at]==_magic) ? _tab[at] : nullptr;
      while (cur != nullptr) {
         //std::cout << "Equal{}(" << cur->_data->get() << ", "<< key << ")" << std::endl; 
         if (Equal{}(cur->_data->get(),key)) {
            val = cur->_data;
            return HTAt(at, true);
         }
         cur = cur->_next;
      }
      return HTAt(at, false);      
   }
   void rawInsertAt(const HTAt& loc,Node<ST>* val) {
      HTNode* head = (_mgc[loc._at]==_magic) ? _tab[loc._at] : nullptr;
      _tab[loc._at] = new (_pool) HTNode {val,head};
      _mgc[loc._at] = _magic;
      ++_nbp;            
   }
   void safeInsertAt(const HTAt& loc,Node<ST>* val) {
      assert(loc._inc == false);
      HTNode* head = (_mgc[loc._at]==_magic) ? _tab[loc._at] : nullptr;
      _tab[loc._at] = new (_pool) HTNode {val,head};
      _mgc[loc._at] = _magic;
      ++_nbp;      
   }
   std::size_t maxSize() const noexcept { return _mxs;}
   unsigned size() const noexcept { return _nbp;}
   void clear() noexcept {
      /*      std::cout << "LHT(" << this << ") had: " << _nbp << " / " << _mxs << " entries\n";
      unsigned lc = std::numeric_limits<unsigned>::min();
      unsigned nec = 0;
      std::size_t idxll = 0;
      double ssq = 0;
      double s   = 0;
      for(auto i=0u;i <_mxs;i++) {
         HTNode* cur = (_mgc[i]==_magic) ? _tab[i] : nullptr;
         nec += (cur != nullptr);
         unsigned nn = 0;
         while (cur) {
            ++nn;
            cur = cur->_next;
         }
         ssq += (nn*nn);
         s   += nn;
         lc =  std::max(lc,nn);
         if (nn == lc)
            idxll = i;
      }
      double asq = ssq / _mxs;
      double as  = s / _mxs;
      double dev = sqrt(asq - (as*as));
      std::cout << "\tL(chain)    = " << lc << " IDX:" << idxll << "\n";
      std::cout << "\t#NE(chains) = " << nec << "\n";
      std::cout << "\t#E(chains)  = " << _mxs - nec << "\n";
      std::cout << "\tDEV         = " << dev << "\n";
      HTNode* cur = (_mgc[idxll]==_magic) ? _tab[idxll] : nullptr;
      while(cur) {
         std::cout << cur->_data->get() << "\n";
         cur = cur->_next;
      }
      */
      ++_magic;
      _nbp = 0;
   }
   template <class Fun>
   void doOnAll(Fun f) {
      for(auto i=0u;i < _mxs;i++) {
         HTNode* cur = (_mgc[i]==_magic) ? _tab[i] : nullptr;
         while (cur) {
            f(cur->_data);
            cur = cur->_next;
         }
      }
   }
};

#endif
