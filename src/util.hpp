#ifndef __UTIL_H
#define __UTIL_H

#include <iostream>
#include <vector>
#include <tuple>
#include <set>
#include <limits>

namespace std {
   template <class T> T min(const set<T>& s) {
      if (s.size() > 0)
         return *s.begin();
      else 
         return numeric_limits<T>::max();
   }
   template <class T> T max(const set<T>& s) {
      if (s.size() > 0)
         return *(s.rbegin());
      else 
         return numeric_limits<T>::min();
   }
   template <class T,typename Pred>
   set<T> filter(set<T> inSet,Pred p) {
      std::set<T> outSet = {};
      for(const auto& v : inSet)
         if (p(v))
            outSet.insert(v);
      return outSet;
   }
   template <class Container,class Pred>
   bool member(const Container& cont,Pred c) {
      for(const auto& v : cont)
         if (c(v)) return true;
      return false;
   }
};

template <class T> std::ostream& operator<<(std::ostream& os,const std::vector<T>& msg) {
   os << '[';
   for(int v : msg)
      os << v << " ";
   return os << ']';
}

template <class T> std::ostream& operator<<(std::ostream& os,const std::set<T>& s) {
   os << "{";
   auto cnt = 0u;
   for(auto i=s.begin();i!= s.end();i++,cnt++)
      os << *i << ((cnt==s.size()-1) ? "" : ",");
   return os << "}";
}

template<class T> struct std::hash<std::set<T>> {
   std::size_t operator()(const std::set<T>& v) const noexcept {
      std::size_t ttl = 0;
      for(auto e : v)
         ttl = (ttl << 3) | std::hash<T>{}(e);
      return ttl;
   }
};

template <typename T>
std::set<T> operator|(const std::set<T>& s1,const std::set<T>& s2)
{
   std::set<T> r = s1;
   r.insert(s2.begin(),s2.end());
   return r;
}

template <typename T>
std::set<T> operator&(const std::set<T>& s1,const std::set<T>& s2)
{
   std::set<T> r {};
   for(const auto& e : s1)
      if (s2.contains(e))
         r.insert(e);
   return r;
}


template <typename T>
std::set<T> remove(const std::set<T>& s1,const T& v) {
   std::set<T> r;
   for(T e : s1) {
      if (e == v) continue;
      r.insert(e);
   }
   return r;
}


#endif
