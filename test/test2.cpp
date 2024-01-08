#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <set>
#include "store.hpp"
#include "vec.hpp"

template <class T,typename TS> std::ostream& operator<<(std::ostream& os,const Vec<T,TS>& v)
{
   os << "(" << v.size() << ")[ ";
   for(auto i=0u;i < v.size();i++)
      os << v[i] << ' ';
   return os << "\b]";
}

int main()
{
   Pool p;
   Vec<int,unsigned> v(&p);
   for(int i=0;i < 32;i++)
      v[i] = i*2;

   std::cout << v << std::endl;  
   return 0;
}
