#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <set>
#include "store.hpp"
#include "vec.hpp"

template <class T> std::ostream& operator<<(std::ostream& os,const Vec<T>& v)
{
   os << "(" << v.size() << ")[ ";
   for(auto i=0u;i < v.size();i++)
      os << v[i] << ' ';
   return os << "\b]";
}

int main()
{
   Pool p;
   Vec<int> v(&p,32);
   for(int i=0;i < 32;i++)
      v.push_back(i);

   std::cout << v << std::endl;  
   return 0;
}
