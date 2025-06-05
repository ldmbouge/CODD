#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdio>
#include "util.hpp"


int main()
{
   std::cout << "[-4, 4]  increasing  v->v*v" << std::endl;
   for(auto i : increasing(Range::close(-4,4),[](int v) {return v*v;})) 
      std::cout << i << " ";
   std::cout << "\n";

   std::cout << "[-4, 4]  decreasing  v -> v*v" << std::endl;
   for(auto i : decreasing(Range::close(-4,4),[](int v) { return v*v;}))
      std::cout << i << " ";
   std::cout << "\n";

   std::cout << "[-4, 4]" << std::endl;
   for(auto i : Range::close(-4,4))
      std::cout << i << " ";
   std::cout << "\n";

   std::cout << "[-4, 4] increasing" << std::endl;
   for(auto i : increasing(Range::close(-4,4)))
      std::cout << i << " ";
   std::cout << "\n";

   std::cout << "[-4, 4] decreasing" << std::endl;
   for(auto i : decreasing(Range::close(-4,4)))
      std::cout << i << " ";
   std::cout << "\n";

   std::cout  << "vectors = ----------------------------------------------------------------------\n";
   std::cout << "[1,2,3,4,5,10,9,8,7]" << std::endl;   
   for(auto i : std::vector<int> {1,2,3,4,5,10,9,8,7})
      std::cout << i << " ";
   std::cout << "\n";
   
   std::cout << "[1,2,3,4,5,10,9,8,7] increasing" << std::endl;   
   for(auto i : increasing(std::vector<int> {1,2,3,4,5,10,9,8,7}))
      std::cout << i << " ";
   std::cout << "\n";

   std::cout << "[1,2,3,4,5,10,9,8,7] decreasing" << std::endl;   
   for(auto i : decreasing(std::vector<int> {1,2,3,4,5,10,9,8,7}))
      std::cout << i << " ";
   std::cout << "\n";
   
   std::cout << "[1,2,3,4,5,10,9,8,7] increasing i -> i" << std::endl;   
   for(auto i : increasing(std::vector<int> {1,2,3,4,5,10,9,8,7},[](auto i) { return i;}))
      std::cout << i << " ";
   std::cout << "\n";

   std::cout << "[1,2,3,4,5,10,9,8,7] increasing i -> -i" << std::endl;   
   for(auto i : increasing(std::vector<int> {1,2,3,4,5,10,9,8,7},[](auto i) { return -i;}))
      std::cout << i << " ";
   std::cout << "\n";

   std::cout << "[1,2,3,4,5,10,9,8,7] decreasing i -> i" << std::endl;   
   for(auto i : decreasing(std::vector<int> {1,2,3,4,5,10,9,8,7},[](auto i) { return i;}))
      std::cout << i << " ";
   std::cout << "\n";

   std::cout << "[1,2,3,4,5,10,9,8,7] decreasing i -> -i" << std::endl;   
   for(auto i : decreasing(std::vector<int> {1,2,3,4,5,10,9,8,7},[](auto i) { return -i;}))
      std::cout << i << " ";
   std::cout << "\n";

   std::cout << "[1..10] increasing i -> |6-i|" << std::endl;   
   for(auto i : increasing(std::vector<int> {1,2,3,4,5,6,7,8,9,10},[](auto i) { return abs(6-i);}))
      std::cout << i << " ";
   std::cout << "\n";

   std::cout << "GNSet: \n";
   auto s0 = GNSet {1,7,4,12,253};
   std::cout << s0 << "\n";
   std::cout << "forward loop:\t";
   for(auto i = s0.begin(); i!=s0.end();i++)
      std::cout << *i << " ";
   std::cout << "\n";
   std::cout << "backward loop:\t";
   auto s0E = s0.end();
   auto s0B = s0.begin();
   s0E = s0E - 1;
   s0B = s0B - 1;
   for(auto i = s0E; i!=s0B;i--)
      std::cout << *i << " ";
   std::cout << "\n";

   std::cout << "GNSet: \n";
   auto s1 = GNSet {};
   srand(time(NULL));
   for(int i = 0; i < 100; i++) s1.insert(rand() % 1000);
   std::cout << s1 << "\n";
   std::cout << "forward loop:\t";
   for(auto i = s1.begin(); i!=s1.end();i++)
      std::cout << *i << " ";
   std::cout << "\n";
   std::cout << "backward loop:\t";
   auto s1E = s1.end();
   auto s1B = s1.begin();
   s1E = s1E - 1;
   s1B = s1B - 1;
   for(auto i = s1E; i!=s1B;i--)
      std::cout << *i << " ";
   std::cout << "\n";

   std::cout << "NatSet: \n";
   auto s2 = NatSet<16> {};
   for(int i = 0; i < 100; i++) s2.insert(rand() % 1024);
   std::cout << s2 << "\n";
   std::cout << "forward loop:\t";
   for(auto i = s2.begin(); i!=s2.end();i++)
      std::cout << *i << " ";
   std::cout << "\n";
   std::cout << "backward loop:\t";
   auto s2E = s2.end();
   auto s2B = s2.begin();
   s2E--;
   s2B--;
   for(auto i = s2E; i!=s2B;i--)
      std::cout << *i << " ";
   std::cout << "\n";

   return 0;
}
