#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <set>
#include "store.hpp"
#include "vec.hpp"
#include "util.hpp"


int main()
{
   Pool p;
   NatSet<1> s0 {};
   NatSet<1> s1 {1};
   NatSet<1> s2 {2};
   NatSet<1> s3 = s1 | s2;
   NatSet<1> s4 {63};
   NatSet<1> s5 = s3 | s4;
   NatSet<2> s6 = {64};
   using namespace std;

   
   cout << s0 << endl;  
   cout << s1 << endl;  
   cout << s2 << endl;  
   cout << s3 << endl;  
   cout << s4 << endl;  
   cout << s5 << endl;
   cout << s6 << endl;  
   return 0;
}
