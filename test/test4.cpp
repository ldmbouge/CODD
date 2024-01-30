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
   GNSet s0 {};
   GNSet s1 {1};
   GNSet s2 {2};
   GNSet s3 = s1 | s2;
   GNSet s4 {63};
   GNSet s5 = s3 | s4;
   GNSet s6 = {64};
   using namespace std;

   
   cout << s0 << endl;  
   cout << s1 << endl;  
   cout << s2 << endl;  
   cout << s3 << endl;  
   cout << s4 << endl;  
   cout << s5 << endl;
   cout << s6 << endl;  
   s6.remove(2);
   cout << s6 << endl;  
   return 0;
}
