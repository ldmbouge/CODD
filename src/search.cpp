#include "search.hpp"
#include "node.hpp"
#include <iostream>
#include <iomanip>

void BAndB::search()
{
   using namespace std;
   cout << "B&B searching..." << endl;
   AbstractDD::Ptr restricted = _theDD->duplicate();
   //restricted->setStrategy(new Restricted(_mxw));
   restricted->setStrategy(new Exact);
   restricted->compute();
}
