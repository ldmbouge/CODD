#include "util.hpp"
#include <iostream>

int main() {
    std::cout << "[-4, 4] decreasing" << std::endl;
    auto o = decreasing(Range::close(-4,4));
    auto gen = makeDDGen(o);
    while(gen->more())
        std::cout << gen->getAndNext() << " ";
    std::cout << "\n";
}