#include "util.hpp"
#include <iostream>

class ANode {
public:
    // Internal Ptr type: pointer to Node
    using Ptr = ANode*;

    // Constructor to initialize the node with an integer value
    ANode(int value) : n(value) {}

    // Getter for the value
    int getValue() const {
        return n;
    }

private:
    int n;  // The integer value in the node

    // Overloading the ostream << operator
    friend std::ostream& operator<<(std::ostream& os, const ANode& node) {
        os << "Node{" << node.n << "}";
        return os;
    }
};
int main() {
    using namespace std;

    ANode tmp[3] = { ANode(1), ANode(2), ANode(3) };
    //vector<ANode::Ptr> nodes = {&tmp[0],&tmp[1],&tmp[2]};
    vector<ANode::Ptr> nodes = {};

    auto begin = nodes.begin();
    for (auto it = nodes.end() - 1; it >= nodes.begin(); --it) {
        auto n = *it;
        cout << *n << endl;

        if (n->getValue() == 3) {
            it = nodes.erase(it-1);
        }
    }

}