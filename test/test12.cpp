#include <iostream>
#include <string>
#include <unordered_map>

struct MyHash {
    std::size_t operator()(const std::string& key) const {
        return 0;
    }
};

int main() {
    std::unordered_map<std::string, int, MyHash> m;

    m["abc"] = 1;
    m["def"] = 2;
    m["ghi"] = 3;

    auto at = *(m.find("abc"));
    std::cout << "find(abc) = (" << at.first << ", " << at.second << ")\n";

    return 0;
}
