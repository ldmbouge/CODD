#pragma once

#include <cxxopts.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

class CliManager {
  constexpr static long long int GB = 1024ll * 1024 * 1024;
  int _fragmentSize = -1;
  int _timeout = std::numeric_limits<int>::max();
  long long int _memSize = -1;
  std::string _instancePath;
  cxxopts::Options _options;

public:
  CliManager() : _options("CODD", "A GPU-accelerated solver for state-based combinatorial problems.") {
    _options.add_options("Available")
        ("h,help","Print usage")
        ("f,fragment", "Fragment size", cxxopts::value(_fragmentSize))
        ("m,memory", "Working memory in GB (required)", cxxopts::value(_memSize))
        ("i,instance", "Path to the instance file (required)", cxxopts::value(_instancePath))
        ("t,timeout", "Timeout in seconds", cxxopts::value(_timeout));
  }

  int fragmentSize() const { return _fragmentSize; }
  int timeout() const { return _timeout; }
  long long int memSize() const { return _memSize; }
  std::string const & instancePath() const { return _instancePath; }
  bool isFragmentSizeAuto() const { return _fragmentSize == -1; }
  bool isTimeoutSet() const { return _timeout != std::numeric_limits<int>::max(); }

  void parse(int argc, char * argv[]) {
    auto const result = _options.parse(argc, argv);

    if (result.count("help") > 0) {
      std::cout << _options.help() << std::endl;
      std::exit(EXIT_SUCCESS);
    }

    bool hasError = false;
    if (_instancePath.empty()) {
      std::cerr << "Error: Instance file is required\n";
      hasError = true;
    } else if (!std::filesystem::exists(_instancePath)) {
      std::cerr << "Error: Instance file does not exist: " << _instancePath << '\n';
      hasError = true;
    } else if (!std::filesystem::is_regular_file(_instancePath)) {
      std::cerr << "Error: Instance path is not a regular file: " << _instancePath << '\n';
      hasError = true;
    } else if (std::ifstream f(_instancePath); !f.is_open()) {
      std::cerr << "Error: Instance file cannot be opened: " << _instancePath << '\n';
      hasError = true;
    }

    if (result.count("memory") == 0) {
      std::cerr << "Error: Memory is required\n";
      hasError = true;
    } else if (_memSize <= 0) {
      std::cerr << "Error: Memory must be greater than 0\n";
      hasError = true;
    } else {
      _memSize *= GB;
    }

    if (result.count("fragment-size") > 0 && _fragmentSize <= 0) {
      std::cerr << "Error: Fragment size must be greater than 0\n";
      hasError = true;
    }

    if (result.count("timeout") > 0 && _timeout <= 0) {
      std::cerr << "Error: Timeout must be greater than 0\n";
      hasError = true;
    }

    if (hasError) {
      std::cerr << '\n' << _options.help() << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  void print(std::ostream & os = std::cout) const {
    os << "Instance: " << _instancePath << '\n'
       << "Fragment: " << (isFragmentSizeAuto() ? "Auto" : std::to_string(_fragmentSize)) << '\n'
       << "Memory: " << (_memSize / GB) << " GB\n"
       << "Timeout: " << (not isTimeoutSet() ? "None" : std::to_string(_timeout) + " s") << '\n';
  }
};
