#pragma once

#include <cxxopts.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

class CliManager
{

    int width_{0};
    int timeout_{std::numeric_limits<int>::max()};
    int toPop_{1};
    double lambda {0.0};
    long long int memSize_{0};
    std::string instance_{};
    cxxopts::Options options_;

public:
    CliManager(std::string const& programName, std::string const& description);
    void parse(int argc, char* argv[]);

    int width() const { return width_; }
    int timeout() const { return timeout_; }
    int pop() const { return toPop_; }
    long long int memSize() const { return memSize_ * 1024ll * 1024ll * 1024ll; }
    std::string const& instance() const { return instance_; }

private:
    void validate();
};

inline
CliManager::CliManager(std::string const& programName, std::string const& description) :
      options_(programName, description)
{
    options_.add_options("Available")
        ("w,width", "DD width", cxxopts::value(width_))
        ("h,help", "Show this help message and exit")
        ("m,memory", "Working memory in GB", cxxopts::value(memSize_))
        ("p,pop", "Max nodes to pop in parallel ", cxxopts::value(toPop_))
        ("i,instance", "Path to the instance file", cxxopts::value(instance_))
        ("l,lambda", "Favour exact nodes", cxxopts::value(lambda))
        ("t,timeout", "Timeout in seconds", cxxopts::value(timeout_));

    options_.parse_positional({"instance"});
    options_.custom_help("<OPTIONS>");
    options_.positional_help("<INSTANCE>");
}

inline
void CliManager::parse(int argc, char* argv[])
{
    auto const result = options_.parse(argc, argv);

    if (result.count("help") > 0)
    {
        std::cout << options_.help() << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    validate();
}

inline
void CliManager::validate()
{
    bool hasError = false;

    if (instance_.empty())
    {
        std::cerr << "Error: Instance file is required" << std::endl;
        hasError = true;
    }
    else if (!std::filesystem::exists(instance_))
    {
        std::cerr << "Error: Instance file does not exist: " << instance_ << std::endl;
        hasError = true;
    }
    else if (!std::filesystem::is_regular_file(instance_))
    {
        std::cerr << "Error: Instance path is not a regular file: " << instance_ << std::endl;
        hasError = true;
    }
    else
    {
        std::ifstream f(instance_);
        if (!f.is_open())
        {
            std::cerr << "Error: Instance file cannot be opened: " << instance_ << std::endl;
            hasError = true;
        }
    }

    if (width_ <= 0)
    {
        std::cerr << "Error: Width must be greater than 0" << std::endl;
        hasError = true;
    }
    if (toPop_ <= 0)
    {
        std::cerr << "Error: Pop must be greater than 0" << std::endl;
        hasError = true;
    }

    if (timeout_ <= 0)
    {
        std::cerr << "Error: Timeout must be greater than 0" << std::endl;
        hasError = true;
    }

    if (memSize_ <= 0)
    {
        std::cerr << "Error: Memory must be greater than 0" << std::endl;
        hasError = true;
    }

    if (hasError)
    {
        std::cerr << "\n" << options_.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}