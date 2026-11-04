// Command-line parsing shared by the apps: only "--name value" pairs.
#pragma once

#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace ssolve_cli {

class Args {
 public:
  // Throws on an option outside `known`, or an option with no value.
  Args(int argc, char** argv, std::set<std::string> known) : known_(std::move(known)) {
    for (int i = 1; i < argc; ++i) {
      const std::string name = argv[i];
      if (name.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument: " + name);
      const std::string key = name.substr(2);
      if (known_.count(key) == 0) throw std::runtime_error("unknown option: " + name);
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
      values_[key] = argv[++i];
    }
  }

  bool has(const std::string& key) const { return values_.count(key) != 0; }

  std::string required(const std::string& key) const {
    const auto it = values_.find(key);
    if (it == values_.end()) throw std::runtime_error("missing required option --" + key);
    return it->second;
  }

  std::string get(const std::string& key, const std::string& fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
  }

  long long get_int(const std::string& key, long long fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    char* end = nullptr;
    const long long value = std::strtoll(it->second.c_str(), &end, 10);
    if (it->second.empty() || *end != '\0') throw std::runtime_error("--" + key + " needs an integer");
    return value;
  }

  double get_double(const std::string& key, double fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    char* end = nullptr;
    const double value = std::strtod(it->second.c_str(), &end);
    if (it->second.empty() || *end != '\0') throw std::runtime_error("--" + key + " needs a number");
    return value;
  }

 private:
  std::set<std::string> known_;
  std::map<std::string, std::string> values_;
};

}  // namespace ssolve_cli
