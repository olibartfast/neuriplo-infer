#include "ArgParser.hpp"

#include <iostream>
#include <sstream>

namespace neuriplo_infer {

namespace {

std::string trim(const std::string &value) {
  const std::size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const std::size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

} // namespace

ArgParser::ArgParser(int argc, const char *const argv[],
                     const std::string &keys) {
  parseKeys(keys);

  for (int i = 1; i < argc; ++i) {
    if (argv[i] == nullptr) {
      continue;
    }
    const std::string arg = argv[i];
    if (arg.size() < 2 || arg[0] != '-') {
      continue;
    }
    const std::size_t start = arg.find_first_not_of('-');
    if (start == std::string::npos) {
      continue;
    }

    const std::string body = arg.substr(start);
    std::string key;
    std::string value;
    bool has_value = false;

    const std::size_t equals = body.find('=');
    if (equals != std::string::npos) {
      key = body.substr(0, equals);
      value = body.substr(equals + 1);
      has_value = true;
    } else {
      key = body;
    }

    const auto alias = aliases_.find(key);
    if (alias == aliases_.end()) {
      // Unknown options are diagnosed by the caller, which already knows
      // every accepted key from the spec string.
      continue;
    }
    const std::string &primary = alias->second;

    if (!has_value && i + 1 < argc && argv[i + 1] != nullptr) {
      const std::string next = argv[i + 1];
      if (next.empty() || next[0] != '-') {
        value = next;
        has_value = true;
        ++i;
      }
    }

    provided_.insert(primary);
    values_[primary] = value;
  }
}

void ArgParser::about(const std::string &message) { about_ = message; }

bool ArgParser::has(const std::string &name) const {
  const auto alias = aliases_.find(name);
  const std::string primary =
      alias == aliases_.end() ? name : alias->second;
  return provided_.count(primary) > 0;
}

bool ArgParser::check() const { return errors_.empty(); }

void ArgParser::printErrors() const {
  for (const auto &error : errors_) {
    std::cerr << error << '\n';
  }
}

void ArgParser::printMessage() const {
  std::cout << about_ << "\n\n";
  std::cout << "Usage: <program> [options]\n\n";
  for (const auto &[primary, option] : options_) {
    std::cout << "  --" << primary;
    for (const auto &[alias, target] : aliases_) {
      if (target == primary && alias != primary) {
        std::cout << ", -" << alias;
      }
    }
    if (!option.default_value.empty()) {
      std::cout << " (default: " << option.default_value << ")";
    }
    std::cout << '\n';
    if (!option.help.empty()) {
      std::cout << "      " << option.help << '\n';
    }
  }
}

std::string ArgParser::rawValue(const std::string &name) const {
  const auto alias = aliases_.find(name);
  const std::string primary =
      alias == aliases_.end() ? name : alias->second;

  const auto value = values_.find(primary);
  if (value != values_.end()) {
    return value->second;
  }
  const auto option = options_.find(primary);
  if (option != options_.end()) {
    return option->second.default_value;
  }
  return {};
}

bool ArgParser::markError(const std::string &message) {
  errors_.push_back(message);
  return false;
}

void ArgParser::parseKeys(const std::string &keys) {
  std::size_t open = keys.find('{');
  while (open != std::string::npos) {
    const std::size_t close = keys.find('}', open);
    if (close == std::string::npos) {
      break;
    }

    std::string entry = keys.substr(open + 1, close - open - 1);
    std::vector<std::string> fields;
    std::size_t bar = entry.find('|');
    while (bar != std::string::npos) {
      fields.push_back(entry.substr(0, bar));
      entry.erase(0, bar + 1);
      bar = entry.find('|');
    }
    fields.push_back(entry);
    while (fields.size() < 3) {
      fields.emplace_back();
    }

    std::istringstream names(fields[0]);
    std::string primary;
    std::string name;
    while (names >> name) {
      if (primary.empty()) {
        primary = name;
        options_[primary] = Option{trim(fields[1]), trim(fields[2])};
      }
      aliases_[name] = primary;
    }

    open = keys.find('{', close);
  }
}

} // namespace neuriplo_infer
