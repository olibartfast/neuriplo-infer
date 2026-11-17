#pragma once

#include <cctype>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace neuriplo_infer {

// App-local replacement for cv::CommandLineParser. The spec format is kept
// byte-compatible with the one the app already uses:
//
//   "{ name alias | default | help }"
//
// Every brace group declares one option: the first whitespace-separated token
// is the primary name, the rest are aliases, the second field is the default
// value and the third is the help text. Options are accepted as
// --name=value, --name value, and the same two forms with an alias (-n value).
// An option given without a value (--flag) reads as the boolean true.
class ArgParser {
  public:
    ArgParser(int argc, const char *const argv[], const std::string &keys);

    void about(const std::string &message);

    [[nodiscard]] bool has(const std::string &name) const;

    // T may be std::string, bool, int, or float/double. A malformed numeric
    // or boolean value is recorded and reported by check(), matching the
    // cv::CommandLineParser behaviour the call sites were written against.
    template <typename T> T get(const std::string &name);

    [[nodiscard]] bool check() const;
    void printErrors() const;
    void printMessage() const;

  private:
    struct Option {
        std::string default_value;
        std::string help;
    };

    [[nodiscard]] std::string rawValue(const std::string &name) const;
    [[nodiscard]] bool markError(const std::string &message);
    void parseKeys(const std::string &keys);

    std::map<std::string, Option> options_; // primary name -> option
    std::map<std::string, std::string> aliases_; // any name -> primary name
    std::set<std::string> provided_; // primary names seen on the command line
    std::map<std::string, std::string> values_; // primary name -> raw value
    std::vector<std::string> errors_;
    std::string about_;
};

template <typename T> T ArgParser::get(const std::string &name) {
    const std::string raw = rawValue(name);

    if constexpr (std::is_same_v<T, std::string>) {
        return raw;
    } else if constexpr (std::is_same_v<T, bool>) {
        if (raw.empty()) {
            return has(name);
        }
        std::string lowered;
        lowered.reserve(raw.size());
        for (const char c : raw) {
            lowered.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(c))));
        }
        if (lowered == "true" || lowered == "1" || lowered == "yes" ||
            lowered == "on") {
            return true;
        }
        if (lowered == "false" || lowered == "0" || lowered == "no" ||
            lowered == "off") {
            return false;
        }
        markError("Invalid boolean value for --" + name + ": '" + raw + "'");
        return false;
    } else if constexpr (std::is_same_v<T, int>) {
        try {
            std::size_t consumed = 0;
            const int parsed = std::stoi(raw, &consumed);
            if (consumed != raw.size()) {
                throw std::invalid_argument("trailing characters");
            }
            return parsed;
        } catch (const std::exception &) {
            markError("Invalid integer value for --" + name + ": '" + raw +
                      "'");
            return 0;
        }
    } else if constexpr (std::is_same_v<T, float>) {
        try {
            std::size_t consumed = 0;
            const float parsed = std::stof(raw, &consumed);
            if (consumed != raw.size()) {
                throw std::invalid_argument("trailing characters");
            }
            return parsed;
        } catch (const std::exception &) {
            markError("Invalid float value for --" + name + ": '" + raw + "'");
            return 0.0F;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        try {
            std::size_t consumed = 0;
            const double parsed = std::stod(raw, &consumed);
            if (consumed != raw.size()) {
                throw std::invalid_argument("trailing characters");
            }
            return parsed;
        } catch (const std::exception &) {
            markError("Invalid number value for --" + name + ": '" + raw +
                      "'");
            return 0.0;
        }
    } else {
        static_assert(sizeof(T) == 0, "ArgParser::get supports string, bool, "
                                      "int, float, and double");
    }
}

} // namespace neuriplo_infer
