#ifdef NEURIPLO_INFER_WITH_KSERVE

#include "KserveProtocol.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <unistd.h>

namespace {

// RAII helper: writes content to a unique temp file and removes it on scope
// exit, so the file-sourcing tests do not leak files.
class TempFile {
public:
  explicit TempFile(const std::string &content) {
    path_ = std::filesystem::temp_directory_path() /
            ("kserve_secret_" + std::to_string(::getpid()) + "_" +
             std::to_string(counter()) + ".tmp");
    std::ofstream out(path_, std::ios::binary);
    out << content;
  }
  ~TempFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  TempFile(const TempFile &) = delete;
  TempFile &operator=(const TempFile &) = delete;

  std::string path() const { return path_.string(); }

private:
  static int counter() {
    static int n = 0;
    return ++n;
  }
  std::filesystem::path path_;
};

} // namespace

TEST(KserveSecurity, TrimTrailingWhitespace) {
  EXPECT_EQ(kserve::trimTrailingWhitespace("token\n"), "token");
  EXPECT_EQ(kserve::trimTrailingWhitespace("token\r\n"), "token");
  EXPECT_EQ(kserve::trimTrailingWhitespace("token \t\r\n"), "token");
  EXPECT_EQ(kserve::trimTrailingWhitespace("token"), "token");
  EXPECT_EQ(kserve::trimTrailingWhitespace(""), "");
  EXPECT_EQ(kserve::trimTrailingWhitespace("   "), "");
  // Leading/internal whitespace is preserved; only trailing is trimmed.
  EXPECT_EQ(kserve::trimTrailingWhitespace("  a b \n"), "  a b");
}

TEST(KserveSecurity, ResolveSecretPrecedence) {
  const std::string env = "env-token";
  const std::string file = "file-token\n";

  // Env wins when non-empty, used verbatim (no trimming of env values).
  EXPECT_EQ(kserve::resolveSecret(&env, &file), "env-token");
  const std::string env_ws = "env-token\n";
  EXPECT_EQ(kserve::resolveSecret(&env_ws, &file), "env-token\n");

  // Empty env falls back to the (trimmed) file contents.
  const std::string empty;
  EXPECT_EQ(kserve::resolveSecret(&empty, &file), "file-token");

  // File only.
  EXPECT_EQ(kserve::resolveSecret(nullptr, &file), "file-token");

  // Neither set.
  EXPECT_EQ(kserve::resolveSecret(nullptr, nullptr), "");
  EXPECT_EQ(kserve::resolveSecret(&empty, nullptr), "");
}

TEST(KserveSecurity, RequireClientCertPair) {
  EXPECT_FALSE(kserve::requireClientCertPair("", ""));
  EXPECT_TRUE(kserve::requireClientCertPair("cert", "key"));
  EXPECT_THROW(kserve::requireClientCertPair("cert", ""), std::runtime_error);
  EXPECT_THROW(kserve::requireClientCertPair("", "key"), std::runtime_error);
}

TEST(KserveSecurity, ReadSecretFromEnvPrecedenceOverFile) {
  TempFile file("file-token\n");
  ::setenv("KSERVE_TEST_TOKEN", "env-token", 1);
  ::setenv("KSERVE_TEST_TOKEN_FILE", file.path().c_str(), 1);

  EXPECT_EQ(kserve::readSecretFromEnvOrFile("KSERVE_TEST_TOKEN",
                                            "KSERVE_TEST_TOKEN_FILE"),
            "env-token");

  ::unsetenv("KSERVE_TEST_TOKEN");
  ::unsetenv("KSERVE_TEST_TOKEN_FILE");
}

TEST(KserveSecurity, ReadSecretFromFileWhenEnvUnset) {
  TempFile file("file-token\n");
  ::unsetenv("KSERVE_TEST_TOKEN");
  ::setenv("KSERVE_TEST_TOKEN_FILE", file.path().c_str(), 1);

  // File is read and its trailing newline trimmed.
  EXPECT_EQ(kserve::readSecretFromEnvOrFile("KSERVE_TEST_TOKEN",
                                            "KSERVE_TEST_TOKEN_FILE"),
            "file-token");

  ::unsetenv("KSERVE_TEST_TOKEN_FILE");
}

TEST(KserveSecurity, ReadSecretEmptyWhenNeitherSet) {
  ::unsetenv("KSERVE_TEST_TOKEN");
  ::unsetenv("KSERVE_TEST_TOKEN_FILE");
  EXPECT_EQ(kserve::readSecretFromEnvOrFile("KSERVE_TEST_TOKEN",
                                            "KSERVE_TEST_TOKEN_FILE"),
            "");
}

TEST(KserveSecurity, ReadFileFromEnvPath) {
  TempFile file("-----BEGIN CERT-----\nabc\n-----END CERT-----\n");
  ::setenv("KSERVE_TEST_CA", file.path().c_str(), 1);
  EXPECT_EQ(kserve::readFileFromEnvPath("KSERVE_TEST_CA"),
            "-----BEGIN CERT-----\nabc\n-----END CERT-----\n");
  ::unsetenv("KSERVE_TEST_CA");

  // Unset env yields empty (no throw).
  EXPECT_EQ(kserve::readFileFromEnvPath("KSERVE_TEST_CA"), "");
}

TEST(KserveSecurity, EndpointTlsClassification) {
  EXPECT_TRUE(kserve::parseEndpoint("https://h:443", 80).tls);
  EXPECT_TRUE(kserve::parseEndpoint("grpcs://h:9000", 80).tls);
  EXPECT_FALSE(kserve::parseEndpoint("http://h:8080", 80).tls);
  EXPECT_FALSE(kserve::parseEndpoint("grpc://h:9000", 80).tls);
  // Scheme-less endpoints default to plaintext.
  EXPECT_FALSE(kserve::parseEndpoint("h:8080", 80).tls);
  EXPECT_FALSE(kserve::parseEndpoint("host", 80).tls);
  // Scheme matching is prefix-anchored; an https-looking host is not TLS.
  EXPECT_FALSE(kserve::parseEndpoint("httpsserver:80", 80).tls);
}

#endif // NEURIPLO_INFER_WITH_KSERVE
