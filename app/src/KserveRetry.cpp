#include "KserveRetry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>

namespace kserve {

namespace {

// Reads a positive integer env var, returning `fallback` when unset or invalid.
int envInt(const char *name, int fallback) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) {
    return fallback;
  }
  try {
    const int value = std::stoi(raw);
    return value > 0 ? value : fallback;
  } catch (...) {
    return fallback;
  }
}

// Reads a [0, 1] double env var, returning `fallback` when unset or invalid.
double envFraction(const char *name, double fallback) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) {
    return fallback;
  }
  try {
    const double value = std::stod(raw);
    if (value >= 0.0 && value <= 1.0) {
      return value;
    }
    return fallback;
  } catch (...) {
    return fallback;
  }
}

// KSERVE_MAX_RETRIES is expressed as the number of retries on top of the first
// attempt, matching operator intuition; this converts it to total attempts,
// falling back to `fallback` when unset or invalid.
int envAttempts(int fallback) {
  const char *raw = std::getenv("KSERVE_MAX_RETRIES");
  if (raw == nullptr) {
    return fallback;
  }
  try {
    const int retries = std::stoi(raw);
    return retries >= 0 ? retries + 1 : fallback;
  } catch (...) {
    return fallback;
  }
}

} // namespace

RetryPolicy retryPolicyFromEnv() {
  RetryPolicy policy;
  policy.max_attempts = envAttempts(policy.max_attempts);
  policy.base_delay_ms = envInt("KSERVE_RETRY_BASE_MS", policy.base_delay_ms);
  policy.max_delay_ms = envInt("KSERVE_RETRY_MAX_MS", policy.max_delay_ms);
  policy.jitter = envFraction("KSERVE_RETRY_JITTER", policy.jitter);
  return policy;
}

int backoffDelayMs(const RetryPolicy &policy, int attempt, double rand_unit) {
  if (attempt < 1) {
    return 0;
  }
  const double max_delay = static_cast<double>(policy.max_delay_ms);
  // Exponential term base * 2^(attempt-1), capped before applying jitter.
  double exp_delay =
      static_cast<double>(policy.base_delay_ms) * std::pow(2.0, attempt - 1);
  exp_delay = std::min(exp_delay, max_delay);
  // Spread by +/- jitter around the exponential term: rand_unit==0 -> lower
  // bound, 0.5 -> exact term, 1 -> upper bound.
  const double clamped_unit = std::clamp(rand_unit, 0.0, 1.0);
  const double factor =
      1.0 - policy.jitter + clamped_unit * 2.0 * policy.jitter;
  double jittered = exp_delay * factor;
  jittered = std::clamp(jittered, 0.0, max_delay);
  return static_cast<int>(std::lround(jittered));
}

bool isRetryableHttpStatus(int status) {
  return status == 429 || status == 502 || status == 503 || status == 504;
}

bool isRetryableGrpcStatus(int grpc_status_code) {
  // grpc::StatusCode values, kept as literals to avoid a gRPC header
  // dependency.
  constexpr int kDeadlineExceeded = 4;
  constexpr int kResourceExhausted = 8;
  constexpr int kUnavailable = 14;
  return grpc_status_code == kDeadlineExceeded ||
         grpc_status_code == kResourceExhausted ||
         grpc_status_code == kUnavailable;
}

void defaultSleep(int ms) {
  if (ms > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }
}

double defaultJitter() {
  static thread_local std::mt19937 engine{std::random_device{}()};
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(engine);
}

} // namespace kserve
