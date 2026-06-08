#ifdef NEURIPLO_INFER_WITH_KSERVE

#include "KserveRetry.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {

kserve::RetryPolicy makePolicy() {
  kserve::RetryPolicy policy;
  policy.max_attempts = 5;
  policy.base_delay_ms = 100;
  policy.max_delay_ms = 2000;
  policy.jitter = 0.2;
  return policy;
}

} // namespace

TEST(KserveRetry, BackoffMonotonicAtMidJitter) {
  const auto policy = makePolicy();
  // rand_unit == 0.5 cancels jitter, exposing the pure exponential schedule.
  EXPECT_EQ(kserve::backoffDelayMs(policy, 1, 0.5), 100);
  EXPECT_EQ(kserve::backoffDelayMs(policy, 2, 0.5), 200);
  EXPECT_EQ(kserve::backoffDelayMs(policy, 3, 0.5), 400);
  EXPECT_EQ(kserve::backoffDelayMs(policy, 4, 0.5), 800);

  int prev = 0;
  for (int attempt = 1; attempt <= 10; ++attempt) {
    const int delay = kserve::backoffDelayMs(policy, attempt, 0.5);
    EXPECT_GE(delay, prev);
    prev = delay;
  }
}

TEST(KserveRetry, BackoffCappedAtMax) {
  const auto policy = makePolicy();
  // Large attempts saturate the exponential term at max_delay_ms.
  EXPECT_EQ(kserve::backoffDelayMs(policy, 20, 0.5), policy.max_delay_ms);
  EXPECT_LE(kserve::backoffDelayMs(policy, 20, 1.0), policy.max_delay_ms);
}

TEST(KserveRetry, BackoffJitterBounds) {
  const auto policy = makePolicy();
  // Attempt 3 -> base term 400ms, jitter +/-20% -> [320, 480].
  EXPECT_EQ(kserve::backoffDelayMs(policy, 3, 0.0), 320);
  EXPECT_EQ(kserve::backoffDelayMs(policy, 3, 1.0), 480);
  for (double u = 0.0; u <= 1.0; u += 0.1) {
    const int delay = kserve::backoffDelayMs(policy, 3, u);
    EXPECT_GE(delay, 320);
    EXPECT_LE(delay, 480);
  }
}

TEST(KserveRetry, BackoffHandlesNonPositiveAttempt) {
  const auto policy = makePolicy();
  EXPECT_EQ(kserve::backoffDelayMs(policy, 0, 0.5), 0);
  EXPECT_EQ(kserve::backoffDelayMs(policy, -3, 0.5), 0);
}

TEST(KserveRetry, RetryableHttpStatuses) {
  EXPECT_TRUE(kserve::isRetryableHttpStatus(429));
  EXPECT_TRUE(kserve::isRetryableHttpStatus(502));
  EXPECT_TRUE(kserve::isRetryableHttpStatus(503));
  EXPECT_TRUE(kserve::isRetryableHttpStatus(504));

  EXPECT_FALSE(kserve::isRetryableHttpStatus(200));
  EXPECT_FALSE(kserve::isRetryableHttpStatus(400));
  EXPECT_FALSE(kserve::isRetryableHttpStatus(401));
  EXPECT_FALSE(kserve::isRetryableHttpStatus(404));
  EXPECT_FALSE(kserve::isRetryableHttpStatus(500));
}

TEST(KserveRetry, RetryableGrpcStatuses) {
  EXPECT_TRUE(kserve::isRetryableGrpcStatus(4));  // DEADLINE_EXCEEDED
  EXPECT_TRUE(kserve::isRetryableGrpcStatus(8));  // RESOURCE_EXHAUSTED
  EXPECT_TRUE(kserve::isRetryableGrpcStatus(14)); // UNAVAILABLE

  EXPECT_FALSE(kserve::isRetryableGrpcStatus(0));  // OK
  EXPECT_FALSE(kserve::isRetryableGrpcStatus(3));  // INVALID_ARGUMENT
  EXPECT_FALSE(kserve::isRetryableGrpcStatus(7));  // PERMISSION_DENIED
  EXPECT_FALSE(kserve::isRetryableGrpcStatus(16)); // UNAUTHENTICATED
}

TEST(KserveRetry, RunWithRetryStopsOnSuccess) {
  const auto policy = makePolicy();
  int calls = 0;
  std::vector<int> sleeps;
  const int result = kserve::runWithRetry(
      policy,
      [&] {
        ++calls;
        return calls; // first non-retryable result
      },
      [](int) { return false; }, [&](int ms) { sleeps.push_back(ms); },
      [] { return 0.5; });
  EXPECT_EQ(result, 1);
  EXPECT_EQ(calls, 1);
  EXPECT_TRUE(sleeps.empty());
}

TEST(KserveRetry, RunWithRetryRetriesUntilNonRetryable) {
  const auto policy = makePolicy(); // 5 attempts
  int calls = 0;
  std::vector<int> sleeps;
  const int result = kserve::runWithRetry(
      policy, [&] { return ++calls; },
      [](int value) { return value < 3; }, // retry while < 3
      [&](int ms) { sleeps.push_back(ms); }, [] { return 0.5; });
  EXPECT_EQ(result, 3);
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(sleeps.size(), 2u); // slept before attempts 2 and 3
}

TEST(KserveRetry, RunWithRetryReturnsLastResultOnExhaustion) {
  kserve::RetryPolicy policy;
  policy.max_attempts = 3;
  int calls = 0;
  const int result = kserve::runWithRetry(
      policy, [&] { return ++calls; }, [](int) { return true; }, [](int) {},
      [] { return 0.0; });
  EXPECT_EQ(result, 3); // returns the final (still-retryable) result
  EXPECT_EQ(calls, 3);
}

TEST(KserveRetry, RunWithRetryRethrowsAfterExhaustion) {
  kserve::RetryPolicy policy;
  policy.max_attempts = 3;
  int calls = 0;
  EXPECT_THROW(
      {
        kserve::runWithRetry(
            policy,
            [&]() -> int {
              ++calls;
              throw std::runtime_error("transport down");
            },
            [](int) { return false; }, [](int) {}, [] { return 0.0; });
      },
      std::runtime_error);
  EXPECT_EQ(calls, 3); // all attempts used before the exception propagates
}

#endif // NEURIPLO_INFER_WITH_KSERVE
