#pragma once

// Pure, dependency-light retry/backoff policy for the KServe V2 clients.
// Nothing here depends on neuriplo, nlohmann/json, gRPC, or sockets, so the
// backoff schedule and the transient-error classifiers are unit-testable in
// isolation (only the standard library). The transports wire `runWithRetry`
// around their round-trips; the policy itself stays decoupled and testable.

#include <cstdint>

namespace kserve {

// Knobs for exponential backoff with jitter. max_attempts counts the first try
// plus retries (so 1 == no retry). Delays grow as base * 2^(attempt-1), clamped
// to max_delay_ms, then spread by +/- jitter (a fraction in [0, 1]).
struct RetryPolicy {
  int max_attempts = 3;
  int base_delay_ms = 100;
  int max_delay_ms = 2000;
  double jitter = 0.2;
};

// Builds a RetryPolicy from the environment, falling back to the struct
// defaults: KSERVE_MAX_RETRIES (total attempts), KSERVE_RETRY_BASE_MS,
// KSERVE_RETRY_MAX_MS, KSERVE_RETRY_JITTER. Invalid / non-positive values are
// ignored in favour of the defaults.
RetryPolicy retryPolicyFromEnv();

// Deterministic backoff (in milliseconds) for a 1-based attempt index given a
// unit random fraction `rand_unit` in [0, 1]. The exponential term is capped at
// max_delay_ms before jitter; the jittered result is clamped to
// [0, max_delay_ms]. Supplying a fixed rand_unit makes the schedule fully
// reproducible for tests (rand_unit == 0.5 yields the un-jittered term).
int backoffDelayMs(const RetryPolicy &policy, int attempt, double rand_unit);

// HTTP status codes worth retrying: 429 (Too Many Requests) and the 5xx
// gateway/availability family 502/503/504. Everything else is treated as a
// permanent failure and surfaces unchanged.
bool isRetryableHttpStatus(int status);

// gRPC status codes worth retrying, classified by integer value to avoid a hard
// dependency on the gRPC headers in this pure module:
//   DEADLINE_EXCEEDED (4), RESOURCE_EXHAUSTED (8), UNAVAILABLE (14).
bool isRetryableGrpcStatus(int grpc_status_code);

// Default backoff sleep (std::this_thread::sleep_for); a no-op for ms <= 0.
void defaultSleep(int ms);

// Default jitter source: a uniform fraction in [0, 1) from a thread-local PRNG.
double defaultJitter();

// Runs `attempt` until it succeeds, returns a non-retryable result, or the
// policy's attempts are exhausted. `attempt()` performs one round-trip and
// returns a Result. `is_retryable(result)` decides whether a (non-throwing)
// result warrants another try. A thrown exception is treated as a transient
// transport failure and retried while attempts remain; on the final attempt the
// last exception is rethrown and the last result returned, so callers keep
// throwing the same exception types/messages as without retries.
//
// `sleeper(ms)` performs the backoff wait and `rng()` yields a unit fraction in
// [0, 1); both are injectable so the loop is deterministically testable.
template <typename Attempt, typename IsRetryable, typename Sleeper,
          typename Rng>
auto runWithRetry(const RetryPolicy &policy, Attempt attempt,
                  IsRetryable is_retryable, Sleeper sleeper,
                  Rng rng) -> decltype(attempt()) {
  const int attempts = policy.max_attempts > 0 ? policy.max_attempts : 1;
  for (int i = 1;; ++i) {
    const bool last = i >= attempts;
    try {
      auto result = attempt();
      if (last || !is_retryable(result)) {
        return result;
      }
    } catch (...) {
      if (last) {
        throw;
      }
    }
    sleeper(backoffDelayMs(policy, i, rng()));
  }
}

} // namespace kserve
