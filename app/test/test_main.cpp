#include <gtest/gtest.h>

// Main entry point for the tests
int main(int argc, char **argv) {
  // Death tests default to fork ("fast"), which hangs where the process is
  // already threaded (this binary links backend runtimes that spawn worker
  // threads at startup). Re-exec ("threadsafe") is robust everywhere,
  // including CI, and costs one exec per death test.
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
