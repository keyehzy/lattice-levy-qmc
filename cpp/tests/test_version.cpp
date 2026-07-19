#include "qmc/version.hpp"

#include <gtest/gtest.h>
#include <string_view>

namespace {

static_assert(qmc::kVersionMajor == 0);
static_assert(qmc::kVersionMinor == 5);
static_assert(qmc::kVersionPatch == 0);
static_assert(qmc::kVersion == std::string_view{"0.5.0"});

TEST(Version, ComponentsMatchVersionString) {
  EXPECT_EQ(qmc::kVersion, "0.5.0");
  EXPECT_EQ(QMC_VERSION_STRING, qmc::kVersion);
}

} // namespace
