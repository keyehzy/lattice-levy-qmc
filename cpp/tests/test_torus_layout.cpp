#include "qmc/torus_layout.hpp"

#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace {

TEST(TorusLayoutTest, EncodesDecodesAndReducesWithAxisZeroLeastSignificant) {
  const qmc::TorusLayout layout(4, 2);
  const std::array<qmc::Coord, 2> reduced{3, 2};
  const std::array<qmc::Coord, 2> covering{-1, 6};
  const std::array<std::size_t, 2> components{3, 2};

  EXPECT_EQ(layout.linear_size(), 4);
  EXPECT_EQ(layout.dimension(), 2U);
  EXPECT_EQ(layout.volume(), 16U);
  ASSERT_EQ(layout.strides().size(), 2U);
  EXPECT_EQ(layout.strides()[0], 1U);
  EXPECT_EQ(layout.strides()[1], 4U);
  EXPECT_EQ(layout.encode(reduced), qmc::SiteId(11));
  EXPECT_EQ(layout.encode(components), qmc::SiteId(11));
  EXPECT_EQ(layout.encode_covering(covering), qmc::SiteId(11));
  EXPECT_EQ(layout.reduce(-1), 3);
  EXPECT_EQ(layout.reduce(covering), (std::vector<qmc::Coord>{3, 2}));
  EXPECT_EQ(layout.decode(qmc::SiteId(11)), (std::vector<std::size_t>{3, 2}));

  std::array<qmc::Coord, 2> reduced_into{};
  layout.reduce_into(covering, reduced_into);
  EXPECT_EQ(reduced_into, reduced);

  std::array<std::size_t, 2> decoded{};
  layout.decode_into(qmc::SiteId(11), decoded);
  EXPECT_EQ(decoded, components);
}

TEST(TorusLayoutTest, ComputesPeriodicDisplacementsAndCoordinateShifts) {
  const qmc::TorusLayout layout(4, 2);
  const std::array<qmc::Coord, 2> origin{3, 1};
  const std::array<qmc::Coord, 2> target{1, 0};
  const qmc::SiteId origin_id = layout.encode(origin);
  const qmc::SiteId target_id = layout.encode(target);

  EXPECT_EQ(layout.flat_displacement(origin_id, target_id), qmc::SiteId(14));
  EXPECT_EQ(layout.shifted(origin_id, 0, 1), layout.encode(std::array<qmc::Coord, 2>{0, 1}));
  EXPECT_EQ(layout.shifted(origin_id, 1, -2), layout.encode(std::array<qmc::Coord, 2>{3, 3}));
  EXPECT_EQ(layout.shifted(origin_id, 0, std::numeric_limits<qmc::Coord>::min()), origin_id);
}

TEST(TorusLayoutTest, EnumeratesUniquePeriodicNeighborhoods) {
  const qmc::TorusLayout layout(5, 2);
  const qmc::SiteId center = layout.encode(std::array<qmc::Coord, 2>{0, 0});
  const auto local = layout.neighbors_within_radius(center, 1);
  const std::unordered_set<qmc::SiteId, qmc::SiteIdHash> local_set(local.begin(), local.end());

  EXPECT_EQ(local.size(), 9U);
  EXPECT_EQ(local_set.size(), local.size());
  for (const qmc::SiteId site : local) {
    EXPECT_TRUE(layout.within_radius(center, site, 1));
  }

  const auto whole_torus = layout.neighbors_within_radius(center, 2);
  const std::unordered_set<qmc::SiteId, qmc::SiteIdHash> whole_set(whole_torus.begin(),
                                                                   whole_torus.end());
  EXPECT_EQ(whole_torus.size(), layout.volume());
  EXPECT_EQ(whole_set.size(), layout.volume());
}

TEST(TorusLayoutTest, DistinguishesEqualVolumeLayouts) {
  const qmc::TorusLayout line(4, 1);
  const qmc::TorusLayout square(2, 2);

  ASSERT_EQ(line.volume(), square.volume());
  EXPECT_NE(line, square);
  EXPECT_EQ(line.decode(qmc::SiteId(3)), (std::vector<std::size_t>{3}));
  EXPECT_EQ(square.decode(qmc::SiteId(3)), (std::vector<std::size_t>{1, 1}));
}

TEST(TorusLayoutTest, RejectsInvalidShapesComponentsAndIdentities) {
  EXPECT_THROW(static_cast<void>(qmc::TorusLayout(0, 1)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(qmc::TorusLayout(1, 0)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(qmc::TorusLayout(std::numeric_limits<qmc::Coord>::max(), 3)),
               std::overflow_error);

  const qmc::TorusLayout layout(3, 2);
  EXPECT_THROW(static_cast<void>(layout.encode(std::array<qmc::Coord, 1>{0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(layout.encode(std::array<qmc::Coord, 2>{0, 3})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(layout.decode(qmc::SiteId(layout.volume()))), std::out_of_range);
  EXPECT_THROW(static_cast<void>(layout.shifted(qmc::SiteId(0), layout.dimension(), 1)),
               std::out_of_range);

  std::array<std::size_t, 1> wrong_size{};
  EXPECT_THROW(layout.decode_into(qmc::SiteId(0), wrong_size), std::invalid_argument);
  std::array<qmc::Coord, 1> wrong_site{};
  EXPECT_THROW(layout.reduce_into(wrong_site, wrong_site), std::invalid_argument);
}

} // namespace
