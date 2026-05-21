//
// Created by kazem on 8/28/25.
//

#include "gtest/gtest.h"

#include "vec_op.h"

#include <vector>

TEST(VecOpTest, BasicAssertions) {
    std::vector<float> a = {1.0, 2.0, 3.0};
    std::vector<float> b = {4.0, 5.0, 6.0};
    std::vector<float> c;

    swiftware::hpp::vec_op(a, b, c);

    ASSERT_EQ(c.size(), a.size());
    EXPECT_FLOAT_EQ(c[0], 1.0 * 4.0 * 0.5);
    EXPECT_FLOAT_EQ(c[1], 2.0 * 5.0 * 0.5);
    EXPECT_FLOAT_EQ(c[2], 3.0 * 6.0 * 0.5);
}

TEST(VecOpZeroVecATest, BasicAssertions) {
    std::vector<float> a = {0.0, 0.0, 0.0};
    std::vector<float> b = {4.0, 5.0, 6.0};
    std::vector<float> c;

    swiftware::hpp::vec_op(a, b, c);

    ASSERT_EQ(c.size(), a.size());
    EXPECT_FLOAT_EQ(c[0], 0.0 * 4.0 * 0.5);
    EXPECT_FLOAT_EQ(c[1], 0.0 * 5.0 * 0.5);
    EXPECT_FLOAT_EQ(c[2], 0.0 * 6.0 * 0.5);
}

TEST(VecOpEmptyVecATest, BasicAssertions) {
    std::vector<float> a = {};
    std::vector<float> b = {4.0, 5.0, 6.0};
    std::vector<float> c;

    swiftware::hpp::vec_op(a, b, c);

    EXPECT_TRUE(c.empty());
}

TEST(VecOpSizeMismatchTest, BasicAssertions) {
    std::vector<float> a = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> b = {4.0, 5.0, 6.0};
    std::vector<float> c;

    swiftware::hpp::vec_op(a, b, c);

    ASSERT_EQ(c.size(), a.size());
    ASSERT_NE(a.size(), b.size());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}