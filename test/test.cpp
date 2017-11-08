//
// Created by m_haid02 on 28.06.17.
//

#include <gtest/gtest.h>

#define RUN_PACXX_TEST(TestCat, TestName)        \
    TEST(TestCat, TestName){                     \
      EXPECT_EQ(0, test_##TestName(0, nullptr)); \
    }

// low level tests
#include "../vadd_low_level/vadd_low_level.h"
#include "../barrier/barrier.h"

// msp tests
#include "../msp_test/vadd_msp.h"

// high level tests
#include "../vadd/vadd.h"
#include "../saxpy/saxpy.h"
#include "../dot/dot.h"
#include "../sum/sum.h"

RUN_PACXX_TEST(BasicTests, vadd_low_level)
RUN_PACXX_TEST(BarrierTest, barrier);
RUN_PACXX_TEST(MSPTests, vadd_msp)

RUN_PACXX_TEST(RangesTests, vadd)
RUN_PACXX_TEST(RangesTests, saxpy)
RUN_PACXX_TEST(RangesTests, sum)
RUN_PACXX_TEST(RangesTests, dot)

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}