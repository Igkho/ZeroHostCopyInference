#include <gtest/gtest.h>
#include "HelpersTests.h"
#include "BlockTests.h"
#include "SafeQueueTests.h"
#include "FFMpegSourceTests.h"
#include "StubDetectorTests.h"
#include "NVJpegSinkTests.h"
#include "PerformanceTimerTests.h"
#include "InferencePipelineTests.h"
#include "DetectorKernelsTests.h"
#include "ObjectTrackerTests.h"
#include "OnnxDetectorTests.h"
#include "TrtDetectorTests.h"
#include "DataStructuresTests.h"

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
