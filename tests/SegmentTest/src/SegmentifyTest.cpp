#include "SegmentifyTest.hpp"

namespace PV {

SegmentifyTest::SegmentifyTest(const char *name, PVParams *params, Communicator const *comm) {
   Segmentify::initialize(name, params, comm);
}

void SegmentifyTest::createComponentTable(char const *description) {
   Segmentify::createComponentTable(description);
   FatalIf(
         mActivityComponent == nullptr,
         "%s failed to create an ActivityComponent.\n",
         getDescription_c());

   mSegmentifyBuffer = mActivityComponent->getComponentByType<SegmentifyBuffer>();
   FatalIf(
         mSegmentifyBuffer == nullptr,
         "%s failed to create a SegmentifyBuffer.\n",
         getDescription_c());
}

/*
 * Segment values are such:
 * 1 1 1 2 2 2 3 3
 * 1 1 1 2 2 2 3 3
 * 1 1 1 2 2 2 3 3
 * 4 4 4 5 5 5 6 6
 * 4 4 4 5 5 5 6 6
 * 4 4 4 5 5 5 6 6
 * 7 7 7 8 8 8 9 9
 * 7 7 7 8 8 8 9 9
 */
float SegmentifyTest::getTargetVal(int yi, int xi, int fi) {
   // We can convert yi and xi to an index between 0 and 2
   int newYi               = yi / 3;
   int newXi               = xi / 3;
   int segmentLabel        = newYi * 3 + newXi + 1;
   int returnLabel         = -1;
   char const *inputMethod = mSegmentifyBuffer->getInputMethod();
   if (strcmp(inputMethod, "sum") == 0) {
      // Account for edge cases
      if (segmentLabel == 3 || segmentLabel == 6 || segmentLabel == 7 || segmentLabel == 8) {
         returnLabel = segmentLabel * 6;
      }
      else if (segmentLabel == 9) {
         returnLabel = segmentLabel * 4;
      }
      else {
         returnLabel = segmentLabel * 9;
      }
   }
   else if (strcmp(inputMethod, "average") == 0 || strcmp(inputMethod, "max") == 0) {
      returnLabel = segmentLabel;
   }
   else {
      // Should never get here
      FatalIf(!(0), "Test failed.\n");
   }
   return returnLabel;
}

int SegmentifyTest::checkOutputVals(int yi, int xi, int fi, float targetVal, float actualVal) {
   // We can convert yi and xi to an index between 0 and 2
   int newYi                = yi / 3;
   int newXi                = xi / 3;
   char const *outputMethod = mSegmentifyBuffer->getOutputMethod();

   if (strcmp(outputMethod, "centroid") == 0) {
      int centX, centY;
      if (newXi == 0) {
         centX = 1;
      }
      else if (newXi == 1) {
         centX = 4;
      }
      else if (newXi == 2) {
         centX = 6;
      }
      if (newYi == 0) {
         centY = 1;
      }
      else if (newYi == 1) {
         centY = 4;
      }
      else if (newYi == 2) {
         centY = 6;
      }

      if (xi == centX && yi == centY) {
         FatalIf(!(actualVal == targetVal), "Test failed.\n");
      }
      else {
         FatalIf(!(actualVal == 0), "Test failed.\n");
      }
   }
   else if (strcmp(outputMethod, "fill") == 0) {
      FatalIf(!(actualVal == targetVal), "Test failed.\n");
   }
   return PV_SUCCESS;
}

Response::Status SegmentifyTest::checkUpdateState(double timef, double dt) {
   // Do update state first
   Segmentify::checkUpdateState(timef, dt);
   PVLayerLoc const *loc = getLayerLoc();

   ActivityComponent *activityComponent = getComponentByType<ActivityComponent>();
   FatalIf(
         activityComponent == nullptr,
         "%s does not contain an ActivityComponent.\n",
         getDescription_c());
   ActivityBuffer *activityBuffer = activityComponent->getComponentByType<ActivityBuffer>();
   FatalIf(
         activityBuffer == nullptr, "%s does not contain an ActivityBuffer.\n", getDescription_c());

   for (int bi = 0; bi < loc->nbatch; bi++) {
      float const *batchA = activityBuffer->getBufferData(bi);
      FatalIf(
            batchA == nullptr, "%s ActivityBuffer->getBufferData() failed.\n", getDescription_c());

      for (int yi = 0; yi < loc->ny; yi++) {
         for (int xi = 0; xi < loc->nx; xi++) {
            for (int fi = 0; fi < loc->nf; fi++) {
               int extIdx = (yi + loc->halo.up) * (loc->nx + loc->halo.lt + loc->halo.rt) * loc->nf
                            + (xi + loc->halo.lt) * loc->nf + fi;
               float actualVal = batchA[extIdx];
               float targetVal = getTargetVal(yi + loc->ky0, xi + loc->kx0, fi);
               checkOutputVals(yi + loc->ky0, xi + loc->kx0, fi, targetVal, actualVal);

               // InfoLog() << "Idx: (" << bi << "," << yi << "," << xi << "," << fi << ") Val: " <<
               // actualVal << " Target: " << targetVal << "\n";
            }
         }
      }
   }

   return Response::SUCCESS;
}

} /* namespace PV */
