#include "utils/RealBuffer.hpp"
#include "utils/PVLog.hpp"
#include "io/BufferIO.hpp"

#include <vector>


void testReadFromPvp() {

   // The input file is 8 x 4 x 2, with 3 frames.
   // The stored value in each is the index, with
   // each frame starting where the last finished
   float val = 0.0f;
   for (int frame = 0; frame < 3; ++frame) {
      vector<float> testData(8 * 4 * 2);
      for (int i = 0; i < 8 * 4 * 2; ++i) {
         testData.at(i) = val++;
      }
      RealBuffer testBuffer();
      double timeVal = BufferIO::readFromPvp("input/input_8x4x2_x3.pvp",
                                             &testBuffer,
                                             frame);
      pvErrorIf(timeVal != (double)frame + 1,
            "Failed on frame %d. Expected time %d, found %d.\n",
            frame + 1, (int)timeVal);
      
      pvErrorIf(testBuffer.getWidth() != 8,
            "Failed on frame %d. Expected width to be 8, found %d.\n",
            frame, testBuffer.getWidth());
      pvErrorIf(testBuffer.getHeight() != 4,
            "Failed on frame %d. Expected height to be 4, found %d.\n",
            frame, testBuffer.getHeight());
      pvErrorIf(testBuffer.getFeatures() != 2,
            "Failed on frame %d. Expected features to be 2, found %d.\n",
            frame, testBuffer.getFeatures());

      vector<float> readData = testBuffer.asVector();
      pvErrorIf(readData.size() != testData.size(),
            "Failed on frame %d. Expected %d elements, found %d.\n",
            frame, testData.size(), readData.size());

      for (int i = 0; i < 8 * 4 * 2; ++i) {
         pvErrorIf(readData.at(i) != testData.at(i),
               "Failed on frame %d. Expected value %d, found %d.\n",
               frame, (int)testData.at(i), (int)readData.at(i));
      }
   }
}

void testWriteToPvp() {
   pvErrorIf(true, "Failed. Not implemented.\n",);
}


void testAppendToPvp() {
}

int main(int argc, char **argv) {

   // TODO: Test reading header first

   pvInfo() << "Testing BufferIO:readFromPvp(): ";
   testReadFromPvp();
   pvInfo() << "Completed.\n";
   
   pvInfo() << "Testing BufferIO:writeToPvp(): ";
   testWriteToPvp();
   pvInfo() << "Completed.\n";
  



   pvInfo() << "BufferIO tests completed successfully!\n";
   return EXIT_SUCCESS;
}
