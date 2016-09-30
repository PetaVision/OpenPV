#include "structures/Buffer.hpp"
#include "structures/SparseList.hpp"
#include "utils/BufferUtilsPvp.hpp"
#include "utils/PVLog.hpp"

#include <vector>

using PV::Buffer;
using PV::SparseList;
using std::vector;
namespace BufferUtils = PV::BufferUtils;

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
      Buffer<float> testBuffer;
      double timeVal =
            BufferUtils::readFromPvp<float>("input/input_8x4x2_x3.pvp", &testBuffer, frame);

      pvErrorIf(
            timeVal != (double)frame + 1,
            "Failed on frame %d. Expected time %d, found %d.\n",
            frame,
            frame + 1,
            (int)timeVal);
      pvErrorIf(
            testBuffer.getWidth() != 8,
            "Failed on frame %d. Expected width to be 8, found %d.\n",
            frame,
            testBuffer.getWidth());
      pvErrorIf(
            testBuffer.getHeight() != 4,
            "Failed on frame %d. Expected height to be 4, found %d.\n",
            frame,
            testBuffer.getHeight());
      pvErrorIf(
            testBuffer.getFeatures() != 2,
            "Failed on frame %d. Expected features to be 2, found %d.\n",
            frame,
            testBuffer.getFeatures());

      vector<float> readData = testBuffer.asVector();
      pvErrorIf(
            readData.size() != testData.size(),
            "Failed on frame %d. Expected %d elements, found %d.\n",
            frame,
            testData.size(),
            readData.size());

      for (int i = 0; i < 8 * 4 * 2; ++i) {
         pvErrorIf(
               readData.at(i) != testData.at(i),
               "Failed on frame %d. Expected value %d, found %d.\n",
               frame,
               (int)testData.at(i),
               (int)readData.at(i));
      }
   }
}

void testWriteToPvp() {

   // This test builds a buffer, writes it to
   // disk, and then reads it back in to verify
   // its contents. The result of this test can
   // only be trusted if the read test passed.
   vector<vector<float>> allFrames(3);
   float val = 0.0f;
   for (int frame = 0; frame < 3; ++frame) {
      vector<float> testData(8 * 4 * 2);
      for (int i = 0; i < 8 * 4 * 2; ++i) {
         testData.at(i) = val++;
      }
      allFrames.at(frame) = testData;
      Buffer<float> outBuffer(testData, 8, 4, 2);

      if (frame == 0) {
         BufferUtils::writeToPvp<float>("test.pvp", &outBuffer, (double)(frame + 1), true);
      } else {
         BufferUtils::appendToPvp<float>("test.pvp", &outBuffer, frame, (double)(frame + 1), true);
      }
   }

   // Now that we've written our pvp file, read it in
   // and check that it's correct
   for (int frame = 0; frame < 3; ++frame) {
      Buffer<float> testBuffer;
      double timeVal             = BufferUtils::readFromPvp<float>("test.pvp", &testBuffer, frame);
      vector<float> expectedData = allFrames.at(frame);

      pvErrorIf(
            timeVal != (double)frame + 1,
            "Failed on frame %d. Expected time %d, found %d.\n",
            frame,
            frame + 1,
            (int)timeVal);
      pvErrorIf(
            testBuffer.getWidth() != 8,
            "Failed on frame %d. Expected width to be 8, found %d.\n",
            frame,
            testBuffer.getWidth());
      pvErrorIf(
            testBuffer.getHeight() != 4,
            "Failed on frame %d. Expected height to be 4, found %d.\n",
            frame,
            testBuffer.getHeight());
      pvErrorIf(
            testBuffer.getFeatures() != 2,
            "Failed on frame %d. Expected features to be 2, found %d.\n",
            frame,
            testBuffer.getFeatures());

      vector<float> readData = testBuffer.asVector();
      pvErrorIf(
            readData.size() != expectedData.size(),
            "Failed on frame %d. Expected %d elements, found %d.\n",
            frame,
            expectedData.size(),
            readData.size());

      for (int i = 0; i < 8 * 4 * 2; ++i) {
         pvErrorIf(
               readData.at(i) != expectedData.at(i),
               "Failed on frame %d. Expected value %d, found %d.\n",
               frame,
               (int)expectedData.at(i),
               (int)readData.at(i));
      }
   }
}

void testSparseFile(const char *fName) {
   // This loop recreates the values in the input pvp file
   vector<float> expected(5 * 5 * 5);
   float val = 1.0f;
   for (int f = 0; f < 5; ++f) {
      for (int i = 0; i < 5 * 5; ++i) {
         if (i % 2 == 0) {
            expected.at(i + f * 5 * 5) = val++;
         }
      }
   }

   for (int f = 0; f < 5; ++f) {
      SparseList<float> list;
      double timeStamp = BufferUtils::readSparseFromPvp(fName, &list, f);
      pvErrorIf(
            (int)timeStamp != f + 1,
            "Failed on frame %d timeStamp. Expected time %d, found %d.\n",
            (int)f,
            f + 1,
            (int)timeStamp);

      pvErrorIf(
            list.getContents().size() != 13,
            "Expected 13 values, found %d.\n",
            list.getContents().size());

      Buffer<float> buffer(5, 5, 1);
      list.toBuffer(buffer, 0.0f);

      vector<float> values = buffer.asVector();

      int frameOffset = f * 5 * 5;

      for (int i = 0; i < values.size(); ++i) {
         pvErrorIf(
               values.at(i) != expected.at(i + frameOffset),
               "Failed on frame %d. Expected value %d at index %d, found %d.\n",
               f,
               (int)expected.at(i + frameOffset),
               i,
               (int)values.at(i));
      }
   }
}

void testReadSparseFromPvp() { testSparseFile("input/sparse_5x5x1_x5.pvp"); }

void testWriteSparseToPvp() {
   vector<float> output(5 * 5, 0.0f);
   float val = 1.0f;
   for (int f = 0; f < 5; ++f) {
      for (int i = 0; i < 5 * 5; ++i) {
         if (i % 2 == 0) {
            output.at(i) = val++;
         }
      }
      Buffer<float> dense(output, 5, 5, 1);
      SparseList<float> list;
      list.fromBuffer(dense, 0.0f);
      if (f == 0) {
         BufferUtils::writeSparseToPvp<float>("sparse.pvp", &list, 1.0, 5, 5, 1, true);
      } else {
         BufferUtils::appendSparseToPvp<float>("sparse.pvp", &list, 1.0 + (double)f, f, true);
      }
   }

   testSparseFile("sparse.pvp");
}

void testReadFromSparseBinaryPvp() {
   for (int frame = 0; frame < 3; ++frame) {
      vector<float> testData(3 * 2 * 1);
      for (int i = 0; i < 3 * 2 * 1; ++i) {
         testData.at(i) = (1 + i + frame) % 2;
      }

      SparseList<float> list;
      double timeVal = BufferUtils::readSparseBinaryFromPvp<float>(
            "input/binary_3x2x1_x3.pvp", &list, frame, 1.0f, nullptr);
      Buffer<float> testBuffer(3, 2, 1);
      list.toBuffer(testBuffer, 0.0f);

      pvErrorIf(
            timeVal != (double)frame + 1,
            "Failed on frame %d. Expected time %d, found %d.\n",
            frame,
            frame + 1,
            (int)timeVal);
      pvErrorIf(
            testBuffer.getWidth() != 3,
            "Failed on frame %d. Expected width to be 3, found %d.\n",
            frame,
            testBuffer.getWidth());
      pvErrorIf(
            testBuffer.getHeight() != 2,
            "Failed on frame %d. Expected height to be 2, found %d.\n",
            frame,
            testBuffer.getHeight());
      pvErrorIf(
            testBuffer.getFeatures() != 1,
            "Failed on frame %d. Expected features to be 1, found %d.\n",
            frame,
            testBuffer.getFeatures());

      vector<float> readData = testBuffer.asVector();
      pvErrorIf(
            readData.size() != testData.size(),
            "Failed on frame %d. Expected %d elements, found %d.\n",
            frame,
            testData.size(),
            readData.size());

      for (int i = 0; i < 3 * 2 * 1; ++i) {
         pvErrorIf(
               readData.at(i) != testData.at(i),
               "Failed on frame %d. Expected value %d, found %d.\n",
               frame,
               (int)testData.at(i),
               (int)readData.at(i));
      }
   }
}
int main(int argc, char **argv) {

   pvInfo() << "Testing BufferUtils:readFromPvp(): ";
   testReadFromPvp();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing BufferUtils:writeToPvp(): ";
   testWriteToPvp();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing BufferUtils:readSparseFromPvp(): ";
   testReadSparseFromPvp();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing BufferUtils:writeSparseToPvp(): ";
   testWriteSparseToPvp();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing BufferUtils:readSparseBinaryFromPvp(): ";
   testReadFromSparseBinaryPvp();
   pvInfo() << "Completed.\n";

   pvInfo() << "BufferUtils tests completed successfully!\n";
   return EXIT_SUCCESS;
}
