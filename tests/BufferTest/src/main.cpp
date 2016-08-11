#include "utils/Buffer.hpp"
#include "utils/PVLog.hpp"

#include <vector>

using PV::Buffer;

// Buffer::at(int row, int column, int feature)
// Buffer::set(int row, int column, int feature, float value)
void testAtSet() {
   Buffer testBuffer(2, 3, 4);
   pvErrorIf(testBuffer.at(1, 1, 1) != 0.0f, "Failed.\n");

   float v = 0;
   for(int r = 0; r < 2; ++r) {
      for(int c = 0; c < 3; ++c) {
         for(int f = 0; f < 4; ++f) {
            testBuffer.set(r, c, f, v++);
         }
      }
   }

   pvErrorIf(testBuffer.at(0, 0, 0) != 0.0f, "Failed.\n");
   pvErrorIf(testBuffer.at(0, 2, 0) != 8.0f, "Failed.\n");
   pvErrorIf(testBuffer.at(1, 2, 3) != 23.0f,"Failed.\n");
}

// Buffer::set(const std::vector<float> &vector)
void testSetVector() {
   Buffer testBuffer(2, 2, 2);
   std::vector<float> testVector = { 
         0.0f, 1.0f,
         2.0f, 3.0f,
         4.0f, 5.0f,
         6.0f, 7.0f
      };
   testBuffer.set(testVector);

   float v = 0;
   for(int r = 0; r < 2; ++r) {
      for(int c = 0; c < 2; ++c) {
         for(int f = 0; f < 2; ++f) {
            pvErrorIf(testBuffer.at(r, c, f) != v++,
                  "Failed. Expected %d, found %d instead.\n", v-1, testBuffer.at(r, c, f));
         }
      }
   }
}

// Buffer::asVector()
void testAsVector() {
   Buffer testBuffer(4, 3, 2);
   std::vector<float> testVector;
   
   float v = 1.0f;
   for(int r = 0; r < 4; ++r) {
      for(int c = 0; c < 3; ++c) {
         for(int f = 0; f < 2; ++f) {
            testVector.push_back(v);
            testBuffer.set(r, c, f, v++);
         }
      }
   }

   std::vector<float> comparison = testBuffer.asVector();
   pvErrorIf(comparison.size() != testVector.size(),
         "Failed: Expected a vector of size %d, found %d instead.\n",
         testVector.size(), comparison.size());

   for(int i = 0; i < testVector.size(); ++i) {
      pvErrorIf(testVector.at(i) != comparison.at(i),
            "Failed: Expected %d, found %d at index %d.\n",
            testVector.at(i), comparison.at(i), i);
   }
}

// Buffer::resize(int rows, int columns, int features)
void testResize() {
   Buffer testBuffer(4, 3, 2);
   pvErrorIf(testBuffer.getRows() != 4,
         "Failed: expected 4, found %d instead.\n", testBuffer.getRows());
   pvErrorIf(testBuffer.getColumns() != 3,
         "Failed: expected 3, found %d instead.\n", testBuffer.getColumns());
   pvErrorIf(testBuffer.getFeatures() != 2,
         "Failed: expected 2, found %d instead.\n", testBuffer.getFeatures());

   // Fill the Buffer with values, then resize.
   // Test the dimensions and make sure the resize
   // cleared all values.
   float v = 0.0f;
   for(int r = 0; r < 4; ++r) {
      for(int c = 0; c < 3; ++c) {
         for(int f = 0; f < 2; ++f) {
            testBuffer.set(r, c, f, v++);
         }
      }
   }
   testBuffer.resize(5, 4, 3);

   pvErrorIf(testBuffer.getRows() != 5,
         "Failed (rows): expected 5, found %d instead.\n", testBuffer.getRows());
   pvErrorIf(testBuffer.getColumns() != 4,
         "Failed (columns): expected 4, found %d instead.\n", testBuffer.getColumns());
   pvErrorIf(testBuffer.getFeatures() != 3,
         "Failed (features): expected 3, found %d instead.\n", testBuffer.getFeatures());

   v = 0.0f;
   for(int r = 0; r < 5; ++r) {
      for(int c = 0; c < 4; ++c) {
         for(int f = 0; f < 3; ++f) {
            v += testBuffer.at(r, c, f);
         }
      }
   }

   pvErrorIf(v != 0.0f, "Failed: Found non-zero value after resizing.\n");
}

// Buffer::crop(int targetRows, int targetColumns, enum OffsetAnchor offsetAnchor, int offsetX, int offsetY)
void testCrop() {
   std::vector<float> bufferContents = {
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f,  10.0f, 11.0f, 12.0f,
         13.0f, 14.0f, 15.0f, 16.0f
      };

   // Test each offset anchor

   Buffer testBuffer(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::NORTH, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 2.0f || testBuffer.at(0, 1, 0) != 3.0f,
         "Failed (north).\n");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::NORTHEAST, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 3.0f || testBuffer.at(1, 1, 0) != 8.0f,
         "Failed (northeast).\n");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::EAST, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 1, 0) != 8.0f || testBuffer.at(1, 1, 0) != 12.0f,
         "Failed (east).\n");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::SOUTHEAST, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 1, 0) != 12.0f || testBuffer.at(1, 0, 0) != 15.0f,
         "Failed (southeast).\n");
   
   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::SOUTH, 0, 0);
   pvErrorIf(
         testBuffer.at(1, 0, 0) != 14.0f || testBuffer.at(1, 1, 0) != 15.0f,
         "Failed (south).\n");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::SOUTHWEST, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 9.0f || testBuffer.at(1, 1, 0) != 14.0f,
         "Failed (southwest).\n");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::WEST, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 5.0f || testBuffer.at(1, 0, 0) != 9.0f,
         "Failed (west).\n");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::NORTHWEST, 0, 0);
   pvErrorIf(
         testBuffer.at(1, 0, 0) != 5.0f || testBuffer.at(0, 1, 0) != 2.0f,
         "Failed (northwest).\n");
   
   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::CENTER, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 6.0f || testBuffer.at(1, 1, 0) != 11.0f,
         "Failed (center).\n");

   // Test offsetX and offsetY
 
   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::CENTER, 1, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 7.0f || testBuffer.at(1, 1, 0) != 12.0f,
         "Failed (offsetX = 1).\n");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::EAST, -1, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 6.0f || testBuffer.at(1, 1, 0) != 11.0f,
         "Failed (offsetX = -1).\n");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::NORTHWEST, 0, 1);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 5.0f || testBuffer.at(1, 1, 0) != 10.0f,
         "Failed (offsetY = 1).\n");
 
   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::SOUTH, 0, -1);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 6.0f || testBuffer.at(1, 1, 0) != 11.0f,
         "Failed (offsetY = -1).\n");
}

// Buffer::rescale(int targetRows, int targetColumns, enum RescaleMethod rescaleMethod, enum InterpolationMethod interpMethod)
void testRescale() {
   //TODO: Figure out how to test rescaling
   pvErrorIf(true, "Not implemented.\n");
}


int main(int argc, char** argv) {
   pvInfo() << "Testing Buffer::at(): ";
   testAtSet();
   pvInfo() << "Completed.\n";
  
   pvInfo() << "Testing Buffer::set(): ";
   testSetVector();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing Buffer::asVector(): ";
   testAsVector();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing Buffer::resize(): ";
   testResize();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing Buffer::crop(): ";
   testCrop();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing Buffer::rescale(): ";
   testRescale();
   pvInfo() << "Completed.\n";
  
   pvInfo() << "Buffer tests completed successfully!\n";
   return EXIT_SUCCESS;
}
