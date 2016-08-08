#include "utils/Buffer.hpp"

#include <vector>

// Buffer::at(int row, int column, int feature)
// Buffer::set(int row, int column, int feature, float value)
bool testAtSet() {
   Buffer testBuffer(2, 3, 4);
   pvErrorIf(testBuffer.at(1, 1, 1) != 0.0f, "Buffer::at() test failed.");

   float v = 0;
   for(int r = 0; r < 2; ++r) {
      for(int c = 0; c < 3; ++c) {
         for(int f = 0; f < 4; ++f) {
            testBuffer.set(r, c, f, v++);
         }
      }
   }

   pvErrorIf(testBuffer.at(0, 0, 0) != 0.0f, "Buffer::at() or Buffer::set() test failed");
   pvErrorIf(testBuffer.at(0, 2, 0) != 6.0f, "Buffer::at() or Buffer::set() test failed");
   pvErrorIf(testBuffer.at(1, 2, 3) != 23.0f,"Buffer::at() or Buffer::set() test failed");

   return true;
}

// Buffer::set(const std::vector<float> &vector)
bool testSetVector() {
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
      for(int c = 0; c < 3; ++c) {
         for(int f = 0; f < 4; ++f) {
            pvErrorIf(testBuffer.at(r, c, f) != v++,
                  "Buffer::set() test failed: expected %d, found %d instead.", v-1, testBuffer.at(r, c, f));
         }
      }
   }

   return true;
}

// Buffer::resize(int rows, int columns, int features)
bool testResize() {
   Buffer testBuffer(4, 3, 2);

   float v = 0.0f;
   for(int r = 0; r < 4; ++r) {
      for(int c = 0; c < 3; ++c) {
         for(int f = 0; f < 2; ++f) {
            testBuffer.set(r, c, f, v++);
         }
      }
   }

   pvErrorIf(testBuffer.getRows() != 4,
         "Buffer::Buffer() or Buffer::getRows() test failed: expected 4, found %d instead.", testBuffer.getRows());
   pvErrorIf(testBuffer.getColumns() != 3,
         "Buffer::Buffer() or Buffer::getColumns() test failed: expected 3, found %d instead.", testBuffer.getColumns());
   pvErrorIf(testBuffer.getFeatures() != 2,
         "Buffer::Buffer() or Buffer::getFeatures() test failed: expected 2, found %d instead.", testBuffer.getFeatures());

   testBuffer.resize(5, 4, 3);

   pvErrorIf(testBuffer.getRows() != 5,
         "Buffer::resize() test (rows) failed: expected 5, found %d instead.", testBuffer.getRows());
   pvErrorIf(testBuffer.getColumns() != 4,
         "Buffer::resize() test (columns) failed: expected 4, found %d instead.", testBuffer.getColumns());
   pvErrorIf(testBuffer.getFeatures() != 3,
         "Buffer::resize() test (features) failed: expected 3, found %d instead.", testBuffer.getFeatures());

   v = 0.0f;
   for(int r = 0; r < 5; ++r) {
      for(int c = 0; c < 4; ++c) {
         for(int f = 0; f < 3; ++f) {
            v += testBuffer.at(r, c, f);
         }
      }
   }

   pvErrorIf(v != 0.0f, "Buffer::resize() test failed: Found non-zero value after resizing.");

   return true;
}

// Buffer::crop(int targetRows, int targetColumns, enum OffsetAnchor offsetAnchor, int offsetX, int offsetY)
bool testCrop() {
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
         "Buffer::crop() test (north) failed.");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::NORTHEAST, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 3.0f || testBuffer.at(1, 1, 0) != 8.0f,
         "Buffer::crop() test (northeast) failed.");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::EAST, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 1, 0) != 8.0f || testBuffer.at(1, 1, 0) != 12.0f,
         "Buffer::crop() test (east) failed.");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::SOUTHEAST, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 1, 0) != 12.0f || testBuffer.at(1, 0, 0) != 15.0f,
         "Buffer::crop() test (southeast) failed.");
   
   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::SOUTH, 0, 0);
   pvErrorIf(
         testBuffer.at(1, 0, 0) != 14.0f || testBuffer.at(1, 1, 0) != 15.0f,
         "Buffer::crop() test (south) failed.");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::SOUTHWEST, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 9.0f || testBuffer.at(1, 1, 0) != 14.0f,
         "Buffer::crop() test (southwest) failed.");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::WEST, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 5.0f || testBuffer.at(1, 0, 0) != 9.0f,
         "Buffer::crop() test (west) failed.");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::NORTHWEST, 0, 0);
   pvErrorIf(
         testBuffer.at(1, 0, 0) != 5.0f || testBuffer.at(0, 1, 0) != 2.0f,
         "Buffer::crop() test (northwest) failed.");
   
   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::CENTER, 0, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 6.0f || testBuffer.at(1, 1, 0) != 11.0f,
         "Buffer::crop() test (center) failed.");

   // Test offsetX and offsetY
 
   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::CENTER, 1, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 7.0f || testBuffer.at(1, 1, 0) != 12.0f,
         "Buffer::crop() test (offsetX = 1) failed.");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::EAST, -1, 0);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 6.0f || testBuffer.at(1, 1, 0) != 11.0f,
         "Buffer::crop() test (offsetX = -1) failed.");

   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::NORTHWEST, 0, 1);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 5.0f || testBuffer.at(1, 1, 0) != 10.0f,
         "Buffer::crop() test (offsetY = 1) failed.");
 
   testBuffer.resize(4, 4, 1);
   testBuffer.set(bufferContents);
   testBuffer.crop(2, 2, Buffer::SOUTHEAST, 0, -1);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 11.0f || testBuffer.at(1, 1, 0) != 16.0f,
         "Buffer::crop() test (offsetY = -1) failed.");

   return true;
}

// Buffer::rescale(int targetRows, int targetColumns, enum RescaleMethod rescaleMethod, enum InterpolationMethod interpMethod)
bool testRescale() {

}

// Buffer::asVector()
bool testAsVector() {

}

int main(int argc, char** argv) {
   assert(testAtSet());
   assert(testSetVector());
   assert(testResize());
   assert(testCrop());
   assert(testRescale());
   assert(testAsVector());
   return EXIT_SUCCESS;
}
