#include "utils/Buffer.hpp"
#include "utils/PVLog.hpp"

#include <vector>

using PV::Buffer;

// Buffer::at(int x, int y, int feature)
// Buffer::set(int x, int y, int feature, float value)
void testAtSet() {
   Buffer testBuffer(3, 2, 4);
   pvErrorIf(testBuffer.at(1, 1, 1) != 0.0f, "Failed.\n");

   float v = 0;
   for(int y = 0; y < 2; ++y) {
      for(int x = 0; x < 3; ++x) {
         for(int f = 0; f < 4; ++f) {
            testBuffer.set(x, y, f, v++);
         }
      }
   }

   pvErrorIf(testBuffer.at(0, 0, 0) != 0.0f, "Failed.\n");
   pvErrorIf(testBuffer.at(2, 0, 0) != 8.0f, "Failed.\n");
   pvErrorIf(testBuffer.at(2, 1, 3) != 23.0f,"Failed.\n");
}

// Buffer::set(const std::vector<float> &vector)
void testSetVector() {
   std::vector<float> testVector = { 
         0.0f, 1.0f,
         2.0f, 3.0f,
         4.0f, 5.0f,
         6.0f, 7.0f
      };
   Buffer testBuffer(testVector, 2, 2, 2);
   float v = 0;
   for(int y = 0; y < 2; ++y) {
      for(int x = 0; x < 2; ++x) {
         for(int f = 0; f < 2; ++f) {
            pvErrorIf(testBuffer.at(x, y, f) != v++,
                  "Failed. Expected %d, found %d instead.\n", v-1, testBuffer.at(x, y, f));
         }
      }
   }
}

// Buffer::asVector()
void testAsVector() {
   Buffer testBuffer(3, 4, 2);
   std::vector<float> testVector;
   
   float v = 1.0f;
   for(int y = 0; y < 4; ++y) {
      for(int x = 0; x < 3; ++x) {
         for(int f = 0; f < 2; ++f) {
            testVector.push_back(v);
            testBuffer.set(x, y, f, v++);
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

// Buffer::resize(int width, int height, int features)
void testResize() {
   Buffer testBuffer(3, 4, 2);
   pvErrorIf(testBuffer.getHeight() != 4,
         "Failed (height): expected 4, found %d instead.\n", testBuffer.getHeight());
   pvErrorIf(testBuffer.getWidth() != 3,
         "Failed (width): expected 3, found %d instead.\n", testBuffer.getWidth());
   pvErrorIf(testBuffer.getFeatures() != 2,
         "Failed (features): expected 2, found %d instead.\n", testBuffer.getFeatures());

   // Fill the Buffer with values, then resize.
   // Test the dimensions and make sure the resize
   // cleared all values.
   float v = 0.0f;
   for(int y = 0; y < 4; ++y) {
      for(int x = 0; x < 3; ++x) {
         for(int f = 0; f < 2; ++f) {
            testBuffer.set(x, y, f, v++);
         }
      }
   }
   testBuffer.resize(4, 5, 3);

   pvErrorIf(testBuffer.getHeight() != 5,
         "Failed (height): expected 5, found %d instead.\n", testBuffer.getHeight());
   pvErrorIf(testBuffer.getWidth() != 4,
         "Failed (width): expected 4, found %d instead.\n", testBuffer.getWidth());
   pvErrorIf(testBuffer.getFeatures() != 3,
         "Failed (features): expected 3, found %d instead.\n", testBuffer.getFeatures());

   v = 0.0f;
   for(int y = 0; y < 5; ++y) {
      for(int x = 0; x < 4; ++x) {
         for(int f = 0; f < 3; ++f) {
            v += testBuffer.at(x, y, f);
         }
      }
   }

   pvErrorIf(v != 0.0f, "Failed: Found non-zero value after resizing.\n");
}

// Buffer::crop(int targetWidth, int targetHeight, enum OffsetAnchor offsetAnchor, int offsetX, int offsetY)
void testCrop() {
   std::vector<float> bufferContents = {
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f,  10.0f, 11.0f, 12.0f,
         13.0f, 14.0f, 15.0f, 16.0f
      };

   std::vector<Buffer::Anchor> anchors = {
         Buffer::NORTH, Buffer::SOUTH,
         Buffer::EAST, Buffer::WEST,
         Buffer::NORTHEAST, Buffer::NORTHWEST,
         Buffer::SOUTHEAST, Buffer::SOUTHWEST,
         Buffer::CENTER 
      };

   // Test cropping to the same size
   Buffer testBuffer(bufferContents, 4, 4, 1);
   
   for(auto anchor : anchors) {
      testBuffer.set(bufferContents, 4, 4, 1);
      testBuffer.crop(4, 4, anchor);
      std::vector<float> contents = testBuffer.asVector();
      for(int i = 0; i < contents.size(); ++i) {
         pvErrorIf(contents.at(i) != bufferContents.at(i),
               "Failed (same size crop).");
      }
   }


   // Test each offset anchor
   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer::NORTH);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 2.0f || testBuffer.at(1, 0, 0) != 3.0f,
         "Failed (north).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer::NORTHEAST);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 3.0f || testBuffer.at(1, 1, 0) != 8.0f,
         "Failed (northeast).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer::EAST);
   pvErrorIf(
         testBuffer.at(1, 0, 0) != 8.0f || testBuffer.at(1, 1, 0) != 12.0f,
         "Failed (east).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer::SOUTHEAST);
   pvErrorIf(
         testBuffer.at(1, 0, 0) != 12.0f || testBuffer.at(0, 1, 0) != 15.0f,
         "Failed (southeast).\n");
   
   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer::SOUTH);
   pvErrorIf(
         testBuffer.at(0, 1, 0) != 14.0f || testBuffer.at(1, 1, 0) != 15.0f,
         "Failed (south).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer::SOUTHWEST);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 9.0f || testBuffer.at(1, 1, 0) != 14.0f,
         "Failed (southwest).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer::WEST);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 5.0f || testBuffer.at(0, 1, 0) != 9.0f,
         "Failed (west).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer::NORTHWEST);
   pvErrorIf(
         testBuffer.at(0, 1, 0) != 5.0f || testBuffer.at(1, 0, 0) != 2.0f,
         "Failed (northwest).\n");
   
   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer::CENTER);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 6.0f || testBuffer.at(1, 1, 0) != 11.0f,
         "Failed (center).\n");

   // Test offsetX and offsetY
 
   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.translate(-1, 0);
   testBuffer.crop(2, 2, Buffer::CENTER);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 7.0f || testBuffer.at(1, 1, 0) != 12.0f,
         "Failed (offsetX = 1).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.translate(1, 0);
   testBuffer.crop(2, 2, Buffer::EAST);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 6.0f || testBuffer.at(1, 1, 0) != 11.0f,
         "Failed (offsetX = -1).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.translate(0, -1);
   testBuffer.crop(2, 2, Buffer::NORTHWEST);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 5.0f || testBuffer.at(1, 1, 0) != 10.0f,
         "Failed (offsetY = 1).\n");
 
   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.translate(0, 1);
   testBuffer.crop(2, 2, Buffer::SOUTH);
   pvErrorIf(
         testBuffer.at(0, 0, 0) != 6.0f || testBuffer.at(1, 1, 0) != 11.0f,
         "Failed (offsetY = -1).\n");
}

// Buffer::rescale(int targetRows, int targetColumns, enum RescaleMethod rescaleMethod, enum InterpolationMethod interpMethod, enum OffsetAnchor offsetAnchor)
void testRescale() {

   std::vector<float> testData = {
      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
   };

   std::vector<float> answerNearest = {
      1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 5.0f, 5.0f, 1.0f,
      1.0f, 5.0f, 5.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f
   };

   std::vector<float> answerCrop = {
      1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f,
   };

   std::vector<float> answerPad = {
      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f,
      1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 5.0f, 5.0f, 1.0f,
      1.0f, 5.0f, 5.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f,
      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f
   };


   Buffer testBuffer(testData, 8, 8, 1);

   // Test nearest neighbor scaling. Rescale method will not be
   // used here because the aspect ratio is the same.
   testBuffer.rescale(4, 4, Buffer::PAD, Buffer::NEAREST, Buffer::CENTER);
   std::vector<float> nearest = testBuffer.asVector();

   pvErrorIf(nearest.size() != answerNearest.size(),
         "Failed (Size). Expected %d elements, found %d.\n",
         answerNearest.size(), nearest.size());
   for(int i = 0; i < nearest.size(); ++i) {
      pvErrorIf(nearest.at(i) != answerNearest.at(i),
         "Failed (Nearest). Expected %f at index %d, found %f.\n",
         answerNearest.at(i), i, nearest.at(i));
   }
 
   // Test Buffer::CROP resizeMethod
   testBuffer.set(testData, 8, 8, 1);
   testBuffer.rescale(8, 4, Buffer::CROP, Buffer::NEAREST, Buffer::CENTER);
   std::vector<float> cropped = testBuffer.asVector();
   pvErrorIf(cropped.size() != answerCrop.size(),
         "Failed (Size). Expected %d elements, found %d.\n",
         answerCrop.size(), cropped.size());
   for(int i = 0; i < cropped.size(); ++i) {
      pvErrorIf(cropped.at(i) != answerCrop.at(i),
         "Failed (Crop). Expected %f at index %d, found %f.\n",
         answerCrop.at(i), i, cropped.at(i));
   }
   
   // Test Buffer::PAD resizeMethod
   testBuffer.set(testData, 8, 8, 1);
   testBuffer.rescale(4, 8, Buffer::PAD, Buffer::NEAREST, Buffer::CENTER);
   std::vector<float> padded = testBuffer.asVector();
   pvErrorIf(padded.size() != answerPad.size(),
         "Failed (Size). Expected %d elements, found %d.\n",
         answerPad.size(), padded.size());
  for(int i = 0; i < padded.size(); ++i) {
      pvErrorIf(padded.at(i) != answerPad.at(i),
         "Failed (Pad). Expected %f at index %d, found %f.\n",
         answerPad.at(i), i, padded.at(i));
   }
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
