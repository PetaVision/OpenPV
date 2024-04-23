#include "include/pv_common.h"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsRescale.hpp"
#include "utils/PVLog.hpp"

#include <vector>

using PV::Buffer;
namespace BufferUtils = PV::BufferUtils;

// Buffer::at(int x, int y, int feature)
// Buffer::set(int x, int y, int feature, float value)
void testAtSet() {
   Buffer<float> testBuffer(3, 2, 4);
   FatalIf(testBuffer.at(1, 1, 1) != 0.0f, "Failed.\n");

   float v = 0;
   for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 3; ++x) {
         for (int f = 0; f < 4; ++f) {
            testBuffer.set(x, y, f, v++);
         }
      }
   }

   FatalIf(testBuffer.at(0, 0, 0) != 0.0f, "Failed.\n");
   FatalIf(testBuffer.at(2, 0, 0) != 8.0f, "Failed.\n");
   FatalIf(testBuffer.at(2, 1, 3) != 23.0f, "Failed.\n");
}

// Buffer::set(const std::vector<float> &vector)
void testSetVector() {
   std::vector<float> testVector = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
   Buffer<float> testBuffer(testVector, 2, 2, 2);
   float v = 0;
   for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
         for (int f = 0; f < 2; ++f) {
            FatalIf(
                  testBuffer.at(x, y, f) != v++,
                  "Failed. Expected %f, found %f instead.\n",
                  (double)v - 1,
                  (double)testBuffer.at(x, y, f));
         }
      }
   }
}

// Buffer::asVector()
void testAsVector() {
   Buffer<float> testBuffer(3, 4, 2);
   std::vector<float> testVector;

   float v = 1.0f;
   for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 3; ++x) {
         for (int f = 0; f < 2; ++f) {
            testVector.push_back(v);
            testBuffer.set(x, y, f, v++);
         }
      }
   }

   std::vector<float> comparison = testBuffer.asVector();
   FatalIf(
         comparison.size() != testVector.size(),
         "Failed: Expected a vector of size %d, found %d instead.\n",
         testVector.size(),
         comparison.size());

   for (std::size_t i = 0; i < testVector.size(); ++i) {
      FatalIf(
            testVector.at(i) != comparison.at(i),
            "Failed: Expected %f, found %f at index %d.\n",
            (double)testVector.at(i),
            (double)comparison.at(i),
            i);
   }
}

// Buffer::resize(int width, int height, int features)
void testResize() {
   Buffer<float> testBuffer(3, 4, 2);
   FatalIf(
         testBuffer.getHeight() != 4,
         "Failed (height): expected 4, found %d instead.\n",
         testBuffer.getHeight());
   FatalIf(
         testBuffer.getWidth() != 3,
         "Failed (width): expected 3, found %d instead.\n",
         testBuffer.getWidth());
   FatalIf(
         testBuffer.getFeatures() != 2,
         "Failed (features): expected 2, found %d instead.\n",
         testBuffer.getFeatures());

   // Fill the Buffer with values, then resize.
   // Test the dimensions and make sure the resize
   // cleared all values.
   float v = 0.0f;
   for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 3; ++x) {
         for (int f = 0; f < 2; ++f) {
            testBuffer.set(x, y, f, v++);
         }
      }
   }
   testBuffer.resize(4, 5, 3);

   FatalIf(
         testBuffer.getHeight() != 5,
         "Failed (height): expected 5, found %d instead.\n",
         testBuffer.getHeight());
   FatalIf(
         testBuffer.getWidth() != 4,
         "Failed (width): expected 4, found %d instead.\n",
         testBuffer.getWidth());
   FatalIf(
         testBuffer.getFeatures() != 3,
         "Failed (features): expected 3, found %d instead.\n",
         testBuffer.getFeatures());

   v = 0.0f;
   for (int y = 0; y < 5; ++y) {
      for (int x = 0; x < 4; ++x) {
         for (int f = 0; f < 3; ++f) {
            v += testBuffer.at(x, y, f);
         }
      }
   }

   FatalIf(v != 0.0f, "Failed: Found non-zero value after resizing.\n");
}

// Buffer::crop(int targetWidth, int targetHeight, enum OffsetAnchor offsetAnchor, int offsetX, int
// offsetY)
void testCrop() {
   std::vector<float> bufferContents = {1.0f,
                                        2.0f,
                                        3.0f,
                                        4.0f,
                                        5.0f,
                                        6.0f,
                                        7.0f,
                                        8.0f,
                                        9.0f,
                                        10.0f,
                                        11.0f,
                                        12.0f,
                                        13.0f,
                                        14.0f,
                                        15.0f,
                                        16.0f};

   std::vector<Buffer<float>::Anchor> anchors = {Buffer<float>::NORTH,
                                                 Buffer<float>::SOUTH,
                                                 Buffer<float>::EAST,
                                                 Buffer<float>::WEST,
                                                 Buffer<float>::NORTHEAST,
                                                 Buffer<float>::NORTHWEST,
                                                 Buffer<float>::SOUTHEAST,
                                                 Buffer<float>::SOUTHWEST,
                                                 Buffer<float>::CENTER};

   // Test cropping to the same size
   Buffer<float> testBuffer(bufferContents, 4, 4, 1);

   for (auto anchor : anchors) {
      testBuffer.set(bufferContents, 4, 4, 1);
      testBuffer.crop(4, 4, anchor);
      std::vector<float> contents = testBuffer.asVector();
      for (std::size_t i = 0; i < contents.size(); ++i) {
         FatalIf(contents.at(i) != bufferContents.at(i), "Failed (same size crop).");
      }
   }

   // Test each offset anchor
   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer<float>::NORTH);
   FatalIf(testBuffer.at(0, 0, 0) != 2.0f || testBuffer.at(1, 0, 0) != 3.0f, "Failed (north).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer<float>::NORTHEAST);
   FatalIf(
         testBuffer.at(0, 0, 0) != 3.0f || testBuffer.at(1, 1, 0) != 8.0f, "Failed (northeast).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer<float>::EAST);
   FatalIf(testBuffer.at(1, 0, 0) != 8.0f || testBuffer.at(1, 1, 0) != 12.0f, "Failed (east).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer<float>::SOUTHEAST);
   FatalIf(
         testBuffer.at(1, 0, 0) != 12.0f || testBuffer.at(0, 1, 0) != 15.0f,
         "Failed (southeast).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer<float>::SOUTH);
   FatalIf(testBuffer.at(0, 1, 0) != 14.0f || testBuffer.at(1, 1, 0) != 15.0f, "Failed (south).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer<float>::SOUTHWEST);
   FatalIf(
         testBuffer.at(0, 0, 0) != 9.0f || testBuffer.at(1, 1, 0) != 14.0f,
         "Failed (southwest).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer<float>::WEST);
   FatalIf(testBuffer.at(0, 0, 0) != 5.0f || testBuffer.at(0, 1, 0) != 9.0f, "Failed (west).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer<float>::NORTHWEST);
   FatalIf(
         testBuffer.at(0, 1, 0) != 5.0f || testBuffer.at(1, 0, 0) != 2.0f, "Failed (northwest).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.crop(2, 2, Buffer<float>::CENTER);
   FatalIf(testBuffer.at(0, 0, 0) != 6.0f || testBuffer.at(1, 1, 0) != 11.0f, "Failed (center).\n");

   // Test offsetX and offsetY

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.translate(-1, 0);
   testBuffer.crop(2, 2, Buffer<float>::CENTER);
   FatalIf(
         testBuffer.at(0, 0, 0) != 7.0f || testBuffer.at(1, 1, 0) != 12.0f,
         "Failed (offsetX = 1).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.translate(1, 0);
   testBuffer.crop(2, 2, Buffer<float>::EAST);
   FatalIf(
         testBuffer.at(0, 0, 0) != 6.0f || testBuffer.at(1, 1, 0) != 11.0f,
         "Failed (offsetX = -1).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.translate(0, -1);
   testBuffer.crop(2, 2, Buffer<float>::NORTHWEST);
   FatalIf(
         testBuffer.at(0, 0, 0) != 5.0f || testBuffer.at(1, 1, 0) != 10.0f,
         "Failed (offsetY = 1).\n");

   testBuffer.set(bufferContents, 4, 4, 1);
   testBuffer.translate(0, 1);
   testBuffer.crop(2, 2, Buffer<float>::SOUTH);
   FatalIf(
         testBuffer.at(0, 0, 0) != 6.0f || testBuffer.at(1, 1, 0) != 11.0f,
         "Failed (offsetY = -1).\n");
}

// BufferUtils::rescale(Buffer<float> &buffer, int targetRows, int targetColumns, enum RescaleMethod
// rescaleMethod, enum InterpolationMethod interpMethod, enum OffsetAnchor offsetAnchor)
void testRescale() {

   std::vector<float> testData = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f,
                                  1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f, 1.0f,
                                  1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 5.0f,
                                  5.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

   std::vector<float> answerNearest = {1.0f,
                                       1.0f,
                                       1.0f,
                                       1.0f,
                                       1.0f,
                                       5.0f,
                                       5.0f,
                                       1.0f,
                                       1.0f,
                                       5.0f,
                                       5.0f,
                                       1.0f,
                                       1.0f,
                                       1.0f,
                                       1.0f,
                                       1.0f};

   std::vector<float> answerCrop = {
         1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f,
         5.0f, 5.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f,
   };

   std::vector<float> answerPad = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                                   1.0f, 1.0f, 5.0f, 5.0f, 1.0f, 1.0f, 5.0f, 5.0f, 1.0f, 1.0f, 1.0f,
                                   1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

   Buffer<float> testBuffer(testData, 8, 8, 1);

   // Test nearest neighbor scaling. Rescale method will not be
   // used here because the aspect ratio is the same.
   BufferUtils::rescale(
         testBuffer, 4, 4, BufferUtils::PAD, BufferUtils::NEAREST, Buffer<float>::CENTER);
   std::vector<float> nearest = testBuffer.asVector();

   FatalIf(
         nearest.size() != answerNearest.size(),
         "Failed (Size). Expected %d elements, found %d.\n",
         answerNearest.size(),
         nearest.size());
   for (std::size_t i = 0; i < nearest.size(); ++i) {
      FatalIf(
            nearest.at(i) != answerNearest.at(i),
            "Failed (Nearest). Expected %f at index %d, found %f.\n",
            (double)answerNearest.at(i),
            i,
            (double)nearest.at(i));
   }

   // Test Buffer::CROP resizeMethod
   testBuffer.set(testData, 8, 8, 1);
   BufferUtils::rescale(
         testBuffer, 8, 4, BufferUtils::CROP, BufferUtils::NEAREST, Buffer<float>::CENTER);
   std::vector<float> cropped = testBuffer.asVector();
   FatalIf(
         cropped.size() != answerCrop.size(),
         "Failed (Size). Expected %d elements, found %d.\n",
         answerCrop.size(),
         cropped.size());
   for (std::size_t i = 0; i < cropped.size(); ++i) {
      FatalIf(
            cropped.at(i) != answerCrop.at(i),
            "Failed (Crop). Expected %f at index %d, found %f.\n",
            (double)answerCrop.at(i),
            i,
            (double)cropped.at(i));
   }

   // Test Buffer::PAD resizeMethod
   testBuffer.set(testData, 8, 8, 1);
   BufferUtils::rescale(
         testBuffer, 4, 8, BufferUtils::PAD, BufferUtils::NEAREST, Buffer<float>::CENTER);
   std::vector<float> padded = testBuffer.asVector();
   FatalIf(
         padded.size() != answerPad.size(),
         "Failed (Size). Expected %d elements, found %d.\n",
         answerPad.size(),
         padded.size());
   for (std::size_t i = 0; i < padded.size(); ++i) {
      FatalIf(
            padded.at(i) != answerPad.at(i),
            "Failed (Pad). Expected %f at index %d, found %f.\n",
            (double)answerPad.at(i),
            i,
            (double)padded.at(i));
   }
}

void testExtract() {
   // Create an 8x8x3 buffer and then extract a 2x2x3 buffer from it.
   int const nf = 3;

   int const nxMain = 8;
   int const nyMain = 8;

   int const nxExtracted = 3;
   int const nyExtracted = 3;

   int const numExtracted = nf * nxExtracted * nyExtracted;

   int const xStart = 1;
   int const yStart = 2;

   Buffer<float> mainBuffer(nxMain, nyMain, nf);
   for (int k = 0; k < mainBuffer.getTotalElements(); ++k) {
      mainBuffer.set(k, static_cast<float>(k));
   }

   Buffer<float> extractedBuffer = mainBuffer.extract(xStart, yStart, nxExtracted, nyExtracted);

   std::vector<float>correctValues(numExtracted);
   for (int k = 0; k < numExtracted; ++k) {
      int f = k % nf;
      int x = ((k / nf)) % nxExtracted;
      int y = (k / (nf * nxExtracted)) % nyExtracted;
      int value = f + nf * ( (x + xStart) + nxMain * (y + yStart));
      correctValues[k] = static_cast<float>(value);
   }

   int status = PV_SUCCESS;
   for (int k = 0; k < numExtracted; ++k) {
      if (extractedBuffer.at(k) != correctValues[k]) {
         ErrorLog().printf("Buffer::extract() failed: entry %d should be %f but is %f.\n",
               k,
               static_cast<double>(correctValues[k]),
               static_cast<double>(extractedBuffer.at(k)));
         if (status == PV_SUCCESS) { InfoLog() << "\n"; }
         status = PV_FAILURE;
      }
   }
   FatalIf(status != PV_SUCCESS, "Buffer::extract() failed.\n");
}

void testInsert() {
   // Create an 8x8x3 buffer and then insert a 2x2x3 buffer into it.
   int const nf = 3;

   int const nxMain = 8;
   int const nyMain = 8;

   int const nxInsert = 3;
   int const nyInsert = 3;

   int const numMain      = nf * nxMain * nyMain;

   int const xStart = 1;
   int const yStart = 2;

   Buffer<float> mainBuffer(nxMain, nyMain, nf);
   for (int k = 0; k < mainBuffer.getTotalElements(); ++k) {
      mainBuffer.set(k, static_cast<float>(k));
   }

   Buffer<float> bufferToInsert(nxInsert, nyInsert, nf);
   for (int k = 0; k < bufferToInsert.getTotalElements(); ++k) {
      bufferToInsert.set(k, static_cast<float>(k + 1000));
   }

   mainBuffer.insert(bufferToInsert, xStart, yStart);

   std::vector<float>correctValues(numMain);
   for (int k = 0; k < numMain; ++k) {
      int f = k % nf;
      int x = ((k / nf)) % nxMain;
      int y = (k / (nf * nxMain)) % nyMain;
      bool inserted =
            (x >= xStart and x < xStart + nxInsert and y >= yStart and y < yStart + nyInsert);
      int value = inserted ? f + nf * (x - xStart + nxInsert * (y - yStart)) + 1000 : k;
      correctValues[k] = static_cast<float>(value);
   }

   int status = PV_SUCCESS;
   for (int k = 0; k < numMain; ++k) {
      if (mainBuffer.at(k) != correctValues[k]) {
         ErrorLog().printf("Buffer::insert() failed: entry %d should be %f but is %f.\n",
               k, static_cast<double>(correctValues[k]), static_cast<double>(mainBuffer.at(k)));
         status = PV_FAILURE;
      }
   }
   FatalIf(status != PV_SUCCESS, "Buffer::insert() failed.\n");
}

int main(int argc, char **argv) {
   InfoLog() << "Testing Buffer::at(): ";
   testAtSet();
   InfoLog() << "Completed.\n";

   InfoLog() << "Testing Buffer::set(): ";
   testSetVector();
   InfoLog() << "Completed.\n";

   InfoLog() << "Testing Buffer::asVector(): ";
   testAsVector();
   InfoLog() << "Completed.\n";

   InfoLog() << "Testing Buffer::resize(): ";
   testResize();
   InfoLog() << "Completed.\n";

   InfoLog() << "Testing Buffer::crop(): ";
   testCrop();
   InfoLog() << "Completed.\n";

   InfoLog() << "Testing BufferUtils::rescale(): ";
   testRescale();
   InfoLog() << "Completed.\n";

   InfoLog() << "Testing Buffer::extract(): ";
   testExtract();
   InfoLog() << "Completed.\n";

   InfoLog() << "Testing Buffer::insert(): ";
   testInsert();
   InfoLog() << "Completed.\n";

   InfoLog() << "Buffer tests completed successfully!\n";
   return EXIT_SUCCESS;
}
