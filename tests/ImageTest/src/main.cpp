#include "utils/Image.hpp"
#include "utils/PVLog.hpp"

using PV::Image;

// Image::Image(filename.png24)
void testPng24Load() {
   pvErrorIf(true, "Not implemented.\n");
}

// Image;:Image(filename.png32)
void testPng32Load() {
   pvErrorIf(true, "Not implemented.\n");
}

// Image::Image(filename.jpg)
void testJpgLoad() {
   pvErrorIf(true, "Not implemented.\n");
}

// Image::Image(filename.bmp)
void testBmpLoad() {
   pvErrorIf(true, "Not implemented.\n");
}

// Image::convertToColor(bool alphaChannel)
void testConvertToColor() {
     pvErrorIf(true, "Not implemented.\n");
}

// Image::convertToColor(bool alphaChannel)
void testConvertToGray() {
   pvErrorIf(true, "Not implemented.\n");
}

int main(int argc, char** argv) {
   pvInfo() << "Testing Image::Image(png24): ";
   testPng24Load();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing Image::Image(png32): ";
   testPng32Load();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing Image::Image(jpg): ";
   testJpgLoad();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing Image::Image(bmp): ";
   testBmpLoad();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing Image::convertToColor(): ";
   testConvertToColor();
   pvInfo() << "Completed.\n";

   pvInfo() << "Testing Image::convertToGray(): ";
   testConvertToGray();
   pvInfo() << "Completed.\n";

   pvInfo() << "Image tests completed successfully!\n";
   return EXIT_SUCCESS;
}
