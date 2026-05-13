/*
 * main.cpp
 *
 * A simple program to verify whether loading a TIFF works as expected.
 * PV_Init is used only to initialize the log file; other arguments
 * from the command line or config file are ignored.
 */

#include <cMakeHeader.h> // Loads the value of PV_USE_TIFF

#ifdef PV_USE_TIFF

#include <cstdlib>
#include <string>
#include <tiffio.h>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <utils/PVLog.hpp>

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;

   PV::PV_Init pv_initObj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   std::string tifile("input/testfile.tif");
   TIFF *tiff = TIFFOpen(tifile.c_str(), "r");

   std::uint32_t width, height, numPixels;
   if (TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) == 0) {
      ErrorLog().printf("Error reading image width\n");
      status = PV_FAILURE;
   };
   if (TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height) == 0) {
      ErrorLog().printf("Error reading image height\n");
      status = PV_FAILURE;
   };
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }

   numPixels = width * height;
   std::uint32_t *imageBuffer = (std::uint32_t*)_TIFFmalloc(numPixels * sizeof(std::uint32_t));
   FatalIf(imageBuffer == nullptr, "Error calling _TIFFmalloc().\n");

   int readResult = TIFFReadRGBAImage(tiff, width, height, imageBuffer, 0 /*stopOnError*/);
   FatalIf(readResult == 0, "Error calling TIFFReadRGBAImage.\n");

   std::vector<std::uint8_t> correctValues{
         0x7f, 0x00, 0x00, 0xff,
         0x00, 0xbf, 0xbf, 0x00,
         0x00, 0x3f, 0x3f, 0x00,
         0xff, 0x00, 0x00, 0x7f};

   for (std::uint32_t n = 0U; n < numPixels; ++n) {
      std::uint32_t &observed = imageBuffer[n];
      std::uint8_t r = TIFFGetR(observed);
      std::uint8_t g = TIFFGetR(observed);
      std::uint8_t b = TIFFGetR(observed);
      std::uint8_t &correct = correctValues[n];
      if (r != correct or g != correct or b != correct) {
         ErrorLog().printf(
               "Bad pixel at index %" PRIu32 ": expected value %" PRIu8 ", "
               "observed value (%" PRIu8 "," PRIu8 "," PRIu8 ")\n",
               n, correct, r, g, b);
         status = PV_FAILURE;
      }
   }
   _TIFFfree(imageBuffer);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else // PV_USE_TIFF

#include <cstdlib>
#include <columns/PV_Init.hpp>
#include <utils/PVLog.hpp>
int main(int argc, char *argv[]) {
   PV::PV_Init pv_initObj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   ErrorLog().printf(
         "%s requires the PV_USE_TIFF option to be on.\n",
         pv_initObj.returnProgramName());
   return EXIT_FAILURE;
}

#endif // PV_USE_TIFF
