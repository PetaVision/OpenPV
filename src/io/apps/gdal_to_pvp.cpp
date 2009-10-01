/*
 * bin_to_tiff.c
 *
 *  Created on: Jan 5, 2009
 *      Author: rasmussn
 */

#include "src/io/imageio.hpp"
#include "src/columns/HyPerCol.hpp"
#include "src/layers/Image.hpp"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

using namespace PV;

/**
 * @argc
 * @argv[]
 */
int main(int argc, char* argv[])
{
   int n, nx, ny;
   unsigned char * buf;

   LayerLoc loc;

   int status = 0;

   if (argc < 5) {
     printf("usage: gdal_to_pvpar infile outfile nx ny\n");
//     exit(1);
   }

   HyPerCol * hc = new HyPerCol("column", argc, argv);

   const char * infile = hc->inputFile();
   const char * outfile = "image.pvp";

   // read an image
   Image * image = new Image("Image", hc, infile);
   
   loc = image->getImageLoc();

   return 0;

   int numItems = loc.nx * loc.ny * loc.nBands;

   buf = new unsigned char[numItems];
   assert(buf != NULL);

   float * data = image->getImageBuffer();
   for (int k = 0; k < numItems; k++) {
      buf[k] = (unsigned char) data[k];
   }

   status = gatherParByteFile(outfile, hc->icCommunicator(), &loc, buf);

   return 0;

   status = scatterParByteFile(outfile, hc->icCommunicator(), &loc, buf);
   for (int k = 0; k < numItems; k++) {
      data[k] = (float) buf[k];
   }

   image->write("image.jpg");

   delete buf;
   delete hc;

   return 0;
}
