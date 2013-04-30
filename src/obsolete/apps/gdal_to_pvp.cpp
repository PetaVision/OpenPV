/*
 * bin_to_tiff.c
 *
 *  Created on: Jan 5, 2009
 *      Author: rasmussn
 */

#include "src/layers/Image.hpp"
#include "src/io/imageio.hpp"
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
   unsigned char * buf;
   LayerLoc loc;
   int status = 0;

   if (argc < 3) {
     printf("usage: gdal_to_pvp infile outfile.pvp [outfile.jpg]\n");
     exit(1);
   }

   const char * infile = argv[1];
   const char * outfile_pvp = argv[2];
   const char * outfile_jpg = NULL;

   if (argc > 3) {
      outfile_jpg = argv[3];
   }

   Communicator * comm = new Communicator(&argc, &argv);

   // allocate memory for image buffer
   
   loc.nPad = 0;
   status = getImageInfo(infile, comm, &loc);

   const int numItems = loc.nx * loc.ny * loc.nBands;
   buf = new unsigned char[numItems];
   assert(buf != NULL);

   // read the image and scatter the local portions
   status = scatterImageFile(infile, comm, &loc, buf);

   if (loc.nBands > 1) {
      Image::convertToGrayScale(&loc, buf);
   }

   // write the image back out to files

   status = gatherImageFile(outfile_pvp, comm, &loc, buf);
   if (outfile_jpg != NULL) {
      status = gatherImageFile(outfile_jpg, comm, &loc, buf);
   }

   delete buf;

   return status;
}
