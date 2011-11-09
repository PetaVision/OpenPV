/*
 * ActivityProbe.cpp
 *
 *  Created on: Oct 20, 2009
 *      Author: travel
 */

#include "ActivityProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include "io.h"
#include "tiff.h"

static FILE *
pv_tiff_open_frame(const char * filename,
                   const PVLayerLoc * loc, pvdata_t ** imageBuf, long * nextFrame);

static int
pv_tiff_close_frame(FILE * fp, pvdata_t * imageBuf, long nextFrame);

static int
pv_tiff_write_frame(FILE * fp, const pvdata_t * data,
                    const PVLayerLoc * loc, pvdata_t * buf, long * nextFrame);



namespace PV {

ActivityProbe::ActivityProbe(const char * filename, HyPerCol * hc, const PVLayerLoc * loc, int f)
{
   outfp = pv_tiff_open_frame(filename, loc, &outBuf, &outFrame);
}

ActivityProbe::~ActivityProbe()
{
   if (outfp != NULL) {
      pv_tiff_close_frame(outfp, outBuf, outFrame);
   }
}

int ActivityProbe::outputState(float time, HyPerLayer * l)
{
   int status = 0;

   if (outfp != NULL) {
      const pvdata_t * data = l->getLayerData();
      status = pv_tiff_write_frame(outfp, data, &l->clayer->loc, outBuf, &outFrame);
   }

   return status;
}

} // namespace PV


static int
pv_tiff_close_frame(FILE * fp, pvdata_t * imageBuf, long nextLoc)
{
        tiff_write_finish(fp, nextLoc);
        fclose(fp);
        free(imageBuf);

        return 0;
}

static FILE *
pv_tiff_open_frame(const char * filename, const PVLayerLoc * loc, pvdata_t ** imageBuf, long * nextLoc)
{
   const int nx = loc->nx;
   const int ny = loc->ny;

   pvdata_t * buf = (pvdata_t *) malloc(nx * ny * sizeof(pvdata_t));
   *imageBuf = buf;

   FILE * fp = fopen(filename, "wb");
   if (fp == NULL) {
      fprintf(stderr, "pv_tiff_open_frame_cube: ERROR opening file %s\n", filename);
      return fp;
   }

   tiff_write_header(fp, nextLoc);

   return fp;
}

static int
pv_tiff_write_frame(FILE * fp, const pvdata_t * data,
                    const PVLayerLoc * loc, pvdata_t * buf, long * nextLoc)
{
   int k;

   float scale = 1.0;
   float max = -1.0e99;
   float min =  1.0e99;

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const int numItems = nx*ny*nf;

   for (k = 0; k < numItems; k++) {
      float val = data[k];
      if (val < min) min = val;
      if (val > max) max = val;
   }
   //   scale = 1.0 / (max - min);

   if (min < 0.0 || min > 1.0) {
      fprintf(stderr, "[ ]: pv_tiff_write_frame: mininum value out of bounds=%f\n", min);
   }
   if (max < 0.0 || max > 1.0) {
      fprintf(stderr, "[ ]: pv_tiff_write_frame: maximum value out of bounds=%f\n", max);
   }

   min = 0.0;
   max = 1.0;
   scale = 1.0;

   PV::HyPerLayer::copyToBuffer(buf, data, loc, true, scale);

   tiff_write_ifd(fp, nextLoc, nx, ny);
   tiff_write_image(fp, buf, nx, ny);

   return 0;
}
