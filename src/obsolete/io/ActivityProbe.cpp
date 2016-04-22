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

static PV_Stream *
pv_tiff_open_frame(const char * filename,
                   const PVLayerLoc * loc, pvdata_t ** imageBuf, long * nextFrame, bool verifyWrites);

static int
pv_tiff_close_frame(PV_Stream * pvstream, pvdata_t * imageBuf, long nextFrame);

static int
pv_tiff_write_frame(PV_Stream * pvstream, const pvdata_t * data,
                    const PVLayerLoc * loc, pvdata_t * buf, long * nextFrame);



namespace PV {

ActivityProbe::ActivityProbe() {
   initialize_base();
}

ActivityProbe::ActivityProbe(const char * probeName, HyPerCol * hc)
{
   initialize_base();
   initialize(probeName, hc);
}

ActivityProbe::~ActivityProbe()
{
   if (outputstream != NULL) {
      pv_tiff_close_frame(outputstream, outBuf, outFrame);
      outputstream = NULL;
   }
}

int ActivityProbe::initialize_base() {
   outFrame = 0L;
   outBuf = NULL;
   return PV_SUCCESS;
}

int ActivityProbe::initialize(const char * probeName, HyPerCol * hc) {
   if (hc->icCommunicator()->commSize()>1) {
      if (hc->columnId()==0) {
         fprintf(stderr, "ActivityProbe \"%s\" error: ActivityProbe is not compatible with MPI.\n", probeName);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(hc->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   return LayerProbe::initialize(probeName, hc);
}

int ActivityProbe::initOutputStream(const char * filename, bool verifyWrites) {
   outputstream = pv_tiff_open_frame(filename, getTargetLayer()->getLayerLoc(), &outBuf, &outFrame, verifyWrites);
   return PV_SUCCESS;
}


int ActivityProbe::outputState(double time)
{
   int status = 0;

   if (outputstream != NULL) {
      const pvdata_t * data = getTargetLayer()->getLayerData();
      status = pv_tiff_write_frame(outputstream, data, getTargetLayer()->getLayerLoc(), outBuf, &outFrame);
   }

   return status;
}

} // namespace PV


static int
pv_tiff_close_frame(PV_Stream * pvstream, pvdata_t * imageBuf, long nextLoc)
{
        tiff_write_finish(pvstream->fp, nextLoc);
        PV::PV_fclose(pvstream);
        free(imageBuf);

        return 0;
}

static PV_Stream *
pv_tiff_open_frame(const char * filename, const PVLayerLoc * loc, pvdata_t ** imageBuf, long * nextLoc, bool verifyWrites)
{
   pvdata_t * buf = (pvdata_t *) malloc(loc->nx * loc->ny * loc->nf * sizeof(pvdata_t));
   *imageBuf = buf;

   PV_Stream * pvstream = PV::PV_fopen(filename, "wb", verifyWrites);
   if (pvstream == NULL) {
      fprintf(stderr, "pv_tiff_open_frame_cube: ERROR opening file %s\n", filename);
      return pvstream;
   }

   tiff_write_header(pvstream->fp, nextLoc);

   return pvstream;
}

static int
pv_tiff_write_frame(PV_Stream * pvstream, const pvdata_t * data,
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

   tiff_write_ifd(pvstream->fp, nextLoc, nx, ny);
   tiff_write_image(pvstream->fp, buf, nx, ny);

   return 0;
}
