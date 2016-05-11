/*
 * ProbeActivityLinear.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: Craig Rasmussen
 */

#include "LinearActivityProbe.hpp"

namespace PV {

/**
 * @probeName
 * @hc
 */
LinearActivityProbe::LinearActivityProbe(const char * probeName, HyPerCol * hc) {
   initLinearActivityProbe_base();
   initLinearActivityProbe(probeName, hc);
}

LinearActivityProbe::LinearActivityProbe() {
   initLinearActivityProbe_base();
   // Derived classes should call initLinearActivityProbe
}

LinearActivityProbe::~LinearActivityProbe() {
   free(dimString); dimString = NULL;
}

int LinearActivityProbe::initLinearActivityProbe_base() {
   hc = NULL;
   dimString = NULL;
   dim = DimX;
   linePos = 0;
   f = 0;
   return PV_SUCCESS;
}

int LinearActivityProbe::initLinearActivityProbe(const char * probeName, HyPerCol * hc) {
   int status = LayerProbe::initialize(probeName, hc);
   return status;
}

int LinearActivityProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_dim(ioFlag);
   ioParam_linePos(ioFlag);
   ioParam_f(ioFlag);
   return status;
}

void LinearActivityProbe::ioParam_dim(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamString(ioFlag, getName(), "dim", &dimString, "DimX");
   if (ioFlag == PARAMS_IO_READ) {
      assert(dimString!=NULL);
      if (!strcmp(dimString, "DimX")) {
         dim = DimX;
      }
      else if (!strcmp(dimString, "DimY")) {
         dim = DimY;
      }
      else {
         if (getParent()->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: parameter \"dim\" must be either \"DimX\" or \"DimY\".\n",
                  getParent()->parameters()->groupKeywordFromName(getName()), getName());
         }
#ifdef PV_USE_MPI
         MPI_Barrier(getParent()->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }
   }
}

void LinearActivityProbe::ioParam_linePos(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "linePos", &linePos, linePos);
}

void LinearActivityProbe::ioParam_f(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "f", &f, f);
}

/**
 * @time
 * @l
 * NOTES:
 *    - layer activity lives in an extended space
 *    - by setting dim to PV::dimX or PV::dimY we can plot activity
 *    along the line or along the column.
 *    .
 */
int LinearActivityProbe::outputState(double timef)
{
   int width, sLine;
   const float * line;

   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();

   const pvdata_t * activity = getTargetLayer()->getLayerData();

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const PVHalo * halo = &loc->halo;

   float dt = hc->getDeltaTime();

   double sum = 0.0;
   float freq;

   if (dim == DimX) {
      width = nx + halo->lt + halo->rt;
      line = activity + (linePos + halo->up) * width * nf;
      sLine = nf;
   }
   else {
      width = ny + halo->dn + halo->up;
      line = activity + (linePos + halo->lt)*nf;
      sLine = nf * (nx + halo->dn + halo->up);

   }

   for (int k = 0; k < width; k++) {
     float a = line[f + k * sLine];
     sum += a;
   }

   freq = sum / (width * dt * 0.001);
   fprintf(outputstream->fp, "t=%6.1f sum=%3d f=%6.1f Hz :", timef, (int)sum, freq);

   for (int k = 0; k < width; k++) {
     float a = line[f + k * sLine];
     if (a > 0.0) { fprintf(outputstream->fp, "*"); }
     else         { fprintf(outputstream->fp, " "); }
   }
   fprintf(outputstream->fp, ":\n");
   fflush(outputstream->fp);

   return 0;
}

} // namespace PV
