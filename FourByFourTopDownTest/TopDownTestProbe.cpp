/*
 * TopDownTestProbe.cpp
 *
 *  Created on:
 *      Author: pschultz
 */

#include "TopDownTestProbe.hpp"

namespace PV {

/**
 * @filename
 * @type
 * @msg
 */
TopDownTestProbe::TopDownTestProbe(const char * filename, HyPerCol * hc, const char * msg, float checkperiod)
   : StatsProbe(filename, hc, msg)
{
   initialize(checkperiod);
}

TopDownTestProbe::~TopDownTestProbe() {
   free(prev_l);
   free(imageLibrary);
   free(scores);
}

int TopDownTestProbe::initialize(float checkperiod) {
   prev_l = NULL;
   numXPixels = 0;
   numXGlobal = 0;
   numYPixels = 0;
   numYGlobal = 0;
   numAllPixels = 0;
   numAllGlobal = 0;
   numImages = 0;
   localXOrigin = 0;
   localYOrigin = 0;
   imageLibrary = NULL;
   scores = NULL;
   this->checkperiod = checkperiod;
   nextupdate = checkperiod;
   return PV_SUCCESS;
}

/**
 * @time
 * @l
 */
int TopDownTestProbe::outputState(float time, HyPerLayer * l) {
   if( checkperiod > 0 && time >= nextupdate ) {
      if( l != prev_l ) {
         resetImageLibrary(l);
      }
#ifdef PV_USE_MPI
      InterColComm * icComm = l->getParent()->icCommunicator();
      MPI_Comm mpi_comm = icComm->communicator();
      int rank = icComm->commRank();
#endif // PV_USE_MPI
      // Compare l's data to each element in the image library in turn.
      for( int n=0; n<numImages; n++ ) {
         scores[n] = l2distsq(l->getV(), imageLibrary+n*numAllPixels);
      }
#ifdef PV_USE_MPI
      assert( (float) 3.456789012 == (pvdata_t) 3.456789012 && sizeof(pvdata_t) == sizeof(float) ); // catch if pvdata_t stops being float
      MPI_Allreduce(MPI_IN_PLACE, scores, numImages, MPI_FLOAT, MPI_SUM, mpi_comm);
      const int rcvProc = 0;
      if( rank != rcvProc ) {
         return PV_SUCCESS;
      }
#endif // PV_USE_MPI
      float minscore = FLT_MAX;
      int minidx = -1;
      for( int n=numImages-1; n>=0; n-- ) {
         if( scores[n] < minscore) {
            minscore = scores[n];
            minidx = n;
         }
      }
      int numatmin = 0;
      for( int n=0; n<numImages; n++ ) {
         scores[n] == minscore && numatmin++;
      }
      assert(numatmin > 0);
      minscore = sqrtf(minscore);
      fprintf(fp,"%stime=%f, reconstruction within %f of image %d in L2",msg, time, minscore, minidx);
      if( numatmin > 1) {
         fprintf(fp, " (as well as %d others)", numatmin-1);
      }
      fprintf(fp,"\n");
      if( minscore > 0.1 ) {
         fprintf(stderr, "%sLayer %s failed to converge to one of the target images.  Exiting.\n", msg, l->getName());
         exit(EXIT_FAILURE);
      }
      nextupdate += checkperiod;
   }

   return PV_SUCCESS;
}

int TopDownTestProbe::resetImageLibrary(HyPerLayer * l) {
   prev_l = l;
   const PVLayerLoc * newLoc = l->getLayerLoc();
   assert(newLoc->nf == 1);
   if( numXPixels != newLoc->nx || numYPixels != newLoc->ny ) {
      numXPixels = newLoc->nx;
      numYPixels = newLoc->ny;
      numAllPixels = numXPixels*numYPixels;
      numXGlobal = newLoc->nxGlobal;
      numYGlobal = newLoc->nyGlobal;
      numAllGlobal = numXGlobal*numYGlobal;
      localXOrigin = newLoc->kx0;
      localYOrigin = newLoc->ky0;
      numImages = numXGlobal + numYGlobal;
      free(scores);
      scores = (pvdata_t *) malloc( numImages*sizeof(pvdata_t) );
      setImageLibrary();
   }
   return PV_SUCCESS;
}

int TopDownTestProbe::setImageLibrary() {
   free(imageLibrary);
   imageLibrary = (pvdata_t *) malloc( numImages*numAllPixels*sizeof(pvdata_t) );
   if( imageLibrary == NULL ) {
      fprintf(stderr, "TopDownTestProbe: Unable to allocate memory for a library of %d images of %d-by-%d pixels.  Exiting\n",numImages,numXPixels,numYPixels);
      exit(EXIT_FAILURE);
   }
   pvdata_t * libptr = imageLibrary;
   // Needs to be MPI aware
   for( int idx=0; idx < numXGlobal; idx++ ) {
      for( int x=0; x<numXPixels; x++ ) {
         for( int y=0; y<numYPixels; y++ ) {
            *libptr = (pvdata_t) (x + localXOrigin == idx);
            libptr++;
         }
      }
   }
   for( int idy=0; idy < numYGlobal; idy++) {
        for( int x=0; x<numXPixels; x++ ) {
            for( int y=0; y<numYPixels; y++ ) {
               *libptr = (pvdata_t) (y + localYOrigin == idy);
               libptr++;
            }
         }

   }
   return PV_SUCCESS;
}

pvdata_t TopDownTestProbe::l2distsq(pvdata_t * x, pvdata_t * y) {
   double val = 0;
   for( int k=0; k<numAllPixels; k++ ) {
      double dif = ((double) x[k])-((double) y[k]);
      val += dif*dif;
   }
   return (pvdata_t) val;
}

} // end namespace PV
