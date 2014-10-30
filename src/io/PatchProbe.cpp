/*
 * PatchProbe.cpp (formerly called ConnectionProbe.cpp)
 *
 *  Created on: Apr 25, 2009
 *      Author: Craig Rasmussen
 */

#include "PatchProbe.hpp"
#include <limits.h>
#include <assert.h>

namespace PV {

// Protected default constructor.  Derived classes should call this
// constructor, and call PatchProbe::initPatchProbe from within their own
// initialization.
PatchProbe::PatchProbe() {
   initialize_base();
}

/*
 * NOTES:
 *     - kxPre, kyPre, are indices in the restricted space.
 *     - kPre is the linear index which will be computed in the
 *     extended space, which includes margins.
 *
 *
 */

PatchProbe::PatchProbe(const char * probename, HyPerCol * hc)
{
   initialize_base();
   initialize(probename, hc);
}

PatchProbe::~PatchProbe()
{
   initialize_base();
}

int PatchProbe::initialize_base() {
   return PV_SUCCESS;
}

int PatchProbe::initialize(const char * probename, HyPerCol * hc) {
   BaseHyPerConnProbe::initialize(probename, hc);
   int status = getPatchID();
   assert(status == PV_SUCCESS);
   return PV_SUCCESS;
}

int PatchProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseHyPerConnProbe::ioParamsFillGroup(ioFlag);
   ioParam_arborID(ioFlag);
   ioParam_kPre(ioFlag);
   ioParam_kxPre(ioFlag);
   ioParam_kyPre(ioFlag);
   ioParam_kfPre(ioFlag);
   ioParam_arborID(ioFlag);
   ioParam_outputWeights(ioFlag);
   ioParam_outputPlasticIncr(ioFlag);
   ioParam_outputPostIndices(ioFlag);
   return status;
}

void PatchProbe::ioParam_kPre(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ || patchIDMethod == INDEX_METHOD) {
      parent->ioParamValue(ioFlag, name, "kPre", &kPre, INT_MIN, false/*warnIfAbsent*/);
   }
}

void PatchProbe::ioParam_kxPre(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ || patchIDMethod == COORDINATE_METHOD) {
      parent->ioParamValue(ioFlag, name, "kxPre", &kxPre, INT_MIN, false/*warnIfAbsent*/);
   }
}

void PatchProbe::ioParam_kyPre(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ || patchIDMethod == COORDINATE_METHOD) {
      parent->ioParamValue(ioFlag, name, "kyPre", &kyPre, INT_MIN, false/*warnIfAbsent*/);
   }
}

void PatchProbe::ioParam_kfPre(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ || patchIDMethod == COORDINATE_METHOD) {
      parent->ioParamValue(ioFlag, name, "kfPre", &kfPre, INT_MIN, false/*warnIfAbsent*/);
   }
}

void PatchProbe::ioParam_arborID(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "arborID", &arborID);
}

void PatchProbe::ioParam_outputWeights(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "outputWeights", &outputWeights, true/*default value*/);
}

void PatchProbe::ioParam_outputPlasticIncr(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "outputPlasticIncr", &outputPlasticIncr);
}

void PatchProbe::ioParam_outputPostIndices(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "outputPostIndices", &outputPostIndices);
}

int PatchProbe::getPatchID() {
   int indexmethod = kPre >= 0;
   int coordmethod = kxPre >= 0 && kyPre >= 0 && kfPre >= 0;
   if( indexmethod && coordmethod ) {
      fprintf(stderr, "%s \"%s\": Ambiguous definition with both kPre and (kxPre,kyPre,kfPre) defined\n", parent->parameters()->groupKeywordFromName(name), name);
      return PV_FAILURE;
   }
   if( !indexmethod && !coordmethod ) {
      fprintf(stderr, "%s \"%s\": Exactly one of kPre and (kxPre,kyPre,kfPre) must be defined\n", parent->parameters()->groupKeywordFromName(name), name);
      return PV_FAILURE;
   }
   if (indexmethod) {
      patchIDMethod = INDEX_METHOD;
   }
   else if (coordmethod) {
      patchIDMethod = COORDINATE_METHOD;
   }
   else assert(false);
   return PV_SUCCESS;
}

int PatchProbe::communicateInitInfo() {
   int status = PV_SUCCESS;
   assert(targetConn);
   targetHyPerConn = dynamic_cast<HyPerConn *>(targetConn);
   if (targetHyPerConn==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "KernelProbe \"%s\" error: targetConn \"%s\" must be a HyPerConn or HyPerConn-derived class.\n",
               this->getName(), targetConn->getName());
      }
      status = PV_FAILURE;
   }
#ifdef PV_USE_MPI
   MPI_Barrier(parent->icCommunicator()->communicator());
#endif
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }
   return status;
}

/**
 * kPre lives in the extended space
 *
 * NOTES:
 *    - kPre is the linear index of the neuron in the extended space.
 *
 */
int PatchProbe::outputState(double timef)
{
   HyPerConn * c = getTargetHyPerConn();
#ifdef PV_USE_MPI
   InterColComm * icComm = c->getParent()->icCommunicator();
   const int rank = icComm->commRank();
   const int size = icComm->commSize();
   MPI_Comm mpi_comm = icComm->communicator();
   int basetag = 100;
#else
   const int rank = 0;
#endif // PV_USE_MPI
   FILE * fp = getStream()->fp;
   int kPre, kxPre, kyPre, kfPre;
   const PVLayerLoc * loc = c->preSynapticLayer()->getLayerLoc();
   int nxGlobal = loc->nxGlobal;
   int nyGlobal = loc->nyGlobal;
   int nf = loc->nf;
   int nxext = nxGlobal+loc->halo.lt+loc->halo.rt;
   int nyext = nyGlobal+loc->halo.dn+loc->halo.up;
   int numPatches = c->preSynapticLayer()->getNumGlobalExtended();
   if( patchIDMethod == INDEX_METHOD ) {
      kPre = this->kPre;
      kxPre = kxPos(this->kPre,nxext,nyext,loc->nf)-loc->halo.lt;
      kyPre = kyPos(this->kPre,nxext,nyext,loc->nf)-loc->halo.up;
      kfPre = featureIndex(this->kPre,nxext,nyext,loc->nf);
   }
   else if( patchIDMethod == COORDINATE_METHOD ) {
      kPre = kIndex(this->kxPre,this->kyPre,this->kfPre,loc->nx+loc->halo.lt+loc->halo.rt,loc->ny+loc->halo.dn+loc->halo.up,loc->nf);
      kxPre = this->kxPre;
      kyPre = this->kyPre;
      kfPre = this->kfPre;
   }
   else assert(false);
   bool errorFound = false;
   if( kPre < 0 || kPre >= numPatches ) {
      fprintf(stderr, "PatchProbe \"%s\" of connection \"%s\": index is out of bounds\n", getName(), c->getName());
      fprintf(stderr, "    value is %d\n (should be between %d and %d)\n", kPre, 0, numPatches);
      errorFound = true;
   }
   if( kxPre < -loc->halo.lt || kxPre >= nxGlobal+loc->halo.rt ) {
      fprintf(stderr, "PatchProbe \"%s\" of connection \"%s\": x-coordinate is out of bounds\n", getName(), c->getName());
      fprintf(stderr, "    value is %d\n (should be between %d and %d)\n", kxPre, -loc->halo.lt, nxGlobal+loc->halo.rt);
      errorFound = true;
   }
   if( kyPre < 0 || kyPre >= numPatches ) {
      fprintf(stderr, "PatchProbe \"%s\" of connection \"%s\": y-coordinate is out of bounds\n", getName(), c->getName());
      fprintf(stderr, "    value is %d\n (should be between %d and %d)\n", kyPre, -loc->halo.up, nyGlobal+loc->halo.dn);
      errorFound = true;
   }
   if( kfPre < 0 || kfPre >= numPatches ) {
      fprintf(stderr, "PatchProbe \"%s\" of connection \"%s\": feature coordinate is out of bounds\n", getName(), c->getName());
      fprintf(stderr, "    value is %d\n (should be between %d and %d)\n", kfPre, 0, nf);
      errorFound = true;
   }
   if( arborID < 0 || arborID >= c->numberOfAxonalArborLists() ) {
      fprintf(stderr, "PatchProbe \"%s\" of connection \"%s\": arbor index is out of bounds\n", getName(), c->getName());
      fprintf(stderr, "    value is %d\n (should be between %d and %d)\n", arborID, 0, c->numberOfAxonalArborLists());
      errorFound = true;
   }
   if( errorFound ) return PV_FAILURE;

// Now convert from global coordinates to local coordinates
   int kxPreLocal = kxPre - loc->kx0;
   int kyPreLocal = kyPre - loc->ky0;
   int nxLocal = loc->nx;
   int nyLocal = loc->ny;
   int kLocal = kIndex(kxPreLocal+loc->halo.lt,kyPreLocal+loc->halo.up,kfPre,nxLocal+loc->halo.lt+loc->halo.rt,nyLocal+loc->halo.dn+loc->halo.up,nf);
   assert(kLocal >=0 && kLocal < c->preSynapticLayer()->getNumExtended());

#ifdef PV_USE_MPI
   int inbounds = kxPreLocal < -loc->halo.lt || kxPreLocal > loc->nx+loc->halo.rt || kyPreLocal < -loc->halo.up || kyPreLocal > loc->ny+loc->halo.dn;
#endif //PV_USE_MPI
   if( rank > 0 ) {
#ifdef PV_USE_MPI
      MPI_Send(&inbounds, 1, MPI_INT, 0, basetag+rank, mpi_comm);
      if(inbounds) {
         // TODO send shrunken patch to process zero
      }
#endif // PV_USE_MPI
   } else {
#ifdef PV_USE_MPI
      if(inbounds) {
         fprintf(fp, "Time %f: rank %d process is in bounds for index %d, x=%d, y=%d, f=%d\n", timef, 0, kPre, kxPre, kyPre, kfPre);
      }
      for( int src=1; src<size; src++ ) {
         int procinbounds;
         MPI_Recv(&procinbounds, 1, MPI_INT, 0, basetag+src, mpi_comm, MPI_STATUS_IGNORE);
         if(procinbounds) {
            fprintf(fp, "Time %f: rank %d process is in bounds for index %d, x=%d, y=%d, f=%d\n", timef, src, kPre, kxPre, kyPre, kfPre);
            // TODO receive shrunken patch from process rank
         }
      }
      // TODO output shrunken patch
#endif // PV_USE_MPI
   }

   PVPatch * w = c->getWeights(kPre, arborID);
   fprintf(fp, "w%d:      \n", kPre);
   if( outputWeights ) {
      text_write_patch(fp, w->nx, w->ny, c->fPatchSize(), c->xPatchStride(), c->yPatchStride(), c->fPatchStride(), c->get_wData(arborID, kPre));
   }
   if( outputPlasticIncr ) {
      pvwdata_t * plasticPatch = c->getPlasticIncr(kPre,arborID);
      if( plasticPatch )
         text_write_patch(fp, w->nx, w->ny, c->fPatchSize(), c->xPatchStride(), c->yPatchStride(), c->fPatchStride(), c->get_dwData(arborID, kPre));
   }
   if (outputPostIndices) {
      int kPost = c->getAPostOffset(kPre, arborID);
      const PVLayerLoc * lPostLoc = c->postSynapticLayer()->getLayerLoc();

      const int nxPostExt = lPostLoc->nx + lPostLoc->halo.lt + lPostLoc->halo.rt;
      const int nyPostExt = lPostLoc->ny + lPostLoc->halo.dn + lPostLoc->halo.up;
      const int nfPost = lPostLoc->nf;

      //const int kxPost = kxPos(kPost, nxPost, nyPost, nfPost) - lPostLoc->nPad;;
      //const int kyPost = kyPos(kPost, nxPost, nyPost, nfPost) - lPostLoc->nPad;;
      int kxPost = kxPos(kPost, nxPostExt, nyPostExt, nfPost) - lPostLoc->halo.lt;
      int kyPost = kyPos(kPost, nxPostExt, nyPostExt, nfPost) - lPostLoc->halo.up;

      //
      // The following is incorrect because w->nx is reduced near boundary.
      // Remove when verified.
      //
      //int kxPost = zPatchHead(kxPre, w->nxGlobal, lPre->xScale, lPost->xScale);
      //int kyPost = zPatchHead(kyPre, w->nyGlobal, lPre->yScale, lPost->yScale);


      write_patch_indices(fp, w, lPostLoc, kxPost, kyPost, 0);
      fflush(fp);
   } // if(outputIndices)
   return PV_SUCCESS;
}

int PatchProbe::text_write_patch(FILE * fp, int nx, int ny, int nf, int sx, int sy, int sf, pvwdata_t * data)
{
   int f, i, j;

//   const int nx = patch->nx;
//   const int ny = patch->ny;
//   const int nf = patch->nf;

//   const int sx = patch->sx;  assert(sx == nf);
//   const int sy = patch->sy;  //assert(sy == nf*nx); // stride could be weird at border
//   const int sf = patch->sf;  assert(sf == 1);

   assert(sf == 1);
   assert(sx == nf);
   // Do not assert(sy == nf*nx) because patch could be shrunken

   assert(fp != NULL);

   for (f = 0; f < nf; f++) {
      fprintf(fp, "f = %i\n  ", f);
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            fprintf(fp, "%5.3f ", data[i*sx + j*sy + f*sf]);
         }
         fprintf(fp, "\n  ");
      }
      fprintf(fp, "\n");
   }

   return 0;
}

/**
 * Write out the layer indices of the positions in a patch.
 * The inputs to the function (patch,loc) can either be from
 * the point of view of the pre- or post-synaptic layer.
 *
 * @patch the patch to iterate over
 * @loc the location information in the layer that the patch projects to
 * @nf the number of features in the patch (should be the same as in the layer)
 * @kx0 the kx index location of the head (neuron) of the patch projection
 * @ky0 the ky index location of the head of the patch projection
 * @kf0 the kf index location of the head of the patch (should be 0)
 *
 * NOTES:
 *    - indices are in the local, restricted space.
 *    - kx0, ky0, are pre patch heads.
 *
 */
int PatchProbe::write_patch_indices(FILE * fp, PVPatch * patch,
      const PVLayerLoc * loc, int kx0, int ky0, int kf0)
{
   int f, i, j;

   const int nx = patch->nx;
   const int ny = patch->ny;
   const int nf = loc->nf; // patch->nf;

   // these strides are from the layer, not the patch
   // NOTE: assumes nf from layer == nf from patch
   //
   const int sx = nf;
   const int sy = loc->nx * nf;

   assert(fp != NULL);

   const int k0 = kIndex(kx0, ky0, kf0, loc->nx, loc->ny, nf);

   fprintf(fp, "  ");

   // loop over patch indices (writing out layer indices)
   //
   for (f = 0; f < nf; f++) {
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            int kf = f;
            int kx = kx0 + i;
            int ky = ky0 + j;
            int k  = k0 + kf + i*sx + j*sy;
            //fprintf(fp, "(%4d, (%4d,%4d,%4d)) ", k, kx, ky, kf);
            fprintf(fp, "%4d %4d %4d %4d  ", k, kx, ky, kf);
         }
         fprintf(fp, "\n  ");
      }
      fprintf(fp, "\n");
   }

   return 0;
}

} // namespace PV
