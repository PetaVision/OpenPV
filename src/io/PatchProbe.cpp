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

PatchProbe::PatchProbe(const char * probename, const char * filename, HyPerConn * conn, int kPre, int arbID)
{
   initialize_base();
   initialize(probename, filename, conn, INDEX_METHOD, kPre, -1, -1, -1, arbID);
}

PatchProbe::PatchProbe(const char * probename, const char * filename, HyPerConn * conn, int kxPre, int kyPre, int kfPre, int arbID)
{
   initialize_base();
   initialize(probename, filename, conn, COORDINATE_METHOD, -1, kxPre, kyPre, kfPre, arbID);
}
PatchProbe::~PatchProbe()
{
}

int PatchProbe::initialize_base() {
   return PV_SUCCESS;
}

int PatchProbe::initialize(const char * probename, const char * filename,
      HyPerConn * conn, PatchIDMethod method, int kPre,
      int kxPre, int kyPre, int kfPre, int arbID) {
   if( method == INDEX_METHOD ) {
      this->kPre = kPre;
      this->kxPre = INT_MIN;
      this->kyPre = INT_MIN;
      this->kfPre = INT_MIN;
      patchIDMethod = method;
   }
   else if( method == COORDINATE_METHOD ) {
      this->kPre = INT_MIN;
      this->kxPre = kxPre;
      this->kyPre = kyPre;
      this->kfPre = kfPre;
   }
   else assert(false);
   patchIDMethod = method;
   arborID = arbID;
   outputWeights = true; // set by setOutputWeights method
   outputPlasticIncr = false; // set by setOutputPlasticIncr method
   outputPostIndices = false; // set by setOutputPostIndices method
   return BaseConnectionProbe::initialize(getName(), filename, conn);
}

/**
 * kPre lives in the extended space
 *
 * NOTES:
 *    - kPre is the linear index of the neuron in the extended space.
 *
 */
int PatchProbe::outputState(float timef)
{
   HyPerConn * c = getTargetConn();
#ifdef PV_USE_MPI
   InterColComm * icComm = c->getParent()->icCommunicator();
   const int rank = icComm->commRank();
   const int size = icComm->commSize();
   MPI_Comm mpi_comm = icComm->communicator();
   int basetag = 100;
#else
   const int rank = 0;
#endif // PV_USE_MPI
   FILE * fp = getFilePtr();
   int kPre, kxPre, kyPre, kfPre;
   const PVLayerLoc * loc = c->preSynapticLayer()->getLayerLoc();
   int nxGlobal = loc->nxGlobal;
   int nyGlobal = loc->nyGlobal;
   int nf = loc->nf;
   int nb = loc->nb;
   int nxext = nxGlobal+2*nb;
   int nyext = nyGlobal+2*nb;
   int numPatches = c->preSynapticLayer()->getNumGlobalExtended();
   if( patchIDMethod == INDEX_METHOD ) {
      kPre = this->kPre;
      kxPre = kxPos(kPre,nxext,nyext,loc->nf)-nb;
      kyPre = kyPos(kPre,nxext,nyext,loc->nf)-nb;
      kfPre = featureIndex(kPre,nxext,nyext,loc->nf);
   }
   else if( patchIDMethod == COORDINATE_METHOD ) {
      kPre = kIndex(kxPre,kyPre,kfPre,loc->nx+2*nb,loc->ny+2*nb,loc->nf);
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
   if( kxPre < -nb || kxPre >= nxGlobal+nb ) {
      fprintf(stderr, "PatchProbe \"%s\" of connection \"%s\": x-coordinate is out of bounds\n", getName(), c->getName());
      fprintf(stderr, "    value is %d\n (should be between %d and %d)\n", kxPre, -nb, nxGlobal+nb);
      errorFound = true;
   }
   if( kyPre < 0 || kyPre >= numPatches ) {
      fprintf(stderr, "PatchProbe \"%s\" of connection \"%s\": y-coordinate is out of bounds\n", getName(), c->getName());
      fprintf(stderr, "    value is %d\n (should be between %d and %d)\n", kyPre, 0, nyGlobal+nb);
      errorFound = true;
   }
   if( kfPre < 0 || kfPre >= numPatches ) {
      fprintf(stderr, "PatchProbe \"%s\" of connection \"%s\": feature coordinate is out of bounds\n", getName(), c->getName());
      fprintf(stderr, "    value is %d\n (should be between %d and %d)\n", kfPre, 0, nf);
      errorFound = true;
   }
   if( arborID < 0 || arborID >= numPatches ) {
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
   int kLocal = kIndex(kxPreLocal+nb,kyPreLocal+nb,kfPre,nxLocal+2*nb,nyLocal+2*nb,nf);
   assert(kLocal >=0 && kLocal < c->preSynapticLayer()->getNumExtended());

#ifdef PV_USE_MPI
   int inbounds = kxPreLocal < -nb || kxPreLocal > loc->nx+nb || kyPreLocal < -nb || kyPreLocal > loc->ny+nb;
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
      pvdata_t * plasticPatch = c->getPlasticIncr(kPre,arborID);
      if( plasticPatch )
         text_write_patch(fp, w->nx, w->ny, c->fPatchSize(), c->xPatchStride(), c->yPatchStride(), c->fPatchStride(), c->get_dwData(arborID, kPre));
   }
   if (outputPostIndices) {
      int kPost = c->getAPostOffset(kPre, arborID);
      const PVLayerLoc * lPostLoc = c->postSynapticLayer()->getLayerLoc();

      const int nxPostExt = lPostLoc->nx + 2*lPostLoc->nb;
      const int nyPostExt = lPostLoc->ny + 2*lPostLoc->nb;
      const int nfPost = lPostLoc->nf;

      //const int kxPost = kxPos(kPost, nxPost, nyPost, nfPost) - lPostLoc->nPad;;
      //const int kyPost = kyPos(kPost, nxPost, nyPost, nfPost) - lPostLoc->nPad;;
      int kxPost = kxPos(kPost, nxPostExt, nyPostExt, nfPost) - lPostLoc->nb;
      int kyPost = kyPos(kPost, nxPostExt, nyPostExt, nfPost) - lPostLoc->nb;

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

int PatchProbe::text_write_patch(FILE * fp, int nx, int ny, int nf, int sx, int sy, int sf, pvdata_t * data)
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
