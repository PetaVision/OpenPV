/*
 * BoundaryConditions.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "BoundaryConditions.hpp"

namespace PV {

BoundaryConditions::BoundaryConditions(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

BoundaryConditions::BoundaryConditions() {}

BoundaryConditions::~BoundaryConditions() {}

void BoundaryConditions::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void BoundaryConditions::setObjectType() { mObjectType = "BoundaryConditions"; }

int BoundaryConditions::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_mirrorBCflag(ioFlag);
   ioParam_valueBC(ioFlag);
   return PV_SUCCESS;
}

void BoundaryConditions::ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "mirrorBCflag", &mMirrorBCflag, mMirrorBCflag);
}

void BoundaryConditions::ioParam_valueBC(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "mirrorBCflag"));
   if (!mMirrorBCflag) {
      parameters()->ioParamValue(ioFlag, name, "valueBC", &mValueBC, mValueBC);
   }
}

Response::Status
BoundaryConditions::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return BaseObject::communicateInitInfo(message);
}

void BoundaryConditions::applyBoundaryConditions(float *buffer, PVLayerLoc const *loc) const {
   if (getMirrorBCflag()) {
      mirrorInteriorToBorder(buffer, buffer, loc);
   }
   else {
      // No need to do anything here; currently the only other boundary condition choice
      // is to fill the border region with ValueBC; this is done during initialization, and nothing
      // ever modifies the border region in that case.
   }
}

void BoundaryConditions::mirrorInteriorToBorder(
      float const *srcBuffer,
      float *destBuffer,
      PVLayerLoc const *loc) const {
   mirrorToNorthWest(srcBuffer, destBuffer, loc);
   mirrorToNorth(srcBuffer, destBuffer, loc);
   mirrorToNorthEast(srcBuffer, destBuffer, loc);
   mirrorToWest(srcBuffer, destBuffer, loc);
   mirrorToEast(srcBuffer, destBuffer, loc);
   mirrorToSouthWest(srcBuffer, destBuffer, loc);
   mirrorToSouth(srcBuffer, destBuffer, loc);
   mirrorToSouthEast(srcBuffer, destBuffer, loc);
}

void BoundaryConditions::mirrorToNorthWest(
      float const *srcBuffer,
      float *destBuffer,
      PVLayerLoc const *loc) const {
   int nbatch     = loc->nbatch;
   int nf         = loc->nf;
   int leftBorder = loc->halo.lt;
   int topBorder  = loc->halo.up;
   size_t sb      = strideBExtended(loc);
   size_t sf      = strideFExtended(loc);
   size_t sx      = strideXExtended(loc);
   size_t sy      = strideYExtended(loc);

   for (int b = 0; b < nbatch; b++) {
      float const *srcData = srcBuffer + b * sb;
      float *destData      = destBuffer + b * sb;

      float const *src0 = srcData + topBorder * sy + leftBorder * sx;
      float *dst0       = destData + (topBorder - 1) * sy + (leftBorder - 1) * sx;

      for (int ky = 0; ky < topBorder; ky++) {
         float *to         = dst0 - ky * sy;
         float const *from = src0 + ky * sy;
         for (int kx = 0; kx < leftBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to -= nf;
            from += nf;
         }
      }
   }
}

void BoundaryConditions::mirrorToNorth(
      float const *srcBuffer,
      float *destBuffer,
      PVLayerLoc const *loc) const {
   int nx         = loc->nx;
   int nf         = loc->nf;
   int leftBorder = loc->halo.lt;
   int topBorder  = loc->halo.up;
   int nbatch     = loc->nbatch;
   size_t sb      = strideBExtended(loc);
   size_t sf      = strideFExtended(loc);
   size_t sx      = strideXExtended(loc);
   size_t sy      = strideYExtended(loc);

   for (int b = 0; b < nbatch; b++) {
      float const *srcData = srcBuffer + b * sb;
      float *destData      = destBuffer + b * sb;
      float const *src0    = srcData + topBorder * sy + leftBorder * sx;
      float *dst0          = destData + (topBorder - 1) * sy + leftBorder * sx;

      for (int ky = 0; ky < topBorder; ky++) {
         float *to         = dst0 - ky * sy;
         float const *from = src0 + ky * sy;
         for (int kx = 0; kx < nx; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to += nf;
            from += nf;
         }
      }
   }
}

void BoundaryConditions::mirrorToNorthEast(
      float const *srcBuffer,
      float *destBuffer,
      PVLayerLoc const *loc) const {
   int nx          = loc->nx;
   int nf          = loc->nf;
   int leftBorder  = loc->halo.lt;
   int rightBorder = loc->halo.rt;
   int topBorder   = loc->halo.up;
   int nbatch      = loc->nbatch;
   size_t sb       = strideBExtended(loc);
   size_t sf       = strideFExtended(loc);
   size_t sx       = strideXExtended(loc);
   size_t sy       = strideYExtended(loc);

   for (int b = 0; b < nbatch; b++) {
      float const *srcData = srcBuffer + b * sb;
      float *destData      = destBuffer + b * sb;
      float const *src0    = srcData + topBorder * sy + (nx + leftBorder - 1) * sx;
      float *dst0          = destData + (topBorder - 1) * sy + (nx + leftBorder) * sx;

      for (int ky = 0; ky < topBorder; ky++) {
         float *to         = dst0 - ky * sy;
         float const *from = src0 + ky * sy;
         for (int kx = 0; kx < rightBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to += nf;
            from -= nf;
         }
      }
   }
}

void BoundaryConditions::mirrorToWest(
      float const *srcBuffer,
      float *destBuffer,
      PVLayerLoc const *loc) const {
   int ny         = loc->ny;
   int nf         = loc->nf;
   int leftBorder = loc->halo.lt;
   int topBorder  = loc->halo.up;
   int nbatch     = loc->nbatch;
   size_t sb      = strideBExtended(loc);
   size_t sf      = strideFExtended(loc);
   size_t sx      = strideXExtended(loc);
   size_t sy      = strideYExtended(loc);

   for (int b = 0; b < nbatch; b++) {
      float const *srcData = srcBuffer + b * sb;
      float *destData      = destBuffer + b * sb;
      float const *src0    = srcData + topBorder * sy + leftBorder * sx;
      float *dst0          = destData + topBorder * sy + (leftBorder - 1) * sx;

      for (int ky = 0; ky < ny; ky++) {
         float *to         = dst0 + ky * sy;
         float const *from = src0 + ky * sy;
         for (int kx = 0; kx < leftBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to -= nf;
            from += nf;
         }
      }
   }
}

void BoundaryConditions::mirrorToEast(
      float const *srcBuffer,
      float *destBuffer,
      PVLayerLoc const *loc) const {
   int nx          = loc->nx;
   int ny          = loc->ny;
   int nf          = loc->nf;
   int leftBorder  = loc->halo.lt;
   int rightBorder = loc->halo.rt;
   int topBorder   = loc->halo.up;
   int nbatch      = loc->nbatch;
   size_t sb       = strideBExtended(loc);
   size_t sf       = strideFExtended(loc);
   size_t sx       = strideXExtended(loc);
   size_t sy       = strideYExtended(loc);

   for (int b = 0; b < nbatch; b++) {
      float const *srcData = srcBuffer + b * sb;
      float *destData      = destBuffer + b * sb;
      float const *src0    = srcData + topBorder * sy + (nx + leftBorder - 1) * sx;
      float *dst0          = destData + topBorder * sy + (nx + leftBorder) * sx;

      for (int ky = 0; ky < ny; ky++) {
         float *to         = dst0 + ky * sy;
         float const *from = src0 + ky * sy;
         for (int kx = 0; kx < rightBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to += nf;
            from -= nf;
         }
      }
   }
}

void BoundaryConditions::mirrorToSouthWest(
      float const *srcBuffer,
      float *destBuffer,
      PVLayerLoc const *loc) const {
   int ny           = loc->ny;
   int nf           = loc->nf;
   int leftBorder   = loc->halo.lt;
   int topBorder    = loc->halo.up;
   int bottomBorder = loc->halo.dn;
   int nbatch       = loc->nbatch;
   size_t sb        = strideBExtended(loc);
   size_t sf        = strideFExtended(loc);
   size_t sx        = strideXExtended(loc);
   size_t sy        = strideYExtended(loc);

   for (int b = 0; b < nbatch; b++) {
      float const *srcData = srcBuffer + b * sb;
      float *destData      = destBuffer + b * sb;
      float const *src0    = srcData + (ny + topBorder - 1) * sy + leftBorder * sx;
      float *dst0          = destData + (ny + topBorder) * sy + (leftBorder - 1) * sx;

      for (int ky = 0; ky < bottomBorder; ky++) {
         float *to         = dst0 + ky * sy;
         float const *from = src0 - ky * sy;
         for (int kx = 0; kx < leftBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to -= nf;
            from += nf;
         }
      }
   }
}

void BoundaryConditions::mirrorToSouth(
      float const *srcBuffer,
      float *destBuffer,
      PVLayerLoc const *loc) const {
   int nx           = loc->nx;
   int ny           = loc->ny;
   int nf           = loc->nf;
   int leftBorder   = loc->halo.lt;
   int rightBorder  = loc->halo.rt;
   int topBorder    = loc->halo.up;
   int bottomBorder = loc->halo.dn;
   int nbatch       = loc->nbatch;
   size_t sb        = strideBExtended(loc);
   size_t sf        = strideFExtended(loc);
   size_t sx        = strideXExtended(loc);
   size_t sy        = strideYExtended(loc);

   for (int b = 0; b < nbatch; b++) {
      float const *srcData = srcBuffer + b * sb;
      float *destData      = destBuffer + b * sb;
      float const *src0    = srcData + (ny + topBorder - 1) * sy + leftBorder * sx;
      float *dst0          = destData + (ny + topBorder) * sy + leftBorder * sx;

      for (int ky = 0; ky < bottomBorder; ky++) {
         float *to         = dst0 + ky * sy;
         float const *from = src0 - ky * sy;
         for (int kx = 0; kx < nx; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to += nf;
            from += nf;
         }
      }
   }
}

void BoundaryConditions::mirrorToSouthEast(
      float const *srcBuffer,
      float *destBuffer,
      PVLayerLoc const *loc) const {
   int nx           = loc->nx;
   int ny           = loc->ny;
   int nf           = loc->nf;
   int leftBorder   = loc->halo.lt;
   int rightBorder  = loc->halo.rt;
   int topBorder    = loc->halo.up;
   int bottomBorder = loc->halo.dn;
   int nbatch       = loc->nbatch;
   size_t sb        = strideBExtended(loc);
   size_t sf        = strideFExtended(loc);
   size_t sx        = strideXExtended(loc);
   size_t sy        = strideYExtended(loc);

   for (int b = 0; b < nbatch; b++) {
      float const *srcData = srcBuffer + b * sb;
      float *destData      = destBuffer + b * sb;
      float const *src0    = srcData + (ny + topBorder - 1) * sy + (nx + leftBorder - 1) * sx;
      float *dst0          = destData + (ny + topBorder) * sy + (nx + leftBorder) * sx;

      for (int ky = 0; ky < bottomBorder; ky++) {
         float *to         = dst0 + ky * sy;
         float const *from = src0 - ky * sy;
         for (int kx = 0; kx < rightBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to += nf;
            from -= nf;
         }
      }
   }
}

void BoundaryConditions::fillWithValue(float *buffer, PVLayerLoc const *loc) const {
   int const nx = loc->nx;
   int const ny = loc->ny;
   int const nf = loc->nf;
   int idx      = 0;
   for (int batch = 0; batch < loc->nbatch; batch++) {
      for (int b = 0; b < loc->halo.up; b++) {
         for (int k = 0; k < (nx + loc->halo.lt + loc->halo.rt) * nf; k++) {
            buffer[idx] = getValueBC();
            idx++;
         }
      }
      for (int y = 0; y < ny; y++) {
         for (int k = 0; k < loc->halo.lt * nf; k++) {
            buffer[idx] = getValueBC();
            idx++;
         }
         idx += nx * nf;
         for (int k = 0; k < loc->halo.rt * nf; k++) {
            buffer[idx] = getValueBC();
            idx++;
         }
      }
      for (int b = 0; b < loc->halo.dn; b++) {
         for (int k = 0; k < (nx + loc->halo.lt + loc->halo.rt) * nf; k++) {
            buffer[idx] = getValueBC();
            idx++;
         }
      }
   }
}

} // namespace PV
