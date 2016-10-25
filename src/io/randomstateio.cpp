#include "randomstateio.hpp"
#include "io/FileStream.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"

namespace PV {

void readRandState(
      std::string const &path,
      Communicator *comm,
      taus_uint4 *randState,
      PVLayerLoc const *loc,
      bool extended) {
   int nxGlobal = loc->nxGlobal;
   int nyGlobal = loc->nyGlobal;
   if (extended) {
      nxGlobal += loc->halo.lt + loc->halo.rt;
      nyGlobal += loc->halo.dn + loc->halo.up;
   }
   int nf        = loc->nf;
   int numGlobal = nxGlobal * nyGlobal * nf;

   Buffer<taus_uint4> buffer{nxGlobal, nyGlobal, nf};
   if (comm->commRank() == 0) {
      taus_uint4 globalData[numGlobal];
      FileStream fileStream{path.c_str(), std::ios_base::in, false /*verifyWrites, not needed*/};
      fileStream.read(globalData, numGlobal);
      buffer.set(globalData, nxGlobal, nyGlobal, nf);
   }
   BufferUtils::scatter(comm, buffer, loc->nx, loc->ny);
   int nxLocal = loc->nx;
   int nyLocal = loc->ny;
   if (extended) {
      nxLocal += loc->halo.lt + loc->halo.rt;
      nyLocal += loc->halo.dn + loc->halo.up;
   }
   int numLocal         = nxLocal * nyLocal * nf;
   std::size_t numBytes = sizeof(taus_uint4) * (std::size_t)numLocal;
   memcpy(randState, buffer.asVector().data(), numBytes);
}

void writeRandState(
      std::string const &path,
      Communicator *comm,
      taus_uint4 const *randState,
      PVLayerLoc const *loc,
      bool extended,
      bool verifyWrites) {
   int nxLocal = loc->nx;
   int nyLocal = loc->ny;
   if (extended) {
      nxLocal += loc->halo.lt + loc->halo.rt;
      nyLocal += loc->halo.dn + loc->halo.up;
   }
   int nf = loc->nf;
   Buffer<taus_uint4> localBuffer{randState, nxLocal, nyLocal, nf};
   Buffer<taus_uint4> globalBuffer = BufferUtils::gather(comm, localBuffer, loc->nx, loc->ny);
   if (comm->commRank() == 0) {
      FileStream fileStream{path.c_str(), std::ios_base::out, verifyWrites};
      fileStream.write(
            globalBuffer.asVector().data(), globalBuffer.getTotalElements() * sizeof(taus_uint4));
   }
}

} // end namespace PV
