#include "randomstateio.hpp"
#include "io/FileStream.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"

namespace PV {

double readRandState(
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
   double timestamp;
   if (comm->commRank() == 0) {
      timestamp = BufferUtils::readFromPvp(path.c_str(), &buffer, 0 /*frameReadIndex*/);
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
   MPI_Bcast(&timestamp, 1, MPI_DOUBLE, 0 /*root proc*/, comm->communicator());
   return timestamp;
}

void writeRandState(
      std::string const &path,
      Communicator *comm,
      taus_uint4 const *randState,
      PVLayerLoc const *loc,
      bool extended,
      double simTime,
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
      BufferUtils::writeToPvp<taus_uint4>(path.c_str(), &globalBuffer, simTime, verifyWrites);
   }
}

namespace BufferUtils {

template <>
HeaderDataType returnDataType<taus_uint4>() { return TAUS_UINT4; }

}  // end namespace BufferUtils

} // end namespace PV
