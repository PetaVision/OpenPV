#include "randomstateio.hpp"
#include "io/FileStream.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"
#include <memory>

namespace PV {

double readRandState(
      std::string const &path,
      std::shared_ptr<MPIBlock const> mpiBlock,
      taus_uint4 *randState,
      PVLayerLoc const *loc,
      bool extended) {
   int nxGlobal = loc->nxGlobal;
   int nyGlobal = loc->nyGlobal;
   if (extended) {
      nxGlobal += loc->halo.lt + loc->halo.rt;
      nyGlobal += loc->halo.dn + loc->halo.up;
   }
   int const nf          = loc->nf;
   int const rootProcess = 0; // process that does the I/O.

   Buffer<taus_uint4> buffer{nxGlobal, nyGlobal, nf};
   std::vector<double> timestamps((size_t)(loc->nbatch * mpiBlock->getBatchDimension()));
   int nxLocal = loc->nx;
   int nyLocal = loc->ny;
   if (extended) {
      nxLocal += loc->halo.lt + loc->halo.rt;
      nyLocal += loc->halo.dn + loc->halo.up;
   }
   int const numLocal   = nxLocal * nyLocal * nf;
   std::size_t numBytes = sizeof(taus_uint4) * (std::size_t)numLocal;

   for (int m = 0; m < mpiBlock->getBatchDimension(); m++) {
      for (int b = 0; b < loc->nbatch; b++) {
         int globalBatchIndex = b + loc->nbatch * m;
         if (mpiBlock->getRank() == rootProcess) {
            timestamps[globalBatchIndex] =
                  BufferUtils::readDenseFromPvp(path.c_str(), &buffer, globalBatchIndex);
         }
         BufferUtils::scatter(mpiBlock, buffer, loc->nx, loc->ny, m, rootProcess);
         if (mpiBlock->getBatchIndex() == m) {
            auto localData = &randState[b * numLocal];
            memcpy(localData, buffer.asVector().data(), numBytes);
         }
      }
   }
   double timestamp;
   if (mpiBlock->getRank() == 0) {
      timestamp = timestamps[0];
      for (size_t n = (size_t)1; n < timestamps.size(); n++) {
         if (timestamps[n] != timestamp) {
            WarnLog() << "readRandState \"" << path << "\": batch element " << n
                      << " has timestamp " << timestamps[n]
                      << ", different from batch element 0 timestamp " << timestamp << "\n";
         }
      }
   }
   MPI_Bcast(&timestamp, 1, MPI_DOUBLE, 0 /*root proc*/, mpiBlock->getComm());
   return timestamp;
}

void writeRandState(
      std::string const &path,
      std::shared_ptr<MPIBlock const> mpiBlock,
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
   int const nf       = loc->nf;
   int const numLocal = nxLocal * nyLocal * nf;
   for (int m = 0; m < mpiBlock->getBatchDimension(); m++) {
      for (int b = 0; b < loc->nbatch; b++) {
         auto localData = &randState[b * numLocal];
         Buffer<taus_uint4> localBuffer{localData, nxLocal, nyLocal, nf};
         Buffer<taus_uint4> globalBuffer =
               BufferUtils::gather(mpiBlock, localBuffer, loc->nx, loc->ny, m, 0);
         if (mpiBlock->getRank() == 0) {
            if (b == 0) {
               BufferUtils::writeToPvp<taus_uint4>(
                     path.c_str(), &globalBuffer, simTime, verifyWrites);
            }
            else {
               BufferUtils::appendToPvp<taus_uint4>(
                     path.c_str(), &globalBuffer, b, simTime, verifyWrites);
            }
         }
      }
   }
}

namespace BufferUtils {

template <>
HeaderDataType returnDataType<taus_uint4>() {
   return TAUS_UINT4;
}

} // end namespace BufferUtils

} // end namespace PV
