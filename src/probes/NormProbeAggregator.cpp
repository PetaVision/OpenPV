#include "NormProbeAggregator.hpp"
#include "arch/mpi/mpi.h"
#include "cMakeHeader.h"
#include "utils/PVAssert.hpp"

namespace PV {

NormProbeAggregator::NormProbeAggregator(
      char const *objName,
      PVParams *params,
      std::shared_ptr<MPIBlock const> mpiBlock) {
   initialize(objName, params, mpiBlock);
}

void NormProbeAggregator::aggregateNormsBatch(
      ProbeData<double> &aggregatedNormsBatch,
      ProbeData<double> const &partialNormsBatch) {
   pvAssert(partialNormsBatch.size() == aggregatedNormsBatch.size());
   int nbatch = static_cast<int>(partialNormsBatch.size());
#ifdef PV_USE_MPI
   double const *partialData = &partialNormsBatch.getValue(0);
   double *aggregateData     = &aggregatedNormsBatch.getValue(0);
   MPI_Comm comm             = mMPIBlock->getComm();
   MPI_Allreduce(partialData, aggregateData, nbatch, MPI_DOUBLE, MPI_SUM, comm);
#else // PV_USE_MPI
   int transferSize = static_cast<int>(sizeof(double)) * nbatch;
   std::memcpy(&aggregatedNormsBatch.getValue(0), &partialNormsBatch.getValue(0), transferSize);
#endif // PV_USE_MPI
}

void NormProbeAggregator::aggregateStoredValues(ProbeDataBuffer<double> const &partialStore) {
   int storeSize = static_cast<int>(partialStore.size());
   for (int n = 0; n < storeSize; ++n) {
      auto &partialNorms = partialStore.getData(n);
      double timestamp   = partialNorms.getTimestamp();
      auto batchSize     = partialNorms.size();

      ProbeData<double> aggregateNorms(timestamp, batchSize);
      aggregateNormsBatch(aggregateNorms, partialNorms);
      mStoredValues.store(aggregateNorms);
   }
}

void NormProbeAggregator::clearStoredValues() { mStoredValues.clear(); }

void NormProbeAggregator::initialize(
      char const *objName,
      PVParams *params,
      std::shared_ptr<MPIBlock const> mpiBlock) {
   ProbeComponent::initialize(objName, params);
   mMPIBlock = mpiBlock;
}

void NormProbeAggregator::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {}

} // namespace PV
