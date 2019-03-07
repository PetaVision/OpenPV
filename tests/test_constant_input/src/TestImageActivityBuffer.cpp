/*
 * TestImageActivityBuffer.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "TestImageActivityBuffer.hpp"

namespace PV {

TestImageActivityBuffer::TestImageActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

TestImageActivityBuffer::~TestImageActivityBuffer() {}

void TestImageActivityBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ActivityBuffer::initialize(name, params, comm);
}

void TestImageActivityBuffer::setObjectType() { mObjectType = "TestImageActivityBuffer"; }

int TestImageActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_constantVal(ioFlag);
   return status;
}

void TestImageActivityBuffer::ioParam_constantVal(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "constantVal", &mConstantVal, mConstantVal);
}

Response::Status
TestImageActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   PVLayerLoc const *loc   = getLayerLoc();
   int const numRestricted = loc->nx * loc->ny * loc->nf;
   for (int kbatch = 0; kbatch < numRestricted * loc->nbatch; kbatch++) {
      int const k = kbatch % numRestricted;
      int kExt    = kIndexExtended(
            k, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      mBufferData[kExt] = mConstantVal;
   }
   return Response::SUCCESS;
}

void TestImageActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {}

} // namespace PV
