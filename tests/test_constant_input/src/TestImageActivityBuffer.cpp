/*
 * TestImageActivityBuffer.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "TestImageActivityBuffer.hpp"

namespace PV {

TestImageActivityBuffer::TestImageActivityBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

TestImageActivityBuffer::~TestImageActivityBuffer() {}

void TestImageActivityBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ActivityBuffer::initialize(paramsIO, comm);
}

void TestImageActivityBuffer::setObjectType() { mObjectType = "TestImageActivityBuffer"; }

int TestImageActivityBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = ActivityBuffer::ioParamsFillGroup(ioSwitch);
   ioParam_constantVal(ioSwitch);
   return status;
}

void TestImageActivityBuffer::ioParam_constantVal(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "constantVal", &mConstantVal);
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
