/* FeedbackConn.cpp
 *
 * Created on: Nov 15, 2010
 *     Author: peteschultz
 */

#include "FeedbackConn.hpp"
#include "components/FeedbackConnectionData.hpp"

namespace PV {

FeedbackConn::FeedbackConn(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

FeedbackConn::FeedbackConn() {}

FeedbackConn::~FeedbackConn() {}

void FeedbackConn::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   TransposeConn::initialize(paramsIO, comm);
}

ConnectionData *FeedbackConn::createConnectionData() {
   return new FeedbackConnectionData(mParamsIO, mCommunicator);
}

} // namespace PV
