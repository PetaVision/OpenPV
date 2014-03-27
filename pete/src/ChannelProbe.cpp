/*
 * ChannelProbe.cpp
 *
 *  Created on: Nov 22, 2010
 *      Author: pschultz
 */

#include "ChannelProbe.hpp"

namespace PV {

ChannelProbe::ChannelProbe(const char * probeName, HyPerCol * hc) : LayerProbe() {
   initChannelProbe_base();
   initChannelProbe(probeName, hc);
}  // end ChannelProbe::ChannelProbe(const char *, HyPerCol *)

int ChannelProbe::initChannelProbe_base() {
   pChannel = CHANNEL_EXC;
   return PV_SUCCESS;
}

int ChannelProbe::initChannelProbe(const char * probeName, HyPerCol * hc) {
   int status = initLayerProbe(probeName, hc);
   return status;
}

int ChannelProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_channelCode(ioFlag);
   return status;
}

void ChannelProbe::ioParam_channelCode(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      int ch = 0;
      getParentCol()->ioParamValueRequired(ioFlag, getProbeName(), "channelCode", &ch);
      int status = HyPerConn::decodeChannel(ch, &pChannel);
      if (status != PV_SUCCESS) {
         if (getParentCol()->columnId()==0) {
            fprintf(stderr, "%s \"%s\": channelCode %d is not a valid channel.\n",
                  getParentCol()->parameters()->groupKeywordFromName(getProbeName()), getProbeName(),  ch);
         }
         MPI_Barrier(getParentCol()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   else if (ioFlag==PARAMS_IO_WRITE) {
      int ch = (int) pChannel;
      getParentCol()->ioParamValueRequired(ioFlag, getProbeName(), "channelCode", &ch);
   }
   else {
      assert(0); // All possibilities of ioFlag are covered above.
   }
}

int ChannelProbe::outputState(double timed) {
    pvdata_t * buf = getTargetLayer()->getChannel(pChannel);
    int n = getTargetLayer()->getNumNeurons();
    for( int k=0; k<n; k++) {
        fprintf(outputstream->fp, "Layer %s, channel %d, time %f, neuron %8d, value=%.8g\n",
        		getTargetLayer()->getName(), (int) pChannel, timed, k, buf[k]);
    }
    return EXIT_SUCCESS;
}  // end ChannelProbe::outputState(float, HyPerLayer *)

}  // end namespace PV
