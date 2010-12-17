/*
 * GenerativeConn.cpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#include "GenerativeConn.hpp"

namespace PV {

GenerativeConn::GenerativeConn() {
    initialize_base();
}  // end of GenerativeConn::GenerativeConn()

GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post) {
       initialize_base();
       initialize(name, hc, pre, post, channel, NULL);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *)

GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, int channel) {
       initialize_base();
       initialize(name, hc, pre, post, channel);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, int)

GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, int channel,
        const char * filename) {
       initialize_base();
       initialize(name, hc, pre, post, channel, filename);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
   //   HyPerLayer *, HyPerLayer *, int, const char *)

int GenerativeConn::initialize_base() {
    weightUpdatePeriod = 1.0;
    nextWeightUpdate = weightUpdatePeriod;
    relaxation = 1.0;
    return EXIT_SUCCESS;
    // KernelConn constructor calls KernelConn::initialize_base()
    // and the similarly for HyPerConn constructor, so I don't need to.
}

int GenerativeConn::initialize(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, int channel) {
    return initialize(name, hc, pre, post, channel, NULL);
}

int GenerativeConn::initialize(const char * name, HyPerCol * hc,
        HyPerLayer * pre, HyPerLayer * post, int channel,
        const char * filename) {
    KernelConn::initialize(name, hc, pre, post, channel, filename);
    weightUpdatePeriod = parent->parameters()->value(name, "weightUpdatePeriod", 1.0f);
    nextWeightUpdate = weightUpdatePeriod;
    relaxation = parent->parameters()->value(name, "relaxation", 1.0f);
    return EXIT_SUCCESS;
}

int GenerativeConn::updateState(float time, float dt) {
    if(time > nextWeightUpdate) {
        nextWeightUpdate += weightUpdatePeriod;
        updateWeights(0);
    }
    return EXIT_SUCCESS;
}  // end of GenerativeConn::updateState(float, float)

int GenerativeConn::updateWeights(int axonID) {
    printf("updateWeights for connection %s\n", name);

    int nPre = preSynapticLayer()->getNumNeurons();
    for(int kPre=0; kPre<nPre;kPre++) {
        int nx = preSynapticLayer()->getCLayer()->loc.nx;
        int ny = preSynapticLayer()->getCLayer()->loc.ny;
        int nf = preSynapticLayer()->getCLayer()->numFeatures;
        int pad = preSynapticLayer()->getCLayer()->loc.nPad;
        int kExt = kIndexExtended(kPre, nx, ny, nf, pad);

        PVAxonalArbor * arbor = axonalArbor(kPre, 0);
        size_t offset = arbor->offset;
        pvdata_t preact = preSynapticLayer()->getCLayer()->activity->data[kExt];
        pvdata_t a = relaxation*preact;
        int nyp = arbor->weights->ny;
        int nk = arbor->weights->nx * arbor->weights->nf;
        pvdata_t * postactRef = &(postSynapticLayer()->getCLayer()->activity->data[offset]);
        int sya = arbor->data->sy;
        pvdata_t * wtpatch = arbor->weights->data;
        int syw = arbor->weights->sy;
        for( int y=0; y<nyp; y++ ) {
            int lineoffsetw = 0;
            int lineoffseta = 0;
            for( int k=0; k<nk; k++ ) {
                wtpatch[lineoffsetw + k] += a*postactRef[lineoffseta + k];
            }
            lineoffsetw += syw;
            lineoffseta += sya;
        }
    }

    return EXIT_SUCCESS;
}  // end of GenerativeConn::updateWeights(int);

}  // end of namespace PV block
