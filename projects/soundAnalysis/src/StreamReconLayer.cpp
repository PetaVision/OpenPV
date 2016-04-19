//
//  StreamReconLayer.cpp
//  
//
//  Created by Brohit's Mac 5054128794 on 7/29/14.
//
//

#include "StreamReconLayer.h"

StreamReconLayer::StreamReconLayer() {
    initialize_base();
}

StreamReconLayer::StreamReconLayer(const char * name, PV::HyPerCol * hc) {
    initialize_base();
    initialize(name, hc);
}  // end StreamReconLayer::StreamReconLayer(const char *, PV::HyPerCol *)

StreamReconLayer::~StreamReconLayer() {
}

int StreamReconLayer::initialize_base() {
    
    bufferLevel = 0;
    return PV_SUCCESS;
}

int StreamReconLayer::updateState(double timef, double dt) {
    
    update_timer->start();
    
    pvdata_t * V = getV();
    
    int nx = getLayerLoc()->nx;
    int ny = getLayerLoc()->ny;
    int nf = getLayerLoc()->nf;
    
    for (int i = 0; i < nx; i++) {
        int vx = i;
        int gx = i;
        for (int j = 0; j < nf; j ++) {
            int vf = j;
            int gf;
            if (vf + bufferLevel < nf) {
                gf = bufferLevel + j;
            }
            
            else {
                gf = bufferLevel + j - nf;
            }
            
            int vindex = kIndex(vx, 0, vf, nx, ny, nf);
            int gindex = kIndex(gx, 0, gf, nx, ny, nf);
            
            V[vindex] = GSyn[0][gindex];
        }
    }
    
    //Copy V to A buffer
    PV::HyPerLayer::setActivity();
    
    if (bufferLevel < nf - 1) {
        bufferLevel++; }
    else {
        bufferLevel = 0;
    }
    
    update_timer->stop();
    return PV_SUCCESS;
} // end update state

PV::BaseObject * createStreamReconLayer(const char * name, PV::HyPerCol * hc) {
   return hc ? new StreamReconLayer(name, hc) : NULL;
}
