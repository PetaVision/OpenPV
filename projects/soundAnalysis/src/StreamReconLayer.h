//
//  StreamReconLayer.h
//  
//
//  Created by Brohit's Mac 5054128794 on 7/29/14.
//
//

#ifndef STREAMRECONLAYER_H_
#define STREAMRECONLAYER_H_

#include <iostream>
#include <layers/HyPerLayer.hpp>

class StreamReconLayer : public PV::HyPerLayer{
public:
    StreamReconLayer(const char * name, PV::HyPerCol * hc);
    virtual ~StreamReconLayer();
    virtual bool activityIsSpiking() { return false; }
    
protected:
    StreamReconLayer();
    virtual int updateState (double time, double dt);
    
private:
    int initialize_base();
    int bufferLevel;
    
}; // end of class StreamReconLayer
    

PV::BaseObject * createStreamReconLayer(char const * name, PV::HyPerCol * hc);

#endif /* defined(STREAMRECONLAYER_H_) */
