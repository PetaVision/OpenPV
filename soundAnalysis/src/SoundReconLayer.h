//
//  SoundReconLayer.h
//  
//
//  Created by Brohit's Mac 5054128794 on 7/24/14.
//
//

#ifndef SOUNDRECONLAYER_H_
#define SOUNDRECONLAYER_H_

#include <iostream>
#include <layers/HyPerLayer.hpp>


namespace PV {
    
    class SoundReconLayer : public HyPerLayer{
    public:
        SoundReconLayer(const char * name, HyPerCol * hc);
        virtual int recvSynapticInput(HyPerConn * conn, const PVLayerCube * cube, int arborID);
        virtual int resetGSynBuffers(double timef, double dt);
        virtual ~SoundReconLayer();
        
    protected:
        SoundReconLayer();
        virtual int doUpdateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A,
                          pvdata_t * V, int num_channels, pvdata_t * GSynHead, bool spiking,
                          unsigned int * active_indices, unsigned int * num_active);
        
    private:
        int initialize_base();
        int bufferLevel;
        
    }; // end of class soundreconLayer
    
}  // end namespace PV


#endif /* defined(SOUNDRECONLAYER_H_) */