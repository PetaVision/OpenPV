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


namespace PV {
    
    class StreamReconLayer : public HyPerLayer{
    public:
        StreamReconLayer(const char * name, HyPerCol * hc);
        virtual ~StreamReconLayer();
        virtual bool activityIsSpiking() { return false; }
        
    protected:
        StreamReconLayer();
        virtual int updateState (double time, double dt);
        
    private:
        int initialize_base();
        int bufferLevel;
        
    }; // end of class streamreconlayer
    
}  // end namespace PV


#endif /* defined(STREAMRECONLAYER_H_) */
