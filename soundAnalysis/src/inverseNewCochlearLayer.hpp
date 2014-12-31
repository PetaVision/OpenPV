/*
 * inverseNewCochlearLayer.hpp
 *
 *  Created on: Aug 5, 2014
 *      Author: mohitdubey
 */

#ifndef INVERSENEWCOCHLEARLAYER_HPP_
#define INVERSENEWCOCHLEARLAYER_HPP_

#include <layers/ANNLayer.hpp>
#include <layers/NewCochlear.h>

namespace PV {
    
    class inverseNewCochlearLayer : public ANNLayer{
    public:
        inverseNewCochlearLayer(const char* name, HyPerCol * hc);
        virtual ~inverseNewCochlearLayer();
        virtual int updateState (double time, double dt);
        
        virtual int communicateInitInfo();
        virtual int allocateDataStructures();
    protected:
        inverseNewCochlearLayer();
        
        int initialize(const char * name, HyPerCol * hc);
        
        virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
        virtual void ioParam_inputLayername(enum ParamsIOFlag ioFlag);
        virtual void ioParam_cochlearLayername(enum ParamsIOFlag ioFlag);
        virtual void ioParam_sampleRate(enum ParamsIOFlag ioFlag);
        virtual void ioParam_bufferLength(enum ParamsIOFlag ioFlag);
        
    private:
        int initialize_base();
        int ringBuffer(int level);
        
        
        float sampleRate;
        char* inputLayername;
        char* cochlearLayername;
        //The layer to grab the input from
        HyPerLayer* inputLayer;
        //The cochlear layer to grab nessessary parameters from
        NewCochlearLayer* cochlearLayer;
        
        int bufferLength;
        pvdata_t ** xhistory; // ring buffer of past x_k(t_j).
        int ringBufferLevel;
        double * timehistory; // may not need
        
        int numFrequencies;
        float * targetFreqs;
        float * deltaFreqs;
        float ** Mreal; // f = sum_j M[j][k] * x_k(t_j).  Should choose a more descriptive name
        float ** Mimag;
        double nextDisplayTime;
        
        
        float* xVal;
        float* lastxVal;
        float* vVal;
        float* lastvVal;
        float* pastxVal;
        float* sound;
        float* energy;
        
        float* omegas;
        float* radianFreqs;
        float* dampingConstants;
        
        float outputsound;
        float sumenergy;
        
    }; // end of class inverseNewCochlearLayer
    
}  // end namespace PV

#endif /* ANNLAYER_HPP_ */
