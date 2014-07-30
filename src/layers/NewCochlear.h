//
//  NewCochlear.h
//  
//
//  Created by Brohit's Mac 5054128794 on 7/29/14.
//
//

#ifndef NEWCOCHLEARLAYER_H_
#define NEWCOCHLEARLAYER_H_

//Only compile this file and its cpp if using sound sandbox
#ifdef PV_USE_SNDFILE

#include "HyPerLayer.hpp"
#include "../columns/HyPerCol.hpp"
#include <vector>
#include <sndfile.h>

#ifndef STAT_H
#include <sys/stat.h>
#endif

namespace PV {
    
    class NewCochlearLayer : public HyPerLayer{
    public:
        NewCochlearLayer(const char* name, HyPerCol * hc);
        virtual ~NewCochlearLayer();
        virtual int updateState (double time, double dt);
        
        virtual int communicateInitInfo();
        virtual int allocateDataStructures();
        
        const std::vector <float> getTargetFreqs() {return targetFreqs;}
        const std::vector <float> getDampingConstants() {return dampingConstants;}
        float getSampleRate() { return sampleRate; }
        float getcochlearScale() { return cochlearScale; }
        

        
    protected:
        NewCochlearLayer();
        
        int initialize(const char * name, HyPerCol * hc);
        
        virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
        virtual void ioParam_FreqMinMax(enum ParamsIOFlag ioFlag);
        virtual void ioParam_nf(enum ParamsIOFlag ioFlag);
        virtual void ioParam_targetChannel(enum ParamsIOFlag ioFlag);
        //virtual void ioParam_inputLayername(enum ParamsIOFlag ioFlag);
        virtual void ioParam_dampingConstant(enum ParamsIOFlag ioFlag);
        virtual void ioParam_sampleRate(enum ParamsIOFlag ioFlag);
        virtual void ioParam_cochlearScale(enum ParamsIOFlag ioFlag);
  
        
        virtual void ioParam_soundInputPath(enum ParamsIOFlag ioFlag);
        virtual void ioParam_frameStart(enum ParamsIOFlag ioFlag);
        //virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);
        virtual double getDeltaUpdateTime();
        //virtual int setActivity();
        
        pvdata_t * soundData; //Buffer containing image
        SF_INFO* fileHeader;
        SNDFILE* fileStream;
        float* soundBuf;
        
        // double displayPeriod;     // Length of time a string 'frame' is displayed
        int frameStart;
        
        int sampleRate;          // sample rate from file in Hz
        // double nextSampleTime;   // time at which next sample is retrieved
        char * filename;          // Path to file if a file exists
        
    private:
        int initialize_base();
        float freqMin;
        float freqMax;
        std::vector <float> targetFreqs;
        std::vector <float> radianFreqs;
        std::vector <float> omegas;
        std::vector <float> dampingConstants;
        HyPerLayer* inputLayer;
        char* inputLayername;
        int targetChannel;
        float dampingConstant;
        float omega;
        //float sampleRate;
        float cochlearScale;
        float* vVal; //velocity value
        float* xVal; //x value
        float samplePeriod;
        
        
    }; // end of class NewCochlearLayer
    
}  // end namespace PV

#endif /* PV_USE_SNDFILE */

#endif /* NEWCOCHLEARLAYER_HPP_ */
