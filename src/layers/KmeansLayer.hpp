/*
 * KmeansLayer.hpp
 *
 *  Created on: Dec. 1, 2014
 *      Author: Xinhua Zhang
 */

#ifndef KMEANSLAYER_HPP_
#define KMEANSLAYER_HPP_

#include "HyPerLayer.hpp"


namespace PV
{
    class KmeansLayer : public HyPerLayer 
    {
    public:
        KmeansLayer(const char* name, HyPerCol * hc);
        virtual bool activityIsSpiking() { return false; }
        virtual ~KmeansLayer();

    protected:
        KmeansLayer();
        int initialize(const char * name, HyPerCol * hc);
        virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,pvdata_t * V, int num_channels, pvdata_t * gSynHead);
        virtual int setActivity();
        virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
        virtual void ioParam_TrainingFlag(enum ParamsIOFlag ioFlag);
        bool trainingFlag;
    private:
        int initialize_base();

    }; // class KmeansLayer

BaseObject * createKmeansLayer(char const * name, HyPerCol * hc);

}  // namespace PV

#endif /* KMEANSLAYER_HPP_ */
