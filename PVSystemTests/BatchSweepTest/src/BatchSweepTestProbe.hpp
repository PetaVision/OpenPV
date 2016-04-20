/*
 * BatchSweepTestProbe.hpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#ifndef BATCHSWEEPTESTPROBE_HPP_
#define BATCHSWEEPTESTPROBE_HPP_

#include <io/StatsProbe.hpp>
#include <layers/HyPerLayer.hpp>
#include <assert.h>
#include <math.h>

namespace PV {

class BatchSweepTestProbe : public StatsProbe {
public:
	BatchSweepTestProbe(const char * probeName, HyPerCol * hc);
	virtual ~BatchSweepTestProbe();

	virtual int outputState(double timed);
protected:
    int initBatchSweepTestProbe(const char * probeName, HyPerCol * hc);
    virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
    virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
    virtual void ioParam_expectedSum(enum ParamsIOFlag ioFlag);
    virtual void ioParam_expectedMin(enum ParamsIOFlag ioFlag);
    virtual void ioParam_expectedMax(enum ParamsIOFlag ioFlag);

private:
    double expectedSum;
    float expectedMin, expectedMax;
};

BaseObject * createBatchSweepTestProbe(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* PARAMETERSWEEPTESTPROBE_HPP_ */
