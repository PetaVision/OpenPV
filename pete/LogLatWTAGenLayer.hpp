/*
 * LogLatWTAGenLayer.hpp
 *
 * A subclass of GenerativeLayer that
 * uses log(lateral winner-take-all) dynamics for sparsity
 *
 *  Created on: Apr 20, 2011
 *      Author: peteschultz
 */

#ifndef LOGLATWTAGENLAYER_HPP_
#define LOGLATWTAGENLAYER_HPP_

#include "GenerativeLayer.hpp"

namespace PV {

class LogLatWTAGenLayer : public GenerativeLayer {
public:
    LogLatWTAGenLayer(const char * name, HyPerCol * hc);
    ~LogLatWTAGenLayer();
    int initialize_base();
    int initialize();

    virtual pvdata_t latWTAterm(pvdata_t * V, int nf);

protected:
    int updateV();

private:
    pvdata_t * dV;
};

}  // end namespace PV block

#endif /* LOGLATWTAGENLAYER_HPP_ */
