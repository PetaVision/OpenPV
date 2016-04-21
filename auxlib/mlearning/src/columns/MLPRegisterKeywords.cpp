#include "../connections/GradientCheckConn.hpp"
#include "../layers/MLPErrorLayer.hpp"
#include "../layers/MLPForwardLayer.hpp"
#include "../layers/MLPSigmoidLayer.hpp"
#include "../layers/MLPOutputLayer.hpp"
//#include "../layers/BatchNorm.hpp"

namespace PVMLearning {

int MLPRegisterKeywords(PV::PV_Init * pv_initObj) {
    int status = PV_SUCCESS;
    assert(PV_SUCCESS==0); // Using the |= operator assumes success is indicated by return value zero.
    status |= pv_initObj->registerKeyword("GradientCheckConn", createGradientCheckConn);
    status |= pv_initObj->registerKeyword("MLPErrorLayer", createMLPErrorLayer);
    status |= pv_initObj->registerKeyword("MLPForwardLayer", createMLPForwardLayer);
    status |= pv_initObj->registerKeyword("MLPSigmoidLayer", createMLPSigmoidLayer);
    status |= pv_initObj->registerKeyword("MLPOutputLayer", createMLPOutputLayer);
    //.status |= pv_initObj->registerKeyword("BatchNorm", createBatchNorm);
    return status;
}

} // end of namespace PVMLearning
