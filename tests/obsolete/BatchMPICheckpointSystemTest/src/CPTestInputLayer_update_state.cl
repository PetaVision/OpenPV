#include <layers/updateStateFunctions.h>
#include "CPTest_updateStateFunctions.h"

//
// update the state of a CPTestInputLayer
//
// To allow porting to GPUs, functions called from this function must be
// static inline functions.  If a subclass needs new behavior, it needs to
// have its own static inline function.
//
void CPTestInputLayer_update_state(
    const int nbatch,
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    float * V,
    const float Vth,
    const float AMin,
    const float AMax,
    float * GSynHead,
    float * activity)
{
   updateV_CPTestInputLayer(nbatch, numNeurons, V);
   setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   resetGSynBuffers_HyPerLayer(nbatch, numNeurons, 2, GSynHead);
}
