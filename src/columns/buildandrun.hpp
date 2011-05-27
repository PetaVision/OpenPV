/*
 * buildandrun.hpp
 *
 *  Created on: May 27, 2011
 *      Author: peteschultz
 */

#ifndef BUILDANDRUN_HPP_
#define BUILDANDRUN_HPP_

#include <time.h>
#include <string>
#include <iostream>

#include "../include/pv_common.h"

#include "../columns/HyPerCol.hpp"

#include "../layers/HyPerLayer.hpp"
#include "../layers/PVLayer.h"
#include "../layers/ANNLayer.hpp"
#include "../layers/GenerativeLayer.hpp"
#include "../layers/LogLatWTAGenLayer.hpp"
#include "../layers/GeislerLayer.hpp"
#include "../layers/PoolingANNLayer.hpp"
#include "../layers/PtwiseProductLayer.hpp"
#include "../layers/TrainingLayer.hpp"
#include "../layers/HMaxSimple.hpp"
#include "../layers/Image.hpp"
#include "../layers/CreateMovies.hpp"
#include "../layers/ImageCreator.hpp"
#include "../layers/Movie.hpp"
#include "../layers/Patterns.hpp"
#include "../layers/LGN.hpp"
#include "../layers/LIF.hpp"
#include "../layers/Retina.hpp"

#include "../connections/HyPerConn.hpp"
#include "../connections/AvgConn.hpp"
#include "../connections/ConvolveConn.hpp"
#include "../connections/HyPerConn.hpp"
#include "../connections/KernelConn.hpp"
#include "../connections/CocircConn.hpp"
#include "../connections/GaborConn.hpp"
#include "../connections/GeislerConn.hpp"
#include "../connections/GenerativeConn.hpp"
#include "../connections/FeedbackConn.hpp"
#include "../connections/PoolingGenConn.hpp"
#include "../connections/IdentConn.hpp"
#include "../connections/CloneKernelConn.hpp"
#include "../connections/TransposeConn.hpp"
#include "../connections/PoolConn.hpp"
#include "../connections/RuleConn.hpp"
#include "../connections/SubunitConn.hpp"

#include "../io/ConnectionProbe.hpp"
#include "../io/LayerProbe.hpp"
#include "../io/PointProbe.hpp"
#include "../io/L2NormProbe.hpp"
#include "../io/SparsityTermProbe.hpp"
#include "../io/GenColProbe.hpp"
#include "../io/LogLatWTAProbe.hpp"

using namespace PV;

int buildandrun(int argc, char * argv[]);
HyPerCol * build(int argc, char * argv[]);

HyPerCol * addHyPerColToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
HyPerLayer * addLayerToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
TrainingLayer * addTrainingLayer(const char * name, HyPerCol *hc);
Image * addImage(const char * name, HyPerCol *hc);
Movie * addMovie(const char * name, HyPerCol *hc);
Patterns * addPatterns(const char * name, HyPerCol *hc);
HyPerConn * addConnToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
PoolingGenConn * addPoolingGenConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel);
ColProbe * addColProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
void insertColProbe(ColProbe * colProbe, HyPerCol * hc, const char * classkeyword);
ConnectionProbe * addConnectionProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
const char * getStringValueFromParameterGroup(const char * groupName, PVParams * params, const char * parameterString, bool warnIfNotPresent);
void getPreAndPostLayers(const char * name, HyPerCol * hc, HyPerLayer ** preLayerPtr, HyPerLayer **postLayer);
HyPerLayer * getLayerFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName);
HyPerConn * getConnFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName);
LayerProbe * addLayerProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
int getLayerFunctionProbeParameters(const char * name, const char * keyword, HyPerCol * hc, HyPerLayer ** targetLayer, const char ** message, const char ** filename);
int decodeChannel(int channel, ChannelType * channelType);
int checknewobject(void * object, const char * kw, const char * name);

#endif /* BUILDANDRUN_HPP_ */
