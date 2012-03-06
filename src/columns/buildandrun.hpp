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
#include "../layers/IncrementLayer.hpp"
#include "../layers/LogLatWTAGenLayer.hpp"
#ifdef OBSOLETE // Marked obsolete Feb 27, 2012.  Replaced by CliqueLayer.
#include "../layers/ODDLayer.hpp"
#endif // OBSOLETE
#include "../layers/CliqueLayer.hpp"
#include "../layers/PoolingANNLayer.hpp"
#include "../layers/PtwiseProductLayer.hpp"
#include "../layers/TrainingLayer.hpp"
#include "../layers/GapLayer.hpp"
#include "../layers/HMaxSimple.hpp"
#include "../layers/Image.hpp"
#include "../layers/CreateMovies.hpp"
#include "../layers/ImageCreator.hpp"
#include "../layers/Movie.hpp"
#include "../layers/Patterns.hpp"
#include "../layers/LIF.hpp"
#include "../layers/LIFGap.hpp"
#include "../layers/Retina.hpp"
#include "../layers/SigmoidLayer.hpp"
#include "../layers/ANNSquaredLayer.hpp"
#include "../layers/ANNDivInh.hpp"

#include "../connections/HyPerConn.hpp"
#ifdef OBSOLETE // Marked Obsolete Oct 22, 2011.  No one seems to be using AvgConn, so the refactoring of arbors will leave it behind.
#include "../connections/AvgConn.hpp"
#endif // OBSOLETE
#ifdef OBSOLETE // Marked Obsolete Feb 22, 2012.  ConvolveConn is incomplete and no one seems to be planning on using it.
#include "../connections/ConvolveConn.hpp"
#endif // OBSOLETE
#include "../connections/HyPerConn.hpp"
#include "../connections/KernelConn.hpp"
#include "../connections/NoSelfKernelConn.hpp"
#include "../connections/GapConn.hpp"
#ifdef OBSOLETE // Marked obsolete Sept 22, 2011.  These classes are replaced by using InitWeights subclasses and setting params in a KernelConn
#include "../connections/CocircConn.hpp"
#include "../connections/GaborConn.hpp"
#endif // OBSOLETE
#ifdef OBSOLETE // Marked obsolete Feb 27, 2012.  Replaced by CliqueConn.
#include "../connections/ODDConn.hpp"
#endif // OBSOLETE
#include "../connections/CliqueConn.hpp"
#ifdef OBSOLETE // Marked obsolete Feb 27, 2012.  Replaced by SiblingConn.
#include "../connections/CliqueApplyConn.hpp"
#endif // OBSOLETE
#include "../connections/SiblingConn.hpp"
#include "../connections/GapConn.hpp"
#include "../connections/GenerativeConn.hpp"
#include "../connections/FeedbackConn.hpp"
#include "../connections/PoolingGenConn.hpp"
#include "../connections/IdentConn.hpp"
#include "../connections/CloneKernelConn.hpp"
#include "../connections/TransposeConn.hpp"
#ifdef OBSOLETE // Marked obsolete Sept 22, 2011.  These classes were replaced by using InitWeights subclasses and setting params in a KernelConn
#include "../connections/PoolConn.hpp"
#include "../connections/RuleConn.hpp"
#endif // OBSOLETE
#include "../connections/STDPConn.hpp"
#ifdef OBSOLETE // Marked obsolete Sept 22, 2011.  These classes were replaced by using the InitWeights subclasses and setting params in a KernelConn
#include "../connections/SubunitConn.hpp"
#endif // OBSOLETE

#include "../weightinit/InitWeights.hpp"
#include "../weightinit/InitCocircWeights.hpp"
#include "../weightinit/InitSmartWeights.hpp"
#include "../weightinit/InitUniformRandomWeights.hpp"
#include "../weightinit/InitGaussianRandomWeights.hpp"
#include "../weightinit/InitGaborWeights.hpp"
#include "../weightinit/InitPoolWeights.hpp"
#include "../weightinit/InitRuleWeights.hpp"
#include "../weightinit/InitSubUnitWeights.hpp"
#include "../weightinit/InitOneToOneWeights.hpp"
#include "../weightinit/InitIdentWeights.hpp"
#include "../weightinit/InitUniformWeights.hpp"
#include "../weightinit/InitSpreadOverArborsWeights.hpp"
#include "../weightinit/Init3DGaussWeights.hpp"
#include "../weightinit/InitWindowed3DGaussWeights.hpp"
#include "../weightinit/InitMTWeights.hpp"
#include "../io/BaseConnectionProbe.hpp"
#include "../io/KernelProbe.hpp"
#include "../io/PatchProbe.hpp"
#include "../io/ReciprocalEnergyProbe.hpp"
#ifdef OBSOLETE // Marked obsolete Feb. 27, 2012.  Replaced by PatchProbe.
#include "../io/ConnectionProbe.hpp"
#endif // OBSOLETE
#include "../io/LayerProbe.hpp"
#include "../io/PointProbe.hpp"
#include "../io/PointLIFProbe.hpp"
#include "../io/StatsProbe.hpp"
#include "../io/L2NormProbe.hpp"
#include "../io/SparsityTermProbe.hpp"
#include "../io/GenColProbe.hpp"
#include "../io/LogLatWTAProbe.hpp"

using namespace PV;

int buildandrun(int argc, char * argv[],
                int (*customadd)(HyPerCol *, int, char **) = NULL,
                int (*customexit)(HyPerCol *, int, char **) = NULL,
                void * (*customgroups)(const char *, const char *, HyPerCol *) = NULL);
HyPerCol * build(int argc, char * argv[], void * (*customgroups)(const char *, const char *, HyPerCol *) = NULL);

HyPerCol * addHyPerColToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
HyPerLayer * addLayerToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
TrainingLayer * addTrainingLayer(const char * name, HyPerCol *hc);
GapLayer * addGapLayer(const char * name, HyPerCol * hc);
Image * addImage(const char * name, HyPerCol *hc);
Movie * addMovie(const char * name, HyPerCol *hc);
Patterns * addPatterns(const char * name, HyPerCol *hc);
SigmoidLayer * addSigmoidLayer(const char * name, HyPerCol * hc);
InitWeights *createInitWeightsObject(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      ChannelType channel);
InitWeights * getDefaultInitWeightsMethod(const char * keyword);
HyPerConn * addConnToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
PoolingGenConn * addPoolingGenConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename, InitWeights *weightInit);
ColProbe * addColProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
void insertColProbe(ColProbe * colProbe, HyPerCol * hc, const char * classkeyword);
BaseConnectionProbe * addBaseConnectionProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
#ifdef OBSOLETE // Marked obsolete Feb. 27, 2012.  Replaced by PatchProbe.
ConnectionProbe * addConnectionProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
#endif // OBSOLETE
const char * getStringValueFromParameterGroup(const char * groupName, PVParams * params, const char * parameterString, bool warnIfAbsent);
int getPreAndPostLayers(const char * name, HyPerCol * hc, HyPerLayer ** preLayerPtr, HyPerLayer **postLayer);
HyPerLayer * getLayerFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName, bool warnIfAbsent=true);
HyPerConn * getConnFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName, bool warnIfAbesnt=true);
LayerProbe * addLayerProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
int getLayerFunctionProbeParameters(const char * name, const char * keyword, HyPerCol * hc, HyPerLayer ** targetLayer, char ** message, const char ** filename);
int decodeChannel(int channel, ChannelType * channelType);
int checknewobject(void * object, const char * kw, const char * name, HyPerCol * hc=NULL); /* Defaulting to NULL is temporary */

#endif /* BUILDANDRUN_HPP_ */
