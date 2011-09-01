/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "HyperConnDebugInitWeights.hpp"
#include "KernelConnDebugInitWeights.hpp"

int addcustom(HyPerCol * hc, int argc, char * argv[]);
// addcustom is for adding objects not understood by build().

int main(int argc, char * argv[]) {
    return buildandrun(argc, argv, &addcustom)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int addcustom(HyPerCol * hc, int argc, char * argv[]) {

    int numOfConns=hc->numberOfConnections();

    for(int i=0; i<numOfConns;i++){
    	HyPerConn *tempConnPtr = hc->getConnection(i);
    	printf("connection %s\n", tempConnPtr->getName());
    	const char * weightInitTypeStr = hc->parameters()->stringValue(tempConnPtr->getName(), "weightInitType");
    	printf("weightInitType %s\n", weightInitTypeStr);

    	ParameterGroup *grp =hc->parameters()->group(tempConnPtr->getName());

    	printf("keyword %s\n", grp->getGroupKeyword());

		char newName[100];
		strncpy(newName, tempConnPtr->getName(), 94);
		strncat(newName, "_copy", 5);
    	printf("new conn name %s\n", newName);
    	HyPerLayer * preLayer = tempConnPtr->getPre();
		HyPerLayer * postLayer = tempConnPtr->getPost();
		// PVParams * params = hc->parameters();
		ChannelType channelType = CHANNEL_INH;
		//int channelNo = (int) params->value(tempConnPtr->getName(), "channelCode", -1);
		HyPerConn * newConn;
		if( !strcmp(grp->getGroupKeyword(), "HyPerConn") ) {
			newConn = new HyperConnDebugInitWeights(tempConnPtr->getName(), hc, preLayer, postLayer, channelType, tempConnPtr);
		}
		else if( !strcmp(grp->getGroupKeyword(), "KernelConn") ) {
			newConn = new KernelConnDebugInitWeights(tempConnPtr->getName(), hc, preLayer, postLayer, channelType, tempConnPtr);
		}

    }


    return PV_SUCCESS;
}
