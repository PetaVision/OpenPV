#include "ANNOutputComponent.hpp"

#include <utils/PVLog.hpp>

namespace PV
{
	BaseObject * createANNOutputComponent(char const * name, HyPerCol * hc)
	{
		return hc ? new ANNOutputComponent() : NULL;
	}
	
	ANNOutputComponent::~ANNOutputComponent()
	{
	   free(verticesV);
	   free(verticesA);
	   free(slopes);
	}

	void ANNOutputComponent::initialize()
	{
		setVertices();
		setSlopes();
	}

	void ANNOutputComponent::setVertices()
	{
	   if (VWidth<0) {
		  VThresh += VWidth;
		  VWidth = -VWidth;
		  if (parent->columnId()==0) {
			 fprintf(stderr, "%s \"%s\" warning: interpreting negative VWidth as setting VThresh=%f and VWidth=%f\n",
				   getKeyword(), name, VThresh, VWidth);
		  }
	   }

	   pvdata_t limfromright = VThresh+VWidth-AShift;
	   if (AMax < limfromright) limfromright = AMax;

	   if (AMin > limfromright) {
		  if (parent->columnId()==0) {
			 if (VWidth==0) {
				fprintf(stderr, "%s \"%s\" warning: nonmonotonic transfer function, jumping from %f to %f at Vthresh=%f\n",
					  getKeyword(), name, AMin, limfromright, VThresh);
			 }
			 else {
				fprintf(stderr, "%s \"%s\" warning: nonmonotonic transfer function, changing from %f to %f as V goes from VThresh=%f to VThresh+VWidth=%f\n",
					  getKeyword(), name, AMin, limfromright, VThresh, VThresh+VWidth);
			 }
		  }
	   }
	   
	   // Initialize slopes to NaN so that we can tell whether they've been initialized.
	   slopeNegInf = std::numeric_limits<double>::quiet_NaN();
	   slopePosInf = std::numeric_limits<double>::quiet_NaN();
	   std::vector<pvpotentialdata_t> vectorV;
	   std::vector<pvadata_t> vectorA;
	   
	   slopePosInf = 1.0f;
	   if (VThresh <= -0.999*max_pvadata_t) {
		  numVertices = 1;
		  vectorV.push_back((pvpotentialdata_t) 0);
		  vectorA.push_back(-AShift);
		  slopeNegInf = 1.0f;
	   }
	   else {
		  assert(VWidth >= (pvpotentialdata_t) 0);
		  if (VWidth == (pvpotentialdata_t) 0 && (pvadata_t) VThresh - AShift == AMin) {  // Should there be a tolerance instead of strict ==?
			 numVertices = 1;
			 vectorV.push_back(VThresh);
			 vectorA.push_back(AMin);
		  }
		  else {
			 numVertices = 2;
			 vectorV.push_back(VThresh);
			 vectorV.push_back(VThresh+VWidth);
			 vectorA.push_back(AMin);
			 vectorA.push_back(VThresh+VWidth-AShift);
		  }
		  slopeNegInf = 0.0f;
	   }
	   if (AMax < 0.999*max_pvadata_t) {
		  assert(slopePosInf == 1.0f);
		  if (vectorA[numVertices-1] < AMax) {
			 pvadata_t interval = AMax - vectorA[numVertices-1];
			 vectorV.push_back(vectorV[numVertices-1]+(pvpotentialdata_t) interval);
			 vectorA.push_back(AMax);
			 numVertices++;
		  }
		  else {
			 // find the last vertex where A < AMax.
			 bool found = false;
			 int v;
			 for (v=numVertices-1; v>=0; v--) {
				if (vectorA[v] < AMax) { found = true; break; }
			 }
			 if (found) {
				assert(v+1 < numVertices && vectorA[v] < AMax && vectorA[v+1] >= AMax);
				pvadata_t interval = AMax - vectorA[v];
				numVertices = v+1;
				vectorA.resize(numVertices);
				vectorV.resize(numVertices);
				vectorV.push_back(vectorV[v]+(pvpotentialdata_t) interval);
				vectorA.push_back(AMax);
				// In principle, there could be a case where a vertex n has A[n]>AMax but A[n-1] and A[n+1] are both < AMax.
				// But with the current ANNLayer parameters, that won't happen.
			 }
			 else {
				// All vertices have A>=AMax.
				// If slopeNegInf is positive, transfer function should increase from -infinity to AMax, and then stays constant.
				// If slopeNegInf is negative or zero, 
				numVertices = 1;
				vectorA.resize(numVertices);
				vectorV.resize(numVertices);
				if (slopeNegInf > 0) {
				   pvadata_t intervalA = vectorA[0]-AMax;
				   pvpotentialdata_t intervalV = (pvpotentialdata_t) (intervalA / slopeNegInf);
				   vectorV[0] = vectorV[0] - intervalV;
				   vectorA[0] = AMax;
				} 
				else {
				   // Everything everywhere is above AMax, so make the transfer function a constant A=AMax.
				   vectorA.resize(1);
				   vectorV.resize(1);
				   vectorV[0] = (pvpotentialdata_t) 0;
				   vectorA[0] = AMax;
				   numVertices = 1;
				   slopeNegInf = 0;
				}
			 }
			 
		  }
		  slopePosInf = 0.0f;
	   }
	   assert(!isnan(slopeNegInf) && !isnan(slopePosInf) && numVertices > 0);
	   assert(vectorA.size()==numVertices && vectorV.size()==numVertices);
	   verticesV = (pvpotentialdata_t *) malloc((size_t) numVertices * sizeof(*verticesV));
	   verticesA = (pvadata_t *) malloc((size_t) numVertices * sizeof(*verticesA));
	   if (verticesV==NULL || verticesA==NULL) {
		  fprintf(stderr, "%s \"%s\" error: unable to allocate memory for vertices:%s\n",
				getKeyword(), name, strerror(errno));
		  throw;
	   }
	   memcpy(verticesV, &vectorV[0], numVertices * sizeof(*verticesV));
	   memcpy(verticesA, &vectorA[0], numVertices * sizeof(*verticesA));
	}

	void ANNOutputComponent::setSlopes()
	{
	   slopes = (float *) malloc((size_t)(numVertices+1)*sizeof(*slopes));
	   slopes[0] = slopeNegInf;
	   slopes[numVertices] = slopePosInf;
	   for(int k = 1; k < numVertices; k++)
	   {
		  float V1 = verticesV[k-1];
		  float V2 = verticesV[k];
		  if (V1!=V2)
		  {
			 slopes[k] = (verticesA[k]-verticesA[k-1])/(V2-V1);
		  }
		  else
		  {
			 slopes[k] = verticesA[k]>verticesA[k-1] ? std::numeric_limits<float>::infinity() :
						 verticesA[k]<verticesA[k-1] ? -std::numeric_limits<float>::infinity() :
						 std::numeric_limits<float>::quiet_NaN();
		  }
	   }
	}
	
	void ANNOutputComponent::transform()
	{
		int last = numVertices-1;

		int numNeurons = mParentLayer->getNumNeurons();
		int numBatches = mParentLayer->getParent()->getNBatch();
		int batchOffset = mParentLayer->calcBatchOffset();
		
		#ifdef PV_USE_OPENMP_THREADS
			#pragma omp parallel for schedule(static)
		#endif
		for(int index = 0; index < numNeurons*numBatches; index++ )
		{
			int b = index / numNeurons;
			int k = index % numNeurons;
			MEM_GLOBAL pvdata_t * activity = mTransformBuffer->data + b * batchOffset;
			MEM_GLOBAL pvdata_t * membranePotential = mParentLayer->getV() + b * numNeurons;
			int extendedK = mParentLayer->calcActivityIndex(k);
			
			pvdata_t potential = membranePotential[k];
			pvdata_t value = 0.0f;
			if (potential < verticesV[0])
			{
				value = verticesA[0] + slopes[0]*(potential-verticesV[0]);
			}
			else if (potential >= verticesV[last])
			{
				value = verticesA[last] + slopes[numVertices]*(potential-verticesV[last]);
			}
			else
			{
				for (int v = 0; v < last; v++)
				{
					if (potential < verticesV[v]) { break; }
					if (potential == verticesV[v])
					{
						value = verticesA[v];
					}
					else if (potential > verticesV[v] && potential < verticesV[v+1])
					{
						value = verticesA[v] + slopes[v+1] * (potential - verticesV[v]);
					}
				}
			}

			activity[extendedK] = value;
		}
	}
	
	void ANNOutputComponent::derivative()
	{
		if(mDerivativeBuffer == nullptr) return;
		
		int last = numVertices-1;

		int numNeurons = mParentLayer->getNumNeurons();
		int numBatches = mParentLayer->getParent()->getNBatch();
		int batchOffset = mParentLayer->calcBatchOffset();
		
		#ifdef PV_USE_OPENMP_THREADS
			#pragma omp parallel for schedule(static)
		#endif
		for(int index = 0; index < numNeurons*numBatches; index++ )
		{
			int b = index / numNeurons;
			int k = index % numNeurons;
			MEM_GLOBAL pvdata_t * deriv = mDerivativeBuffer->data + b * batchOffset;
			MEM_GLOBAL pvdata_t * membranePotential = mParentLayer->getV() + b * numNeurons;
			int extendedK = mParentLayer->calcActivityIndex(k);
			
			pvdata_t potential = membranePotential[k];
			pvdata_t value = 0.0f;
			if (potential < verticesV[0])
			{
				value = slopes[0];
			}
			else if (potential >= verticesV[last])
			{
				value = slopes[numVertices];
			}
			else
			{
				for (int v = 0; v < last; v++)
				{
					if (potential < verticesV[v]) { break; }
					if (potential == verticesV[v])
					{
						value = 0.0f;
					}
					else if (potential > verticesV[v] && potential < verticesV[v+1])
					{
						value = slopes[v+1];
					}
				}
			}
			//Just the slope
			deriv[extendedK] = value;
		}
	}
	
	//*****************************
	// Params
	//*****************************
	
	void ANNOutputComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag)
	{
		ioParam_VThresh(ioFlag);
		ioParam_AMax(ioFlag);
		ioParam_AMin(ioFlag);
		ioParam_AShift(ioFlag);
		ioParam_VWidth(ioFlag);
	}

	void ANNOutputComponent::ioParam_VThresh(enum ParamsIOFlag ioFlag)
	{
		parent->ioParamValue(ioFlag, mParentLayer->getName(), "VThresh", &VThresh, -max_pvvdata_t);
	}

	void ANNOutputComponent::ioParam_AMin(enum ParamsIOFlag ioFlag)
	{
		parent->ioParamValue(ioFlag, mParentLayer->getName(), "AMin", &AMin, VThresh);
	}

	void ANNOutputComponent::ioParam_AMax(enum ParamsIOFlag ioFlag)
	{
		parent->ioParamValue(ioFlag, mParentLayer->getName(), "AMax", &AMax, max_pvvdata_t);
	}

	void ANNOutputComponent::ioParam_AShift(enum ParamsIOFlag ioFlag)
	{
		parent->ioParamValue(ioFlag, mParentLayer->getName(), "AShift", &AShift, (pvdata_t) 0);
	}

	void ANNOutputComponent::ioParam_VWidth(enum ParamsIOFlag ioFlag)
	{
		parent->ioParamValue(ioFlag, mParentLayer->getName(), "VWidth", &VWidth, (pvdata_t) 0);
	}
}
