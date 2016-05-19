#include "ComponentLayer.hpp"

#include <include/default_params.h>
#include <include/pv_types.h>
#include <layers/InitV.hpp>
#include <connections/HyPerConn.hpp>
#include <utils/PVLog.hpp>

namespace PV
{
    BaseObject * createComponentLayer(char const * name, HyPerCol * hc)
	{
		return hc ? new ComponentLayer(name, hc) : NULL;
	}
    
    ComponentLayer::ComponentLayer(const char * name, HyPerCol * hc)
    {
        initialize_base();
        initialize(name, hc);
    }
    
    ComponentLayer::~ComponentLayer()
    {
        delete mInputComponent;
        delete mStateComponent;
        delete mOutputComponent;
    }

    //****************************
    // Containing layer functions
    //****************************
   
      //TODO: Make sure moving this stuff from allocateDataStructures to initialize didn't break anything
   
    int ComponentLayer::initialize(const char * name, HyPerCol * hc)
    {
       HyPerLayer::initialize(name, hc);
       
       if(mInputType != nullptr && strcmp(mInputType, "none") != 0)
        {
            BaseObject *object = getParent()->getPV_InitObj()->create(mInputType, nullptr, getParent());
            if(object == nullptr) { pvError() << getName() << ": Error, Could not create component " << mInputType << ", exiting." << std::endl; throw; }
            mInputComponent = dynamic_cast<InputComponent*>(object);
            if(object != nullptr && mInputComponent == nullptr) { pvError() << getName() << ": Error, " << mInputType << " is not an InputComponent, exiting." << std::endl; throw; }
            mInputComponent->setParentLayer(this);
            mInputComponent->setParentColumn(getParent());
            mInputComponent->ioParamsFillGroup(PARAMS_IO_READ);
            mInputComponent->initialize();
        }
        else { pvInfo() << getName() << ": Warning, no input component present." << std::endl; }
        
        if(mStateType != nullptr && strcmp(mStateType, "none") != 0)
        {
            BaseObject *object = getParent()->getPV_InitObj()->create(mStateType, nullptr, getParent());
            if(object == nullptr) { pvError() << getName() << ": Error, Could not create component " << mStateType << ", exiting." << std::endl; throw; }
            mStateComponent = dynamic_cast<StateComponent*>(object);
            if(object != nullptr && mStateComponent == nullptr) { pvError() << getName() << ": Error, " << mStateType << " is not a StateComponent, exiting."; throw; }
            mStateComponent->setParentLayer(this);
            mStateComponent->setParentColumn(getParent());
            mStateComponent->ioParamsFillGroup(PARAMS_IO_READ);
        }
        else { pvInfo() << getName() << ": Warning, no state component present." << std::endl; }
        
        if(mOutputType != nullptr && strcmp(mOutputType, "none") != 0)
        {
            BaseObject *object = getParent()->getPV_InitObj()->create(mOutputType, nullptr, getParent());
            if(object == nullptr) { pvError() << getName() << ": Error, Could not create component " << mOutputType << ", exiting." << std::endl; throw; }
            mOutputComponent = dynamic_cast<OutputComponent*>(object);
            if(object != nullptr && mOutputComponent == nullptr) { pvError() << getName() << ": Error, " << mOutputType << " is not an OutputComponent, exiting."; throw; }
            
            mOutputComponent->setParentLayer(this);
            mOutputComponent->setParentColumn(getParent());
            mOutputComponent->ioParamsFillGroup(PARAMS_IO_READ);
        }
        else { pvError() << getName() << ": No output component present." << std::endl; throw; } //Output is the only required component
       
       return PV_SUCCESS;
    }
    
    int ComponentLayer::initialize_base()
    {
        mCallDerivative = false;
        return PV_SUCCESS;
    }
    
    int ComponentLayer::allocateDataStructures()
    {
        return HyPerLayer::allocateDataStructures();
    }

    int ComponentLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag)
    {
      ioParam_callDerivative(ioFlag);

      ioParam_inputComponent(ioFlag);
      ioParam_stateComponent(ioFlag);
      ioParam_outputComponent(ioFlag);

      //If we're reading, components haven't been instantiated yet.
      //We'll call their ioParamsFillGroup in allocateDataStructures.
      //Otherwise, we're writing and they already exist, so let's let
      //them write as needed
      if(mInputComponent != nullptr) mInputComponent->ioParamsFillGroup(ioFlag);
      if(mStateComponent != nullptr) mStateComponent->ioParamsFillGroup(ioFlag);
      if(mOutputComponent != nullptr) mOutputComponent->ioParamsFillGroup(ioFlag);


      return HyPerLayer::ioParamsFillGroup(ioFlag);
    }

    int ComponentLayer::callUpdateState(double timed, double dt)
    {
        update_timer->start();
         
        if(mInputComponent != nullptr) mInputComponent->transform();
        if(mStateComponent != nullptr) mStateComponent->transform();
        if(mOutputComponent != nullptr) mOutputComponent->transform();
        
        if(mCallDerivative)
        {
            if(mInputComponent != nullptr) mInputComponent->derivative();
            if(mStateComponent != nullptr) mStateComponent->derivative();
            if(mOutputComponent != nullptr) mOutputComponent->derivative();
        }

        update_timer->stop();

        return PV_SUCCESS;
    }

   int ComponentLayer::setInitialValues()
   {
      initializeV();
      initializeActivity();
      return PV_SUCCESS;
   }

   //********************
   // Utility functions
   //********************
   
   int ComponentLayer::calcBatchOffset()
   {
      int nx = getLayerLoc()->nx;
      int ny = getLayerLoc()->ny;
      int nf = getLayerLoc()->nf;
      int lt = getLayerLoc()->halo.lt;
      int rt = getLayerLoc()->halo.rt;
      int up = getLayerLoc()->halo.up;
      int dn = getLayerLoc()->halo.dn;
      return ((nx+lt+rt)*(ny+up+dn)*nf);
   }

   int ComponentLayer::calcActivityIndex(int k)
   {
      int nx = getLayerLoc()->nx;
      int ny = getLayerLoc()->ny;
      int nf = getLayerLoc()->nf;
      int lt = getLayerLoc()->halo.lt;
      int rt = getLayerLoc()->halo.rt;
      int up = getLayerLoc()->halo.up;
      int dn = getLayerLoc()->halo.dn;
      return kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
   }

    //**************************
    // Input related functions
    //**************************
    
    int ComponentLayer::resetGSynBuffers(double timef, double dt)
    {
       if(mInputComponent != nullptr) mInputComponent->resetBuffer(timef, dt);
    }
    
    int ComponentLayer::recvAllSynapticInput()
    {
        if(mInputComponent == nullptr) return PV_SUCCESS;
        
        if(needUpdate(parent->simulationTime(), parent->getDeltaTime()))
        {
            //TODO: GPU support
            recvsyn_timer->start();
            mInputComponent->receive(&recvConns);
            recvsyn_timer->stop();
        }
        return PV_SUCCESS;
    }
    
    int ComponentLayer::allocateGSyn()
    {
        if(mInputComponent == nullptr) return PV_SUCCESS;

        mInputComponent->allocateTransformBuffer();
        if(mCallDerivative) mInputComponent->allocateDerivativeBuffer();
        
        GSyn = mInputComponent->getTransformBuffer();
        
        return PV_SUCCESS;
    }
    
    //**************************
    // State related functions
    //**************************
    
    int ComponentLayer::allocateV()
    {
        if(mStateComponent == nullptr) return PV_SUCCESS;
        
        mStateComponent->allocateTransformBuffer();
        if(mCallDerivative)  mStateComponent->allocateDerivativeBuffer();
        
        clayer->V = mStateComponent->getTransformBuffer();
    }
       int ComponentLayer::initializeV()
      {
         if(mStateComponent == nullptr) return PV_SUCCESS;
         if(initVObject != NULL) initVObject->calcV(this);
         mStateComponent->initialize();
         return PV_SUCCESS;
      }

      int ComponentLayer::readVFromCheckpoint(const char * cpDir, double * timeptr)
      {
         if(mStateComponent == nullptr) return PV_SUCCESS;

         HyPerLayer::readVFromCheckpoint(cpDir, timeptr);
         mStateComponent->initialize();
         return PV_SUCCESS;
      }
    //**************************
    // Output related functions
    //**************************
    
    int ComponentLayer::setActivity()
    {
        mOutputComponent->transform();
        if(mCallDerivative) mOutputComponent->derivative();
        return PV_SUCCESS;
    }
    
    int ComponentLayer::allocateActivity()
    {
        mOutputComponent->allocateTransformBuffer();
        if(mCallDerivative) mOutputComponent->allocateDerivativeBuffer();
        
        clayer->activity = mOutputComponent->getTransformBuffer();
        return clayer->activity!=NULL ? PV_SUCCESS : PV_FAILURE;
    }
    
    int ComponentLayer::initializeActivity()
    {
      mOutputComponent->initialize();
      mOutputComponent->clearActivity();
      return PV_SUCCESS;
    }
   
    int ComponentLayer::readActivityFromCheckpoint(const char * cpDir, double * timeptr)
    {
       HyPerLayer::readActivityFromCheckpoint(cpDir, timeptr);
       mOutputComponent->initialize();
       return PV_SUCCESS;
    }

    double ComponentLayer::getDeltaUpdateTime()
    {
       double dt = mOutputComponent->getDeltaUpdateTime();
       if(dt != -1.0) return dt;
       return HyPerLayer::getDeltaUpdateTime();
    }
			
    //************************
    // Params
    //************************
    
    void ComponentLayer::ioParam_inputComponent(enum ParamsIOFlag ioFlag)
    {
       parent->ioParamString(ioFlag, name, "inputComponent", &mInputType, NULL, false);
    }
    
    void ComponentLayer::ioParam_stateComponent(enum ParamsIOFlag ioFlag)
    {
       parent->ioParamString(ioFlag, name, "stateComponent", &mStateType, NULL, false);
    }
    
    void ComponentLayer::ioParam_outputComponent(enum ParamsIOFlag ioFlag)
    {
       parent->ioParamString(ioFlag, name, "outputComponent", &mOutputType, NULL, false);
    }
    
    void ComponentLayer::ioParam_callDerivative(enum ParamsIOFlag ioFlag)
    {
        parent->ioParamValue(ioFlag, name, "callDerivative", &mCallDerivative, false);
    }
}
