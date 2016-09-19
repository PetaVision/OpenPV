#include "HyPerConnGPU.hpp"

namespace GPULCA {

HyPerConnGPU::HyPerConnGPU() {
  initialize_base();
  initialize();
}

HyPerConnGPU::HyPerConnGPU(const char* name, PV::HyPerCol* hc)
    : HyPerConn(name, hc) {
  initialize_base();
  initialize();
}

HyPerConnGPU::~HyPerConnGPU() {
  /*  cuDnn destroy */
  cudnnStatusDestructorCheck(cudnnDestroy(cudnnHandle), "destroy handle");

  cudnnStatusDestructorCheck(
      cudnnDestroyTensorDescriptor(cudnnTensorDescriptorPre),
      "destroy tensor descriptor");
  cudnnStatusDestructorCheck(
      cudnnDestroyTensorDescriptor(cudnnTensorDescriptorPost),
      "destroy tensor descriptor");
  cudnnStatusDestructorCheck(
      cudnnDestroyFilterDescriptor(cudnnFilterDescriptor),
      "destroy filter descriptor");
  cudnnStatusDestructorCheck(
      cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor),
      "destroy convolution descriptor");

  cudnnStatusDestructorCheck(
      cudnnDestroyTensorDescriptor(cudnnTensorDescriptorPreNHWC),
      "destroy tensor descriptor");

  if (!isPreGPULayer) {
    delete PreNHWC;
    delete pre;
  }

  if (isWeightSparse) {
    /*  cuSparse destroy */
    cusparseStatusDestructorCheck(cusparseDestroy(cusparseHandle),
                                  "destory handle");
    cusparseStatusDestructorCheck(cusparseDestroyMatDescr(cusparseMatDescr),
                                  "destory matrix descriptor");
  }
}

int HyPerConnGPU::allocateDataStructures() {
  int status = HyPerConn::allocateDataStructures();

  const PVLayerLoc* preLoc = preSynapticLayer()->getLayerLoc(),
                    * postLoc = postSynapticLayer()->getLayerLoc();
  MatrixInfo weightParams = {.n = postLoc->nf,
                             .height = yPatchSize(),
                             .width = xPatchSize(),
                             .channel = preLoc->nf,
                             .layout = NCHW},
             transposedWeightParams = {.n = preLoc->nf,
                                       .height = yPatchSize(),
                                       .width = xPatchSize(),
                                       .channel = postLoc->nf,
                                       .layout = NCHW},
             sparseWeightParams = {.n = postLoc->nf,
                                   .height = yPatchSize(),
                                   .width = xPatchSize(),
                                   .channel = preLoc->nf,
                                   .layout = NHWC},
             transposedSparseWeightParams = {.n = preLoc->nf,
                                             .height = yPatchSize(),
                                             .width = xPatchSize(),
                                             .channel = postLoc->nf,
                                             .layout = NHWC},
             preNHWCParams = {.n = 1,
                              .height = preLoc->ny,
                              .width = preLoc->nx,
                              .channel = preLoc->nf,
                              .layout = NHWC},
             preParams = {.n = 1,
                          .height = preLoc->ny,
                          .width = preLoc->nx,
                          .channel = preLoc->nf,
                          .layout = NCHW};

  if (isPreGPULayer) {
    PreNHWC = nullptr;
    Pre = &((dynamic_cast<ANNLayerGPU*>(preSynapticLayer()))->getActivity());

  } else {
    PreNHWC = new PVCudaWrapper<pvwdata_t>(preNHWCParams);
    Pre = new PVCudaWrapper<pvwdata_t>(preParams);
  }

  W.resize(numAxonalArborLists);
  WT.resize(numAxonalArborLists);

  if (isWeightSparse) {
    auto initFunc = [&, this](PVCudaWrapper<pvwdata_t>& w) {
      w.sparse.init(sparseWeightParams, &cusparseHandle, &cusparseMatDescr,
                    true);
    };
    std::for_each(W.begin(), W.end(), initFunc);

    auto transposedInitFunc = [&, this](PVCudaWrapper<pvwdata_t>& w) {
      w.sparse.init(transposedSparseWeightParams, &cusparseHandle,
                    &cusparseMatDescr, true);
    };
    std::for_each(WT.begin(), WT.end(), transposedInitFunc);

  } else {
    auto initFunc = [&, this](PVCudaWrapper<pvwdata_t>& w) {
      w.dense.init(weightParams);
      w.dense.getCudaVector().setDeviceData(w.dense.getSize(),
                                            get_wDataStart(0));
    };

    std::for_each(W.begin(), W.end(), initFunc);

    auto transposedInitFunc = [&, this](PVCudaWrapper<pvwdata_t>& w) {
      w.dense.init(transposedWeightParams);
      w.dense.getCudaVector().setDeviceData(w.dense.getSize(),
                                            get_wDataStart(0));
    };

    std::for_each(WT.begin(), WT.end(), initFunc);

    /*  cuDnn initialization */
    if (!isPreGPULayer) {
      cudnnStatusCheck(
          cudnnSetTensor4dDescriptor(cudnnTensorDescriptorPreNHWC,
                                     CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1,
                                     preLoc->nf, preLoc->ny, preLoc->nx),
          "set 4D tensor");
    }

    cudnnStatusCheck(
        cudnnSetTensor4dDescriptor(cudnnTensorDescriptorPre, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT, 1, preLoc->nf, preLoc->ny,
                                   preLoc->nx),
        "set 4D tensor");

    cudnnStatusCheck(
        cudnnSetTensor4dDescriptor(cudnnTensorDescriptorPost, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT, 1, postLoc->nf,
                                   postLoc->ny, postLoc->nx),
        "set 4D tensor");

    // 2D Convolution
    PV::HyPerLayer* pre = preSynapticLayer(), * post = postSynapticLayer();
    int xStride = (post->getCLayer()->xScale - pre->getCLayer()->xScale) * 2,
        yStride = (post->getCLayer()->yScale - pre->getCLayer()->yScale) * 2;
    int pad_h, pad_w, out_h, out_w;
    if (preLoc->nx % xStride != 0 || preLoc->ny % yStride != 0)
      throw logic_error("The size of image is: (" + to_string(preLoc->nx) +
                        "," + to_string(preLoc->ny) + ")\nThe stride is: (" +
                        to_string(xStride) + "," + to_string(yStride) + ").");
    out_h = preLoc->ny / yStride;
    out_w = preLoc->nx / xStride;
    pad_h = (out_h - 1) * yStride + W.front().dense.getH() - preLoc->ny;
    pad_w = (out_w - 1) * xStride + W.front().dense.getW() - preLoc->nx;

    if (pad_h % 2 != 0 || pad_w % 2 != 0)
      throw logic_error("The pad size is not integer: (" +
                        to_string(pad_h / 2.0) + "," + to_string(pad_w / 2.0) +
                        ").");
    pad_h /= 2;
    pad_w /= 2;

    if (!getUpdateGSynFromPostPerspective()) {
      cudnnStatusCheck(cudnnSetConvolution2dDescriptor(
                           cudnnConvolutionDescriptor, pad_h, pad_w, xStride,
                           yStride, 1.0, 1.0, CUDNN_CONVOLUTION),
                       "set 2D convolution descriptor");
    } else {
      cudnnStatusCheck(cudnnSetConvolution2dDescriptor(
                           cudnnConvolutionDescriptor, pad_h, pad_w, xStride,
                           yStride, 1.0, 1.0, CUDNN_CROSS_CORRELATION),
                       "set 2D convolution descriptor");
    }

    cudnnStatusCheck(
        cudnnSetFilter4dDescriptor(
            cudnnFilterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            W.front().dense.getN(), W.front().dense.getC(),
            W.front().dense.getH(), W.front().dense.getW()),
        "set 4D filter");

    int status = computeTransposeMap();
    if (getChannel() != 0) status = findCudnnAlgo();
    return status;
  }

  return PV_SUCCESS;
}

int HyPerConnGPU::deliver() {
  int channelNum = getChannel();
  int preLayerSize = preSynapticLayer()->getCLayer()->numNeurons;
  pvdata_t* postDeviceData = (dynamic_cast<ANNLayerGPU*>(postSynapticLayer()))
                                 ->getGSyn()
                                 .at(channelNum)
                                 .dense.getDeviceData();
  try {
    if (isWeightSparse) {
      cerr << "No sparse weight implementation yet.\n";
      return PV_FAILURE;
    } else {
      if (!isPreGPULayer) {
        pvdata_t* preHostData = preSynapticLayer()->getActivity();

        PreNHWC->dense.getCudaVector().setDeviceData(preLayerSize, preHostData);

        /* change pre-layer layout (need a parameter to specify whether
         * pre-layer is a GPULCA layer or not ) */
        pvdata_t alpha = 1, beta = 0;
        cudnnStatus_t cudnnStatus = cudnnTransformTensor(
            cudnnHandle, &alpha, cudnnTensorDescriptorPreNHWC,
            PreNHWC->dense.getDeviceData(), &beta, cudnnTensorDescriptorPre,
            Pre->dense.getDeviceData());
        cudnnStatusCheck(cudnnStatus, "cudnnTransformTensor");
      }

      /* convolution  */
      int alpha = 1, beta = 1;
      auto convolFunc = [&](PVCudaWrapper<pvwdata_t>& w) {
        cudnnStatus_t cudnnStatus = cudnnConvolutionForward(
            cudnnHandle, &alpha, cudnnTensorDescriptorPre,
            Pre->dense.getDeviceData(), cudnnFilterDescriptor,
            w.dense.getDeviceData(), cudnnConvolutionDescriptor, algoFwd,
            workspaceForward.getDeviceData(), workspaceSizeForward, &beta,
            cudnnTensorDescriptorPost, postDeviceData);
        cudnnStatusCheck(cudnnStatus, "convolution");
      };

      std::for_each(W.begin(), W.end(), convolFunc);
    }
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }
  return PV_SUCCESS;
}

void HyPerConnGPU::initialize_base() {
  isPreGPULayer = true;
  isWeightSparse = false;
}

int HyPerConnGPU::initialize() {
  /*  cuDnn initialization */
  cudnnStatusCheck(cudnnCreate(&cudnnHandle), "create handle");
  cudnnStatusCheck(cudnnCreateTensorDescriptor(&cudnnTensorDescriptorPreNHWC),
                   "create tensor descriptor");
  cudnnStatusCheck(cudnnCreateTensorDescriptor(&cudnnTensorDescriptorPre),
                   "create tensor descriptor");
  cudnnStatusCheck(cudnnCreateTensorDescriptor(&cudnnTensorDescriptorPost),
                   "create tensor descriptor");
  cudnnStatusCheck(
      cudnnCreateConvolutionDescriptor(&cudnnConvolutionDescriptor),
      "create convolution descriptor");
  cudnnStatusCheck(cudnnCreateFilterDescriptor(&cudnnFilterDescriptor),
                   "create filter descriptor");

  if (isWeightSparse) {
    try {
      cusparseStatusCheck(cusparseCreate(&cusparseHandle), "create handle");
      cusparseStatusCheck(cusparseCreateMatDescr(&cusparseMatDescr),
                          "create matrix descriptor");
    } catch (exception& e) {
      cerr << e.what() << endl;
      return PV_FAILURE;
    }
  }

  return PV_SUCCESS;
}

int HyPerConnGPU::update_dW(int arborId) { return PV_SUCCESS; }

int HyPerConnGPU::updateWeights(int arborId) {
  int status = transposeWeight();
  return status;
}

int HyPerConnGPU::findCudnnAlgo() {
  int n, c, h, w;
  const PVLayerLoc* postLoc = postSynapticLayer()->getLayerLoc();
  cudnnStatusCheck(cudnnGetConvolution2dForwardOutputDim(
                       cudnnConvolutionDescriptor, cudnnTensorDescriptorPre,
                       cudnnFilterDescriptor, &n, &c, &h, &w),
                   "cudnnGetConvolution2dForwardOutputDim");

  if (c != postLoc->nf || h != postLoc->ny || w != postLoc->nx) {
    cout << ("Convolution result dimension mismatched.\n" + to_string(c) + " " +
             to_string(h) + " " + to_string(w) + " vs. " +
             to_string(postLoc->nf) + " " + to_string(postLoc->ny) + " " +
             to_string(postLoc->nx) + " ") << endl;
    return PV_FAILURE;
  }

  int m = 8;
  std::vector<cudnnConvolutionFwdAlgoPerf_t> p =
      std::vector<cudnnConvolutionFwdAlgoPerf_t>(m);
  cudnnStatusCheck(cudnnFindConvolutionForwardAlgorithm(
                       cudnnHandle, cudnnTensorDescriptorPre,
                       cudnnFilterDescriptor, cudnnConvolutionDescriptor,
                       cudnnTensorDescriptorPost, m, &n, p.data()),
                   "cudnnFindConvolutionForwardAlgorithm");

  cudnnStatusCheck(
      cudnnGetConvolutionForwardAlgorithm(
          cudnnHandle, cudnnTensorDescriptorPre, cudnnFilterDescriptor,
          cudnnConvolutionDescriptor, cudnnTensorDescriptorPost,
          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algoFwd),
      "cudnnGetConvolutionForwardAlgorithm");

  cudnnStatusCheck(
      cudnnGetConvolutionForwardWorkspaceSize(
          cudnnHandle, cudnnTensorDescriptorPre, cudnnFilterDescriptor,
          cudnnConvolutionDescriptor, cudnnTensorDescriptorPost, algoFwd,
          &workspaceSizeForward),
      "cudnnGetConvolutionForwardWorkspaceSize");

  workspaceForward.resize(workspaceSizeForward);

  return PV_SUCCESS;
}

int HyPerConnGPU::ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) {
  HyPerConn::ioParamsFillGroup(ioFlag);
  ioParam_isPreGPULayer(ioFlag);
  ioParam_isWeightSparse(ioFlag);

  if (!isWeightSparse) {
    cerr << "HyPerConnGPU doesn't support sparse weight right now.\n";
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

void HyPerConnGPU::ioParam_isPreGPULayer(enum PV::ParamsIOFlag ioFlag) {
  parent->ioParamValue(ioFlag, name, "isPreGPULayer", &isPreGPULayer,
                       isPreGPULayer, false);
}

void HyPerConnGPU::ioParam_isWeightSparse(enum PV::ParamsIOFlag ioFlag) {
  parent->ioParamValue(ioFlag, name, "isWeightSparse", &isWeightSparse,
                       isWeightSparse, false);
}

int HyPerConnGPU::computeTransposeMap() {
  try {
    if (getIsWeightSparseFlag()) {
      cerr << "TransposeConnGPU doesn't support sparse weight right now.\n";
      return PV_FAILURE;
    } else {
      int k = WT.front().dense.getN(), c = WT.front().dense.getC(),
          hw = WT.front().dense.getH() * WT.front().dense.getW();
      int current = 0;
      auto backpermuteFunctor = [&]() {
        int c = current % hw,
            b = (int)floor(((float)current) / ((float)hw)) % k,
            a = floor(((float)floor(((float)current) / ((float)hw))) /
                      ((float)k));
        current++;
        return b * c * hw + a * hw + c;
      };

      std::generate_n(map.getHostData(), map.getSize(), backpermuteFunctor);
      map.host2Device();
    }
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }
  return PV_SUCCESS;
}

int HyPerConnGPU::transposeWeight() {
  try {
    auto it1 = W.begin(), it2 = WT.begin(), end1 = W.end(), end2 = WT.end();
    for (; (it1 != end1) && (it2 != end2);
         std::advance(it1, 1), std::advance(it2, 1)) {
      permuteWeight(
          (*it2).dense.getSize(), (*it1).dense.getDeviceData(),
          (*it2).dense.getDeviceData(), map.getDeviceData());
      cudaStatusCheck("permuting weight");
    }
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }
  return PV_SUCCESS;
}
}
