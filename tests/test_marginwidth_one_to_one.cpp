/*
 * test_patch_size.cpp
 *
 * Tests HyPerCol::checkPatchSize() ( a private method called by HyPerCol::run(int) )
 * by trying to run three problems.
 * The first has a marginWidth too small; it should fail.
 * The second has a marginWidth larger than necessary; it should pass.
 * The third has a correct-sized marginWidth; it should also pass.
 */

#include "../src/columns/HyPerCol.hpp"
#include "../src/connections/KernelConn.hpp"
#include "../src/layers/Image.hpp"
#include "../src/layers/Retina.hpp"
#include "../src/layers/ANNLayer.hpp"

#include <assert.h>

using namespace PV;

int main(int argc, char * argv[]) {
    HyPerCol * hc;
    ANNLayer * layerA;

    printf("    Beginning check with too-small marginWidth...\n");
    fprintf(stderr, "If test_marginwidth_one_to_one works correctly, there will be two margin width errors below.\n");
    hc = new HyPerCol("test_marginwidth_one_to_one column", argc, argv);
    Retina * retinaSmall = new Retina("test_marginwidth_one_to_one Retina small marginWidth", hc);
    layerA = new ANNLayer("test_marginwidth_one_to_one ANNLayer", hc);
    KernelConn * retinaSmall_layerA = new KernelConn("test_marginwidth_one_to_one Retina to ANNLayer", hc, retinaSmall, layerA, CHANNEL_EXC);
    assert(retinaSmall_layerA);
    // The small connection has pre-synaptic marginWidth smaller than needed for nxp, nyp.  This run should fail.
    int status1 = hc->run(1);
    delete hc;
    if( status1 == EXIT_FAILURE ) printf("    Too-small marginWidth failed as expected.\n");
    else fprintf(stderr, "    ERROR: Too-small marginWidth succeeded when it should have failed.\n");

    printf("    Beginning check with too-large marginWidth...\n");
    hc = new HyPerCol("test_marginwidth_one_to_one column", argc, argv);
    Retina * retinaLarge = new Retina("test_marginwidth_one_to_one Retina large marginWidth", hc);
    layerA = new ANNLayer("test_marginwidth_one_to_one ANNLayer", hc);
    // need to redefine this layer since previous "delete hc;" statement deleted the layers.
    KernelConn * retinaLarge_layerA = new KernelConn("test_marginwidth_one_to_one Retina to ANNLayer", hc, retinaLarge, layerA, CHANNEL_EXC);
    assert(retinaLarge_layerA);
    // The large connection has pre-synaptic marginWidth larger than needed for nxp, nyp.  Creating the connection should pass.
    int status2 = hc->run(1);
    delete hc;
    if( status2 == EXIT_SUCCESS ) printf("    Too-large marginWidth succeeded as expected.\n");
    else fprintf(stderr, "    ERROR: Too-large marginWidth failed.\n");

    printf("    Beginning check with correct-sized marginWidth...\n");
    hc = new HyPerCol("test_marginwidth_one_to_one column", argc, argv);
    Retina * retinaCorrect = new Retina("test_marginwidth_one_to_one Retina correct marginWidth", hc);
    layerA = new ANNLayer("test_marginwidth_one_to_one ANNLayer", hc);
    // need to redefine this layer since previous "delete hc;" statement deleted the layers.
    KernelConn * retinaCorrect_layerA = new KernelConn("test_marginwidth_one_to_one Retina to ANNLayer", hc, retinaCorrect, layerA, CHANNEL_EXC);
    assert(retinaCorrect_layerA);
    // The correct-sized connection has pre-synaptic marginWidth the correct size needed for nxp, nyp.  Creating the connection should pass.
    int status3 = hc->run(1);
    delete hc;
    if( status3 == EXIT_SUCCESS ) printf("    Correct-sized marginWidth succeeded as expected.\n");
    else fprintf(stderr, "    ERROR: Correct-sized marginWidth failed.\n");

    int status = ( status1 == EXIT_FAILURE && status2 == EXIT_SUCCESS && status3 == EXIT_SUCCESS ) ? EXIT_SUCCESS : EXIT_FAILURE;
    return status;
}
