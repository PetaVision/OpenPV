/*
 * test_marginwidth_correctsize_many_to_one.cpp
 * Tests HyPerCol::checkPatchSize() (a private method called by HyPerCol::run(int) )
 * on a many-to-one connection where the margin width is the correct size.
 *
 * There are many other tests in the test_marginwidth_* family to test the
 * other possibilities of type of connection and size of marginWidth.
 */
#include "../src/columns/HyPerCol.hpp"
#include "../src/connections/KernelConn.hpp"
#include "../src/layers/Retina.hpp"
#include "../src/layers/ANNLayer.hpp"

#define ARGC 3

using namespace PV;

int main(int argc, char * argv[])
{
   HyPerCol * hc;
   Retina * pre;
   ANNLayer * post;
   KernelConn * conn;
   char * cl_args[3];
   cl_args[0] = strdup(argv[0]);
   cl_args[1] = strdup("-p");
   cl_args[2] = strdup("input/test_marginwidth_correctsize_many_to_one.pv");

   hc = new HyPerCol("test_marginwidth_many_to_one column", 3, cl_args);
   pre = new Retina("presynaptic layer", hc);
   post = new ANNLayer("postsynaptic layer", hc);
   conn = new KernelConn("pre to post connection", hc, pre, post, CHANNEL_EXC);
   int status = hc->run();
   delete hc;
   for( int k=0; k<ARGC; k++ )
   {
      free(cl_args[k]);
   }
   return status;
}
