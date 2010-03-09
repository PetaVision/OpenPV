/*
 * geisler.cpp
 *
 */

#include <stdlib.h>

#include <src/columns/HyPerCol.hpp>
#include <src/connections/HyPerConn.hpp>
#include <src/connections/KernelConn.hpp>
#include <src/connections/CocircConn.hpp>
#include <src/connections/AvgConn.hpp>
#include <src/layers/Image.hpp>
#include <src/layers/Retina.hpp>
#include <src/layers/V1.hpp>
#include <src/io/ConnectionProbe.hpp>
#include <src/io/GLDisplay.hpp>
#include <src/io/PostConnProbe.hpp>
#include <src/io/LinearActivityProbe.hpp>
#include <src/io/PointProbe.hpp>
#include <src/io/StatsProbe.hpp>

#include <src/io/imageio.hpp>

using namespace PV;

int main(int argc, char* argv[]) {
	// create the managing hypercolumn
	//
	HyPerCol * hc = new HyPerCol("column", argc, argv);

	// create the visualization display
	//
	//GLDisplay * display = new GLDisplay(&argc, argv, hc, 2, 2);

	// create the image
	//
	const char * amoeba_filename = "./input/hard4.bmp";
	Image * image = new Image("Image", hc, amoeba_filename);
	//display->setDelay(0);
	//display->setImage(image);

	// create the layers
	//
	HyPerLayer * retina = new Retina("Retina", hc, image);
	HyPerLayer * lgn = new V1("LGN", hc);
	HyPerLayer * lgninh = new V1("LGNInh", hc);
	HyPerLayer * l1 = new V1("L1", hc);
	HyPerLayer * l1inh = new V1("L1Inh", hc);

	// create averaging layers
//	HyPerLayer * l1avg = new V1("L1Avg", hc);

	//display->addLayer(l1);
	//display->addLayer(l1inh);
	//display->addLayer(l1avg);

	// create the connections
	//
	HyPerConn * r_lgn =
		new KernelConn("Retina to LGN", hc, retina, lgn,
			CHANNEL_EXC);
	HyPerConn * lgn_l1 =
		new KernelConn("LGN to L1",     hc, lgn,    l1,
			CHANNEL_EXC);
	HyPerConn * lgninh_lgn =
		new KernelConn("LGNInh to LGN", hc, lgninh, lgn,
			CHANNEL_INH);
	HyPerConn * lgninh_lgninh_exc =
		new KernelConn("LGNInh to LGNInh Exc", hc, lgninh, lgninh,
			CHANNEL_EXC);
	HyPerConn * lgninh_lgninh =
		new KernelConn("LGNInh to LGNInh Inh", hc, lgninh, lgninh,
			CHANNEL_INH);
	HyPerConn * l1_lgn =
		new KernelConn("L1 to LGN",  hc, l1,     lgn,
			CHANNEL_EXC);
	HyPerConn * l1_lgninh =
		new KernelConn("L1 to LGNInh",  hc, l1,     lgninh,
			CHANNEL_EXC);
	const char * kernel_filename_exc = "./input/hard4_smallkernel_exc.pvp";
	HyPerConn * l1_l1 =
		new KernelConn("L1 to L1",      hc, l1,     l1,
			CHANNEL_EXC, kernel_filename_exc);
	HyPerConn * l1_l1inh =
		new KernelConn("L1 to L1Inh",   hc, l1,     l1inh,
			CHANNEL_EXC, kernel_filename_exc);
	HyPerConn * l1inh_l1 =
		new CocircConn("L1Inh to L1",   hc, l1inh,  l1,
			CHANNEL_INH);
	HyPerConn * l1inh_l1inh_exc =
		new CocircConn("L1Inh to L1Inh Exc",   hc, l1inh,  l1inh,
			CHANNEL_EXC);
	HyPerConn * l1inh_l1inh_inh =
		new CocircConn("L1Inh to L1Inh Inh",   hc, l1inh,  l1inh,
			CHANNEL_INH);

	// create averaging connections
	//
//	HyPerConn * l1_l1avg   = new AvgConn("L1 to L1Avg", hc, l1, l1avg, CHANNEL_EXC, NULL);


	// add probes

	HyPerLayer * displayLayer = retina;

	const int nyDisplay = displayLayer->clayer->loc.ny;

	LayerProbe * statsretina = new StatsProbe(BufActivity,     "Retina :");
	LayerProbe * statslgn = new StatsProbe(BufActivity,        "LGN :");
	LayerProbe * statslgninh = new StatsProbe(BufActivity,     "LGNInh :");
	LayerProbe * statsl1 = new StatsProbe(BufActivity,         "L1     :");
	LayerProbe * statsl1inh = new StatsProbe(BufActivity,       "L1Inh:  ");
//	LayerProbe * statsl1avg = new StatsProbe(BufActivity,      "L1Avg  :");

	retina->insertProbe(statsretina);
	lgn->insertProbe(statslgn);
	lgninh->insertProbe(statslgninh);
	l1->insertProbe(statsl1);
	l1inh->insertProbe(statsl1inh);
//	l1avg->insertProbe(statsl1avg);

	int npad, nx, ny, nf;

	npad = lgn->clayer->loc.nPad;
	nx = lgn->clayer->loc.nx;
	ny = lgn->clayer->loc.ny;
	nf = lgn->clayer->loc.nBands;

	const char * Vmem_filename_LGNa1 = "Vmem_LGNa1.txt";
	LayerProbe * Vmem_probe_LGNa1 =
		new PointProbe(Vmem_filename_LGNa1, 54, 119, 0, "LGNA1:(78,102,0)");
	lgn->insertProbe(Vmem_probe_LGNa1);

	const char * Vmem_filename_LGNa2 = "Vmem_LGNa2.txt";
	LayerProbe * Vmem_probe_LGNa2 =
		new PointProbe(Vmem_filename_LGNa2, 78, 108, 0, "LGNA2:(78,102,0)");
	lgn->insertProbe(Vmem_probe_LGNa2);

	const char * Vmem_filename_LGNa3 = "Vmem_LGNa3.txt";
	LayerProbe * Vmem_probe_LGNa3 =
		new PointProbe(Vmem_filename_LGNa3, 73, 94, 0, "LGNA3:(78,102,0)");
	lgn->insertProbe(Vmem_probe_LGNa3);

	const char * Vmem_filename_LGNc1 = "Vmem_LGNc1.txt";
	LayerProbe * Vmem_probe_LGNc1 =
		new PointProbe(Vmem_filename_LGNc1, 100, 52, 0, "LGNC1:(78,102,0)");
	lgn->insertProbe(Vmem_probe_LGNc1);

	const char * Vmem_filename_LGNc2 = "Vmem_LGNc2.txt";
	LayerProbe * Vmem_probe_LGNc2 =
		new PointProbe(Vmem_filename_LGNc2, 111, 89, 0, "LGNC2:(78,102,0)");
	lgn->insertProbe(Vmem_probe_LGNc2);

	const char * Vmem_filename_LGNc3 = "Vmem_LGNc3.txt";
	LayerProbe * Vmem_probe_LGNc3 =
		new PointProbe(Vmem_filename_LGNc3, 106, 113, 0, "LGNC3:(78,102,0)");
	lgn->insertProbe(Vmem_probe_LGNc3);


	npad = lgninh->clayer->loc.nPad;
	nx = lgninh->clayer->loc.nx;
	ny = lgninh->clayer->loc.ny;
	nf = lgninh->clayer->loc.nBands;

	const char * Vmem_filename_LGNInha1 = "Vmem_LGNInha1.txt";
	LayerProbe * Vmem_probe_LGNInha1 =
		new PointProbe(Vmem_filename_LGNInha1, 71, 119, 0, "LGNInhA1:(78,102,0)");
	lgninh->insertProbe(Vmem_probe_LGNInha1);

	const char * Vmem_filename_LGNInha2 = "Vmem_LGNInha2.txt";
	LayerProbe * Vmem_probe_LGNInha2 =
		new PointProbe(Vmem_filename_LGNInha2, 58, 91, 0, "LGNInhA2:(78,102,0)");
	lgninh->insertProbe(Vmem_probe_LGNInha2);

	const char * Vmem_filename_LGNInha3 = "Vmem_LGNInha3.txt";
	LayerProbe * Vmem_probe_LGNInha3 =
		new PointProbe(Vmem_filename_LGNInha3, 78, 110, 0, "LGNInhA3:(78,102,0)");
	lgninh->insertProbe(Vmem_probe_LGNInha3);

	const char * Vmem_filename_LGNInhc1 = "Vmem_LGNInhc1.txt";
	LayerProbe * Vmem_probe_LGNInhc1 =
		new PointProbe(Vmem_filename_LGNInhc1, 88, 73, 0, "LGNInhC1:(78,102,0)");
	lgninh->insertProbe(Vmem_probe_LGNInhc1);

	const char * Vmem_filename_LGNInhc2 = "Vmem_LGNInhc2.txt";
	LayerProbe * Vmem_probe_LGNInhc2 =
		new PointProbe(Vmem_filename_LGNInhc2, 83, 105, 0, "LGNInhC2:(78,102,0)");
	lgninh->insertProbe(Vmem_probe_LGNInhc2);

	const char * Vmem_filename_LGNInhc3 = "Vmem_LGNInhc3.txt";
	LayerProbe * Vmem_probe_LGNInhc3 =
		new PointProbe(Vmem_filename_LGNInhc3, 75, 86, 0, "LGNInhC3:(78,102,0)");
	lgninh->insertProbe(Vmem_probe_LGNInhc3);

	npad = l1->clayer->loc.nPad;
	nx = l1->clayer->loc.nx;
	ny = l1->clayer->loc.ny;
	nf = l1->clayer->loc.nBands;

	const char * Vmem_filename_V1a1 = "Vmem_V1a1.txt";
	LayerProbe * Vmem_probe_V1a1 =
		new PointProbe(Vmem_filename_V1a1, 78, 102, 4, "V1A1:(78,102,0)");
	l1->insertProbe(Vmem_probe_V1a1);

	const char * Vmem_filename_V1a2 = "Vmem_V1a2.txt";
	LayerProbe * Vmem_probe_V1a2 =
		new PointProbe(Vmem_filename_V1a2, 71, 119, 10, "V1A2:(78,102,0)");
	l1->insertProbe(Vmem_probe_V1a2);

	const char * Vmem_filename_V1a3 = "Vmem_V1a3.txt";
	LayerProbe * Vmem_probe_V1a3 =
		new PointProbe(Vmem_filename_V1a3, 72, 93, 7, "V1A3:(78,102,0)");
	l1->insertProbe(Vmem_probe_V1a3);

	const char * Vmem_filename_V1c1 = "Vmem_V1c1.txt";
	LayerProbe * Vmem_probe_V1c1 =
		new PointProbe(Vmem_filename_V1c1, 96, 46, 6, "V1C1:(78,102,0)");
	l1->insertProbe(Vmem_probe_V1c1);

	const char * Vmem_filename_V1c2 = "Vmem_V1c2.txt";
	LayerProbe * Vmem_probe_V1c2 =
		new PointProbe(Vmem_filename_V1c2, 88, 38, 3, "V1C2:(78,102,0)");
	l1->insertProbe(Vmem_probe_V1c2);

	const char * Vmem_filename_V1c3 = "Vmem_V1c3.txt";
	LayerProbe * Vmem_probe_V1c3 =
		new PointProbe(Vmem_filename_V1c3, 110, 99, 1, "V1C3:(78,102,0)");
	l1->insertProbe(Vmem_probe_V1c3);

	npad = l1inh->clayer->loc.nPad;
	nx = l1inh->clayer->loc.nx;
	ny = l1inh->clayer->loc.ny;
	nf = l1inh->clayer->loc.nBands;

	const char * Vmem_filename_V1Inha1 = "Vmem_V1Inha1.txt";
	LayerProbe * Vmem_probe_V1Inha1 =
		new PointProbe(Vmem_filename_V1Inha1, 78, 108, 8, "V1InhA1:(78,102,0)");
	l1inh->insertProbe(Vmem_probe_V1Inha1);

	const char * Vmem_filename_V1Inha2 = "Vmem_V1Inha2.txt";
	LayerProbe * Vmem_probe_V1Inha2 =
		new PointProbe(Vmem_filename_V1Inha2, 53, 94, 4, "V1InhA2:(78,102,0)");
	l1inh->insertProbe(Vmem_probe_V1Inha2);

	const char * Vmem_filename_V1Inha3 = "Vmem_V1Inha3.txt";
	LayerProbe * Vmem_probe_V1Inha3 =
		new PointProbe(Vmem_filename_V1Inha3, 49, 106, 0, "V1InhA3:(78,102,0)");
	l1inh->insertProbe(Vmem_probe_V1Inha3);

	const char * Vmem_filename_V1Inhc1 = "Vmem_V1Inhc1.txt";
	LayerProbe * Vmem_probe_V1Inhc1 =
		new PointProbe(Vmem_filename_V1Inhc1, 105, 57, 1, "V1InhC1:(78,102,0)");
	l1inh->insertProbe(Vmem_probe_V1Inhc1);

	const char * Vmem_filename_V1Inhc2 = "Vmem_V1Inhc2.txt";
	LayerProbe * Vmem_probe_V1Inhc2 =
		new PointProbe(Vmem_filename_V1Inhc2, 69, 51, 9, "V1InhC2:(78,102,0)");
	l1inh->insertProbe(Vmem_probe_V1Inhc2);

	const char * Vmem_filename_V1Inhc3 = "Vmem_V1Inhc3.txt";
	LayerProbe * Vmem_probe_V1Inhc3 =
		new PointProbe(Vmem_filename_V1Inhc3, 102, 70, 1, "V1InhC3:(78,102,0)");
	l1inh->insertProbe(Vmem_probe_V1Inhc3);


	if (0) { // ma
		LinearActivityProbe * laProbes[nyDisplay]; // array of ny pointers to PV::LinearActivityProbe

		for (int i = 0; i < nyDisplay; i++) {
			laProbes[i] = new PV::LinearActivityProbe(hc, PV::DimX, i, 0);
			retina->insertProbe(laProbes[i]);
		}
	}
	// run the simulation
	hc->initFinish();

	// write text weights
	const char * r_lgn_filename = "r_lgn_gauss.txt";
	HyPerLayer * pre = r_lgn->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	r_lgn->writeTextWeights(r_lgn_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

	const char * lgn_l1_filename = "lgn_l1_gauss.txt";
	pre = lgn_l1->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	lgn_l1->writeTextWeights(lgn_l1_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

	const char * lgninh_lgn_filename = "lgninh_lgn_gauss.txt";
	pre = lgninh_lgn->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	lgninh_lgn->writeTextWeights(lgninh_lgn_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

	const char * l1_lgn_filename = "l1_lgn_gauss.txt";
	pre = l1_lgn->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	l1_lgn->writeTextWeights(l1_lgn_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

	const char * l1_lgninh_filename = "l1_lgninh_gauss.txt";
	pre = l1_lgninh->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	l1_lgninh->writeTextWeights(l1_lgninh_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

	const char * l1_l1_filename = "l1_l1_geisler_exc.txt";
	pre = l1_l1->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	l1_l1->writeTextWeights(l1_l1_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);
	const char * l1_l1inh_filename = "l1_l1inh_geisler_exc.txt";
	l1_l1inh->writeTextWeights(l1_l1inh_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);
	const char * l1inh_l1_filename = "l1inh_l1_geisler_inh.txt";
	l1inh_l1->writeTextWeights(l1inh_l1_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

	hc->run();



	/* clean up (HyPerCol owns layers and connections, don't delete them) */
	delete hc;


	return 0;
}
