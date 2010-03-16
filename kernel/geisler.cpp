/*
 * geisler.cpp
 *
 */

#include <stdlib.h>
#include <time.h>

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

	int iseed = time(NULL);
	srand ( iseed );

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
	HyPerLayer * lgninhff = new V1("LGNInhFF", hc);
	HyPerLayer * lgninh = new V1("LGNInh", hc);
	HyPerLayer * l1 = new V1("L1", hc);
	HyPerLayer * l1inhff = new V1("L1InhFF", hc);
	HyPerLayer * l1inh = new V1("L1Inh", hc);

	// create averaging layers
//	HyPerLayer * l1avg = new V1("L1Avg", hc);

	//display->addLayer(l1);
	//display->addLayer(l1inh);
	//display->addLayer(l1avg);


	// create the connections
	//
	// retinal connections
	HyPerConn * r_lgn =
		new KernelConn("Retina to LGN", hc, retina, lgn,
			CHANNEL_EXC);
	HyPerConn * r_lgninhff =
		new KernelConn("Retina to LGNInhFF", hc, retina, lgninhff,
			CHANNEL_EXC);


	// LGN connections
	HyPerConn * lgn_lgninh =
		new KernelConn("LGN to LGNInh", hc, lgn, lgninh,
			CHANNEL_EXC);
	HyPerConn * lgn_l1 =
		new KernelConn("LGN to L1",     hc, lgn,    l1,
			CHANNEL_EXC);
	HyPerConn * lgn_l1inhff =
		new KernelConn("LGN to L1InhFF",     hc, lgninhff,    l1,
			CHANNEL_EXC);

	// LGNInhFF connections
	HyPerConn * lgninhff_lgn =
		new KernelConn("LGNInhFF to LGN", hc, lgninhff, lgn,
			CHANNEL_INH);
	HyPerConn * lgninhff_lgninhff_exc =
		new KernelConn("LGNInhFF to LGNInhFF Exc", hc, lgninhff, lgninhff,
			CHANNEL_EXC);

	// LGNInh connections
	HyPerConn * lgninh_lgn =
		new KernelConn("LGNInh to LGN", hc, lgninh, lgn,
			CHANNEL_INH);
	HyPerConn * lgninh_lgninhff =
		new KernelConn("LGNInh to LGNInhFF", hc, lgninh, lgninhff,
			CHANNEL_INH);
	HyPerConn * lgninh_lgninh_exc =
		new KernelConn("LGNInh to LGNInh Exc", hc, lgninh, lgninh,
			CHANNEL_EXC);


	// V1 connections
	const char * kernel_filename_exc = "./input/hard4_smallkernel_exc.pvp";
	HyPerConn * l1_lgn =
		new KernelConn("L1 to LGN",  hc, l1,        lgn,
			CHANNEL_EXC);
	HyPerConn * l1_lgninh =
		new KernelConn("L1 to LGNInh",  hc, l1,     lgninh,
			CHANNEL_EXC);
	HyPerConn * l1_l1 =
		new KernelConn("L1 to L1",      hc, l1,     l1,
			CHANNEL_EXC, kernel_filename_exc);
	HyPerConn * l1_l1inh =
		new KernelConn("L1 to L1Inh",   hc, l1,     l1inh,
			CHANNEL_EXC, kernel_filename_exc);

	// V1 Inh FF connections
	HyPerConn * l1inhff_l1 =
		new CocircConn("L1InhFF to L1",   hc, l1inhff,  l1,
			CHANNEL_INH);
	HyPerConn * l1inhff_l1inhff_exc =
		new CocircConn("L1InhFF to L1InhFF Exc",   hc, l1inhff,  l1inhff,
			CHANNEL_EXC);

	// V1 Inh connections
	HyPerConn * l1inh_l1 =
		new CocircConn("L1Inh to L1",   hc, l1inh,  l1,
			CHANNEL_INH);
	HyPerConn * l1inh_l1inhff =
		new CocircConn("L1Inh to L1InhFF",   hc, l1inh,  l1inhff,
			CHANNEL_INH);
	HyPerConn * l1inh_l1inh_exc =
		new CocircConn("L1Inh to L1Inh Exc",   hc, l1inh,  l1inh,
			CHANNEL_EXC);

	// create averaging connections
	//
//	HyPerConn * l1_l1avg   = new AvgConn("L1 to L1Avg", hc, l1, l1avg, CHANNEL_EXC, NULL);


	// add probes

	HyPerLayer * displayLayer = retina;

	const int nyDisplay = displayLayer->clayer->loc.ny;

	LayerProbe * statsretina = new StatsProbe(BufActivity,     "Retina :");
	LayerProbe * statslgn = new StatsProbe(BufActivity,        "LGN :");
	LayerProbe * statslgninhff = new StatsProbe(BufActivity,     "LGNInhFF :");
	LayerProbe * statslgninh = new StatsProbe(BufActivity,     "LGNInh :");
	LayerProbe * statsl1 = new StatsProbe(BufActivity,         "L1     :");
	LayerProbe * statsl1inhff = new StatsProbe(BufActivity,       "L1InhFF:  ");
	LayerProbe * statsl1inh = new StatsProbe(BufActivity,       "L1Inh:  ");
//	LayerProbe * statsl1avg = new StatsProbe(BufActivity,      "L1Avg  :");

	retina->insertProbe(statsretina);
	lgn->insertProbe(statslgn);
	lgninhff->insertProbe(statslgninhff);
	lgninh->insertProbe(statslgninh);
	l1->insertProbe(statsl1);
	l1inhff->insertProbe(statsl1inhff);
	l1inh->insertProbe(statsl1inh);
//	l1avg->insertProbe(statsl1avg);

	int npad, nx, ny, nf;

	npad = lgn->clayer->loc.nPad;
	nx = lgn->clayer->loc.nx;
	ny = lgn->clayer->loc.ny;
	nf = lgn->clayer->loc.nBands;

	const char * Vmem_filename_LGNa1 = "Vmem_LGNa1.txt";
	LayerProbe * Vmem_probe_LGNa1 =
		new PointProbe(Vmem_filename_LGNa1, 78, 102, 0, "LGNA1:(78,102,0)");
	lgn->insertProbe(Vmem_probe_LGNa1);

	const char * Vmem_filename_LGNc1 = "Vmem_LGNc1.txt";
	LayerProbe * Vmem_probe_LGNc1 =
		new PointProbe(Vmem_filename_LGNc1, 77, 101, 0, "LGNC1:(77,101,0)");
	lgn->insertProbe(Vmem_probe_LGNc1);

	const char * Vmem_filename_LGNInhFFa1 = "Vmem_LGNInhFFa1.txt";
	LayerProbe * Vmem_probe_LGNInhFFa1 =
		new PointProbe(Vmem_filename_LGNInhFFa1, 78, 102, 0, "LGNInhA1:(78,102,0)");
	lgninhff->insertProbe(Vmem_probe_LGNInhFFa1);

	const char * Vmem_filename_LGNInhFFc1 = "Vmem_LGNInhFFc1.txt";
	LayerProbe * Vmem_probe_LGNInhFFc1 =
		new PointProbe(Vmem_filename_LGNInhFFc1, 77, 101, 0, "LGNInhFFC1:(77,101,0)");
	lgninhff->insertProbe(Vmem_probe_LGNInhFFc1);

	const char * Vmem_filename_LGNInha1 = "Vmem_LGNInha1.txt";
	LayerProbe * Vmem_probe_LGNInha1 =
		new PointProbe(Vmem_filename_LGNInha1, 78, 102, 0, "LGNInhA1:(78,102,0)");
	lgninh->insertProbe(Vmem_probe_LGNInha1);

	const char * Vmem_filename_LGNInhc1 = "Vmem_LGNInhc1.txt";
	LayerProbe * Vmem_probe_LGNInhc1 =
		new PointProbe(Vmem_filename_LGNInhc1, 77, 101, 0, "LGNInhC1:(77,101,0)");
	lgninh->insertProbe(Vmem_probe_LGNInhc1);

	const char * Vmem_filename_V1a1 = "Vmem_V1a1.txt";
	LayerProbe * Vmem_probe_V1a1 =
		new PointProbe(Vmem_filename_V1a1, 78, 104, 5, "V1A1:(78,104,5)");
	l1->insertProbe(Vmem_probe_V1a1);

	const char * Vmem_filename_V1c1 = "Vmem_V1c1.txt";
	LayerProbe * Vmem_probe_V1c1 =
		new PointProbe(Vmem_filename_V1c1, 77, 101, 3, "V1C1:(77,101,3)");
	l1->insertProbe(Vmem_probe_V1c1);

	const char * Vmem_filename_V1InhFFa1 = "Vmem_V1InhFFa1.txt";
	LayerProbe * Vmem_probe_V1InhFFa1 =
		new PointProbe(Vmem_filename_V1InhFFa1, 76, 99, 4, "V1InhFFA1:(76,99,4)");
	l1inhff->insertProbe(Vmem_probe_V1InhFFa1);

	const char * Vmem_filename_V1InhFFc1 = "Vmem_V1InhFFc1.txt";
	LayerProbe * Vmem_probe_V1InhFFc1 =
		new PointProbe(Vmem_filename_V1InhFFc1, 76, 100, 4, "V1InhFFC1:(76,100,4)");
	l1inh->insertProbe(Vmem_probe_V1InhFFc1);

	const char * Vmem_filename_V1Inha1 = "Vmem_V1Inha1.txt";
	LayerProbe * Vmem_probe_V1Inha1 =
		new PointProbe(Vmem_filename_V1Inha1, 76, 99, 4, "V1InhA1:(76,99,4)");
	l1inh->insertProbe(Vmem_probe_V1Inha1);

	const char * Vmem_filename_V1Inhc1 = "Vmem_V1Inhc1.txt";
	LayerProbe * Vmem_probe_V1Inhc1 =
		new PointProbe(Vmem_filename_V1Inhc1, 76, 100, 4, "V1InhC1:(76,100,4)");
	l1inh->insertProbe(Vmem_probe_V1Inhc1);


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

	const char * lgninh_lgninh_exc_filename = "lgninh_lgninh_gap.txt";
	pre = lgninh_lgninh_exc->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	lgninh_lgninh_exc->writeTextWeights(lgninh_lgninh_exc_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

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

	const char * l1inhff_l1_filename = "l1inhff_l1_cocirc.txt";
	pre = l1inhff_l1->preSynapticLayer();
	npad = pre->clayer->loc.nPad;
	nx = pre->clayer->loc.nx;
	ny = pre->clayer->loc.ny;
	nf = pre->clayer->loc.nBands;
	l1inhff_l1->writeTextWeights(l1inhff_l1_filename, nf*(nx+npad)/2 + nf*(nx+2*npad)*(ny+2*npad)/2);

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
