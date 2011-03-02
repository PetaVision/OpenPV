Patch size and margin width requirements

The internals of PetaVision make certain assumptions regarding layers'
neuron densities, connections' patch sizes, and layers' margin widths. The
code tests for these conditions and exits with an error if they are not met.
For some of these conditions, it may be technically possible, even
desirable, to eliminate the assumption---however the current incarnation of
the code has not been tested on cases where the assumptions do not hold.

The following discusses the x-direction (parameter nxScale in the layers and
parameter nxp in the connections), but the same applies to the y-direction
parameters. At present, a single marginWidth covers the x- and y-directions;
it must be large enough to accomodate the requirements of both dimensions.

nxScale must be an integral power of two (1,2,4,... or 1/2, 1/4, 1/8,...).
If the input parameters specify an nxScale that is not an integral power of
two, then nxScale is effectively replaced with 2^round(log2(nxScale)).
(Internally, the scale parameter is converted to xScale = -log2(nxScale) )

nxp is the patch size of each pre-synaptic neuron's connection to the
post-synaptic layer.  The parameters file does not need to specify
the size of the patch in the pre-synaptic layer that each post-synaptic
neuron is connected to.

Every connection imposes requirements on the connection's patch size nxp
and on its pre-synaptic layer's marginWidth.  A connection does *not* impose a
requirement on the post-synaptic marginWidth.  The requirements depend on the
relative neural densities of the connection's pre-synaptic and post-synaptic
layers.  For brevity, the descriptions below use the following notation:
nxScalePre:     pre-synaptic layer's nxScale parameter
nxScalePost:    post-synaptic layer's nxScale parameter
nxScaleRatio:   nxScalePost/nxScalePre
marginWidthPre: the pre-synaptic layer's marginWidth.

===============================================================

Case 1: nxScalePost = nxScalePre.

nxp must be odd.

marginWidthPre must be at least (nxp-1)/2.

===============================================================

Case 2: nxScalePost > nxScalePre.
(the post-synaptic layer has more neurons than the pre-synaptic layer)

nxp must be an odd multiple of the nxScaleRatio

marginWidthPre must be at least ( (nxp/nxScaleRatio) - 1 )/2

Example:  If the presynaptic layer is 32-by-32 and the postsynaptic
layer is 128-by-128, then nxScaleRatio = 4.
Hence, nxp must be an odd multiple of 4:  nxp = 4, 12, 20, 28, etc.
If nxp = 20, then marginWidthPre must be at least (20/4 - 1)/2 = 2.

===============================================================

Case 3: nxScalePost < nxScalePre.
(the post-synaptic layer has fewer neurons than the pre-synaptic layer)

nxp must be odd.

marginWidthPre must be at least (nxp-1)/2 * (1/nxScaleRatio)
(since nxScalePre will be a positive multiple of nxScalePost,
1/nxScaleRatio is a positive integer).

Example:  If the presynaptic layer is 128-by-128 and the postsynaptic
layer is 64-by-64, then nxScaleRatio = 0.5.
nxp must be odd.
If nxp = 7, then marginWidthPre must be at least 3*2 = 6

===============================================================

A final note:  when doing a non-MPI (single processor) run, marginWidth
can be set to zero.  The program will give a warning, but not exit with
an error.  However, the boundary effects may not be what you want.
It is preferable to give the correct marginWidth and the correct boundary
conditions.

If you give a non-zero but insufficient marginWidth, the program will
exit with an error, even in a non-MPI run.