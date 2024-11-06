#ifndef REQUIREDCONVOLVEMARGIN_HPP_
#define REQUIREDCONVOLVEMARGIN_HPP_

namespace PV {

/**
 * For a convolution-based connection between two layers, computes the
 * margin width the presynaptic layer must have, given the size of the
 * presynaptic and postsynaptic layers, and the patch size (the number of
 * postsynaptic neurons each presynaptic neuron connects to).
 * If nPre and nPost are not the same, the larger must be an 2^k times the
 * smaller for some positive integer k.
 * If nPre == nPost, patchSize must be odd.
 * If nPre > nPost (many-to-one), any patchSize is permissible.
 * If nPost > nPre (one-to-many), patchSize must be a multiple of (nPost/nPre).
 */
int requiredConvolveMargin(int nPre, int nPost, int patchSize, char axis, char const *objectName);

} // namespace PV

#endif // REQUIREDCONVOLVEMARGIN_HPP_
