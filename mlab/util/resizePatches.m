function resizePatches(inputweightfile, outputweightfile, new_nxp, new_nyp, nxGlobalPost, nyGlobalPost, x_offset, y_offset);
% resizeToShrunken(inputweightfile, outputweightfile, new_nxp, new_nyp, nxGlobalPost, nyGlobalPost, x_offset, y_offset)
%
% Takes a pvp file as input (either shared weights or non-shared weights), and creates a
% new pvpfile with new patch size given by new_nxp and new_nyp.
%
% Inputs:
%     inputweightfile is a string indicating the path of the input pvp file.
%     outputweightfile is a string indicating the path of the output pvp file.
%         If the file exists, it will be clobbered.
%         Currently, it should work to have inputweightfile and outputweight file be the same
%         path, to modify weights in place, but this is not guaranteed to work in the future.
%
%     new_nxp, new_nyp.  The new patch dimensions.  Note that nfp cannot be changed in this script.
%         If new_nxp is less than the inputweightfile's nxp, weights on the edges of the patches will
%         be discarded.  This use was the motivation for this file, to provide a means to convert
%         pvp files created when nxpShrunken was a parameter and could be less than nxp, to pvp files
%         whose nxp was the nxpShrunken.
%
%         If new_nxp is greater than the inputweightfile's nxp, the new patches will be padded with zeros.
%
%         The above applies to new_nyp and nyp in the same way.
%
%     nxGlobalPost, nyGlobalPost.  The dimensions of the restricted postsynaptic layer.  If inputweightfile
%         points to a shared-weights file, these arguments are ignored.  However, if weights are not shared,
%         these arguments are needed in order to call writepvpweightfile.m
%
%     x_offset, y_offset.  x_offset is the number of weights in the x-direction to discard from the start
%         of the weight patch to discard.  If weights are being added to the start of the weight patch,
%         x_offset should be negative.  The default for x_offset is floor( (nxp - new_nxp)/2 ).
%
%     The above applies to y_offset, nyp, and new_nyp in the same way.
%
%     (Note: internally, because MATLAB and Octave are column-major and PetaVision is row-major, patches
%     have nxp rows and nxp columns.  x_offset specifies the number of rows at the beginning to discard,
%     and nyp specifies the number of columns at the beginning to discard.  Since the input parameters
%     are paths and the resulting weights is not returned as a matlab/octave variable, the issues of
%     row-major versus column-major should be invisible when using this function m-file.

[pvpdata,hdr] = readpvpfile(inputweightfile);
fn = fieldnames(hdr);
for k=1:numel(fn)
    if isequal(fn{k},'nxp')
        infilenxp = hdr.nxp;
    elseif isequal(fn{k},'nyp')
        infilenyp = hdr.nyp;
    else
       % skip
    end%if
end%for
if ~exist('infilenxp','var') || ~exist('infilenyp','var')
    error('resizePatches:notweightfile','inputfile %s is not a weight file.\n',inputweightfile);
end%if

numframes = numel(pvpdata);
for frame=1:numframes
    timestamp = pvpdata{frame}.time;
    numarbors = numel(pvpdata{frame}.values);
    for arbor=1:numarbors;
        assert(infilenxp == size(pvpdata{frame}.values{arbor},1)); % see above comment regarding row-major versus column-major. 
        assert(infilenyp == size(pvpdata{frame}.values{arbor},2));
        if ~exist('x_offset','var')
            x_offset = floor( (infilenxp - new_nxp)/2);
        end%if
        if ~exist('y_offset','var')
            y_offset = floor( (infilenyp - new_nyp)/2);
        end%if
        nfp = size(pvpdata{frame}.values{arbor},3);
        numpatches = size(pvpdata{frame}.values{arbor},4);
        patchdata = zeros(new_nxp, new_nyp, nfp,numpatches);

        xstartnew = max(1,1+(-x_offset));
        xstartold = max(1,1+x_offset);
        xstopold = xstartold + new_nxp-1;
        xstopnew = xstartnew + new_nxp-1;
        edge = max(xstopold-infilenxp,xstopnew-new_nxp);
        if (edge>0)
            xstopold = xstopold - edge;
            xstopnew = xstopnew - edge;
        end%if
        assert(xstopold-xstartold==xstopnew-xstartnew);
        assert(xstopnew>=1 && xstopnew <= new_nxp);
        assert(xstopold>=1 && xstopold <= infilenxp);

        ystartnew = max(1,1+(-y_offset));
        ystartold = max(1,1+y_offset);
        ystopold = ystartold + new_nyp-1;
        ystopnew = ystartnew + new_nyp-1;
        edge = max(ystopold-infilenyp,ystopnew-new_nyp);
        if (edge>0)
            ystopold = ystopold - edge;
            ystopnew = ystopnew - edge;
        end%if
        assert(ystopold-ystartold==ystopnew-ystartnew);
        assert(ystopnew>=1 && ystopnew <= new_nyp);
        assert(ystopold>=1 && ystopold <= infilenyp);

        for patchindex=1:numpatches
            patchdata(xstartnew:xstopnew,ystartnew:ystopnew,:,patchindex) = pvpdata{frame}.values{arbor}(xstartold:xstopold,ystartold:ystopold,:,patchindex);
        end%for patchindex
        pvpdata{frame}.values{arbor} = patchdata;
    end%for % arbors
end%for % frames

if hdr.filetype == 5 % shared weights
    writepvpsharedweightfile(outputweightfile, pvpdata);
else
    nxGlobalPre = hdr.nxGlobal;
    nyGlobalPre = hdr.nyGlobal;
    nfPre = hdr.nf;
    nbPre = hdr.nb;
    % nxGlobalPost set in input arguments
    % nyGlobalPost set in input arguments
    postweightsflag = false;
    writepvpweightfile(outputweightfile, pvpdata, nxGlobalPre, nyGlobalPre, nfPre, nbPre, nxGlobalPost, nyGlobalPost, postweightsflag);
end%if
