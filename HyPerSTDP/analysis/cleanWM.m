function w=cleanWM(wm, ncells, hdr, ign_w, img_size)

    w = zeros(img_size*sqrt(ncells), img_size*sqrt(ncells));

    for v=1:ncells %Loop over cells

        [r c] = ind2sub([sqrt(ncells) sqrt(ncells)], v);

        wtmp=wm((r-1)*hdr.nxp+1:r*hdr.nxp, (c-1)*hdr.nyp+1:c*hdr.nyp);
        rold = r;
        cold = c;
        r = cold;
        c = rold;
        w((rold-1)*img_size+1:rold*img_size,(cold-1)*img_size+1:cold*img_size) = wtmp((hdr.nxp-(ign_w*(r-1)-1))-img_size:(hdr.nxp-(ign_w*(r-1))), (hdr.nxp-(ign_w*(c-1)-1))-img_size:(hdr.nxp-(ign_w*(c-1))));
    end
    keyboard
end