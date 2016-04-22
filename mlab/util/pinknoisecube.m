function [Yre,Yim] = pinknoisecube(ny,nx,nz)
% [Yre,Yim] = pinknoisecube(ny,nx,nz)
% Yre and Yim will be independent ny-by-nx-by-nz pinknoise arrays
Nx = ceil(nx/2)*2;
Ny = ceil(ny/2)*2;
Nz = ceil(nz/2)*2;

assert(mod(Nx,2)==0 && mod(Ny,2)==0 && mod(Nz,2)==0);

X = repmat((-Nx/2:Nx/2-1),[Ny,1,Nz]);
Y = repmat((-Ny/2:Ny/2-1)',[1,Nx,Nz]);
Z = repmat(reshape(-Nz/2:Nz/2-1,[1 1 Nz]), [Ny Nx 1]);
rsq = X.^2 + Y.^2 + Z.^2;

% The power spectrum falls off in such a way that the integral over the
% spherical shell between r1 and r2 depends only on the ratio r2/r1.
% In three dimensions, the power spectrum is 1/r^3, so that the integral is
% \int_{r1<r<r2} 1/r^3*(r^2 sin(theta)dr) = 4\pi*(log(r2)-log(r1))
powerspectrum = rsq.^(-3/2);
powerspectrum(rsq==0)=0;

randphase = exp(2i*pi*rand(Ny,Nx,Nz));
fhat = powerspectrum .* randphase;

f = ifftn(fhat);

Yre = real(f);
chkbd = zeros(2,2,2);
chkbd(:,:,1) = [1 -1; -1 1];
chkbd(:,:,2) = [-1 1; 1 -1];
chkbd = repmat(chkbd,[Ny/2,Nx/2,Nz/2]);
Yre = real(f).*chkbd;
Yre = Yre(1:ny,1:nx,1:nz);
if nargout > 1
    Yim = imag(f).*chkbd;
    Yim = Yim(1:ny,1:nx,1:nz);
end
