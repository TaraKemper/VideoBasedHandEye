function [fre] = calProjErr(M, P2D, P3D)
% given the 3x4 projection matrix M, calculate the projection error between
% the paired 2D/3D fiducials
%
% INPUTS:  M - 3x4 camera projection matrix
%          P2D - 2xn 2D image points
%          P3D - 3xn 3D fiducials with 1:1 correspondence to P2D
%
% OUTPUT:  fre - mean projection error, i.e. euclidean distance between P3D
%                after projection from P2D
%
% Elvis C.S. Chen
% chene@robarts.ca
%
% Feb. 12, 2022
%

[m,n] = size(P3D);

% calculate the projection of P3D given M
temp = M*[P3D; ones(1,n)];
P = zeros(2,n);
P(1,:) = temp(1,:) ./ temp(3,:);
P(2,:) = temp(2,:) ./ temp(3,:);

fre = sum(vecnorm(P-P2D))/n;
end





















