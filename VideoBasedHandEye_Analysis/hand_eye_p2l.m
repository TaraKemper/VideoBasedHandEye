function [R, t] = hand_eye_p2l( X, Q, A, tol ) 
% INPUTS:   X : (3xn) 3D coordinates , ( tracker space) %
%           Q : (2xn) 2D pixel locations (image space)
%           A : (3x3) camera matrix
%           tol : exit condition 
% OUTPUTS:  R : 3x3 orthonormal rotation %
%           t : 3x1 translation

n= size (Q,2); 
e = ones(1 ,n); 
J = eye(n)-((e'*e)./n);
Q = normc(inv(A)*[Q;e]); 
Y=Q; 
err = +Inf; 
E_old = 1000*ones(3,n);

while err > tol
    [U,~,V] = svd(Y*J*X'); 
    R= U * [1 0 0; 0 1 0; 0 0 det(U*V' )] * V'; % rotation
    t= mean(Y - R*X, 2);                          
    Y= repmat(dot(R*X+ t*e, Q), [3 ,1]).*Q; % reprojection
    E= Y- R*X- t*e; err = norm(E-E_old, 'fro'); 
    E_old = E;
end