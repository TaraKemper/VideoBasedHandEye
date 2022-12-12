%clear all; close all;
%load('finalData.mat');
[M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2D,P3D);
[m,n] = size(P3D);
e = ones(1,n);
%[R, t] = hand_eye_p2l(P3D, zeros([m,n]), normc(inv(M_int_est) * normc([P2D;e])), 1e-9);
[R, t] = hand_eye_p2l(P3D, P2D, GT_Mint, 1e-9);


Sec111 = true; Sec112 = true; Sec113 = true; Sec121 = true; Sec122 = true; Sec123 = true; Sec131 = true; Sec132 = true; Sec133 = true; 
Sec211 = true; Sec212 = true; Sec213 = true; Sec221 = true; Sec222 = true; Sec223 = true; Sec231 = true; Sec232 = true; Sec233 = true; 
Sec311 = true; Sec312 = true; Sec313 = true; Sec321 = true; Sec322 = true; Sec323 = true; Sec331 = true; Sec332 = true; Sec333 = true; 


% 6 fiducials
nTest = 5000; %5000
TRE6 = zeros(1,nTest);
e = ones(1, 6);


for i=1:nTest
    P3= [];
    P2= [];
    
    if size(G311,2) > 0
        idx = randi(size(G311,2));
        P3 = [P3 G311(:, idx)];
        P2 = [P2 p311(:, idx)];
    else
        Sec311 = false;
    end

    if size(G313,2) > 0
        idx = randi(size(G313,2));
        P3 = [P3 G313(:, idx)];
        P2 = [P2 p313(:, idx)];
    else
        Sec313 = false;
    end

    if size(G332,2) > 0
        idx = randi(size(G332,2));
        P3 = [P3 G332(:, idx)];
        P2 = [P2 p332(:, idx)];
    else
        Sec332 = false;
    end
    
    if size(G212,2) > 0
        idx = randi(size(G212,2));
        P3 = [P3 G212(:, idx)];
        P2 = [P2 p212(:, idx)];
    else
        Sec212 = false;
    end
    
    if size(G231,2) > 0
        idx = randi(size(G231,2));
        P3 = [P3 G231(:, idx)];
        P2 = [P2 p231(:, idx)];
    else
        Sec231 = false;
    end
    
    if size(G232,2) > 0
        idx = randi(size(G232,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec232 = false;
    end
    
    % compute a calibration using all points
    % [M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2, P3 );
    %[R, t] = p2l(P3, zeros(size(P3)), normc(inv(GT_Mint)* normc([P2;e])), 1e-9);
    [R, t] = hand_eye_p2l(P3, P2, GT_Mint, 1e-9);
    M_ext_est = eye(4); M_ext_est(1:3,1:3) = R; M_ext_est(1:3,4) = t;
    TRE6(1,i) = calProjErr(GT_Mint*M_ext_est(1:3,:), P2D,P3D);
end
mean6 = mean(TRE6)

TRE7 = zeros(1,nTest);
e = ones(1, 7);
for i=1:nTest
    P3=[];
    P2= [];

    if size(G311,2) > 0
        idx = randi(size(G311,2));
        P3 = [P3 G311(:, idx)];
        P2 = [P2 p311(:, idx)];
    else
        Sec311 = false;
    end

    if size(G313,2) > 0
        idx = randi(size(G313,2));
        P3 = [P3 G313(:, idx)];
        P2 = [P2 p313(:, idx)];
    else
        Sec313 = false;
    end
    
    if size(G332,2) > 0
        idx = randi(size(G332,2));
        P3 = [P3 G332(:, idx)];
        P2 = [P2 p332(:, idx)];
    else
        Sec332 = false;
    end
    
    %new
    if size(G211,2) > 0
        idx = randi(size(G211,2));
        P3 = [P3 G211(:, idx)];
        P2 = [P2 p211(:, idx)];
    else
        Sec211 = false;
    end    
    
    %new
    if size(G213,2) > 0
        idx = randi(size(G213,2));
        P3 = [P3 G213(:, idx)];
        P2 = [P2 p213(:, idx)];
    else
        Sec213 = false;
    end
    
    if size(G231,2) > 0
        idx = randi(size(G231,2));
        P3 = [P3 G231(:, idx)];
        P2 = [P2 p231(:, idx)];
    else
        Sec231 = false;
    end
    
    %new
    if size(G233,2) > 0
        idx = randi(size(G233,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec233 = false;
    end
    
    % compute a calibration using all points
    % [M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2, P3 );
    %[R, t] = p2l(P3, zeros(size(P3)), normc(inv(GT_Mint)* normc([P2;e])), 1e-9);
    [R, t] = hand_eye_p2l(P3, P2, GT_Mint, 1e-9);
    M_ext_est = eye(4); M_ext_est(1:3,1:3) = R; M_ext_est(1:3,4) = t;
    TRE7(1,i) = calProjErr(GT_Mint*M_ext_est(1:3,:), P2D, P3D);
end
mean7 = mean(TRE7)

TRE8 = zeros(1,nTest);
e = ones(1, 8);
for i=1:nTest
    P3=[];
    P2= [];

     if size(G311,2) > 0
        idx = randi(size(G311,2));
        P3 = [P3 G311(:, idx)];
        P2 = [P2 p311(:, idx)];
    else
        Sec311 = false;
    end
    
    %new
    if size(G312,2) > 0
        idx = randi(size(G312,2));
        P3 = [P3 G312(:, idx)];
        P2 = [P2 p312(:, idx)];
    else
        Sec312 = false;
    end        
    
    if size(G313,2) > 0
        idx = randi(size(G313,2));
        P3 = [P3 G313(:, idx)];
        P2 = [P2 p313(:, idx)];
    else
        Sec313 = false;
    end

    if size(G332,2) > 0
        idx = randi(size(G332,2));
        P3 = [P3 G332(:, idx)];
        P2 = [P2 p332(:, idx)];
    else
        Sec332 = false;
    end
    
    if size(G211,2) > 0
        idx = randi(size(G211,2));
        P3 = [P3 G211(:, idx)];
        P2 = [P2 p211(:, idx)];
    else
        Sec211 = false;
    end    
    
    if size(G213,2) > 0
        idx = randi(size(G213,2));
        P3 = [P3 G213(:, idx)];
        P2 = [P2 p213(:, idx)];
    else
        Sec213 = false;
    end

    if size(G231,2) > 0
        idx = randi(size(G231,2));
        P3 = [P3 G231(:, idx)];
        P2 = [P2 p231(:, idx)];
    else
        Sec231 = false;
    end
    
    if size(G233,2) > 0
        idx = randi(size(G233,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec233 = false;
    end
    
    % compute a calibration using all points
    % [M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2, P3 );
    %[R, t] = p2l(P3, zeros(size(P3)), normc(inv(GT_Mint)* normc([P2;e])), 1e-9);
    [R, t] = hand_eye_p2l(P3, P2, GT_Mint, 1e-9);
    M_ext_est = eye(4); M_ext_est(1:3,1:3) = R; M_ext_est(1:3,4) = t;
    TRE8(1,i) = calProjErr(GT_Mint*M_ext_est(1:3,:), P2D, P3D);
end
mean8 = mean(TRE8)

TRE9 = zeros(1,nTest);
e = ones(1,9);
for i=1:nTest
    P3=[];
    P2= [];

   if size(G311,2) > 0
        idx = randi(size(G311,2));
        P3 = [P3 G311(:, idx)];
        P2 = [P2 p311(:, idx)];
    else
        Sec311 = false;
    end
    
    if size(G312,2) > 0
        idx = randi(size(G312,2));
        P3 = [P3 G312(:, idx)];
        P2 = [P2 p312(:, idx)];
    else
        Sec312 = false;
    end        
    
    if size(G313,2) > 0
        idx = randi(size(G313,2));
        P3 = [P3 G313(:, idx)];
        P2 = [P2 p313(:, idx)];
    else
        Sec313 = false;
    end

    if size(G332,2) > 0
        idx = randi(size(G332,2));
        P3 = [P3 G332(:, idx)];
        P2 = [P2 p332(:, idx)];
    else
        Sec332 = false;
    end
    
    if size(G211,2) > 0
        idx = randi(size(G211,2));
        P3 = [P3 G211(:, idx)];
        P2 = [P2 p211(:, idx)];
    else
        Sec211 = false;
    end    
    
    if size(G213,2) > 0
        idx = randi(size(G213,2));
        P3 = [P3 G213(:, idx)];
        P2 = [P2 p213(:, idx)];
    else
        Sec213 = false;
    end

    if size(G231,2) > 0
        idx = randi(size(G231,2));
        P3 = [P3 G231(:, idx)];
        P2 = [P2 p231(:, idx)];
    else
        Sec231 = false;
    end
    
    % new
    if size(G232,2) > 0
        idx = randi(size(G232,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec232 = false;
    end
    
    if size(G233,2) > 0
        idx = randi(size(G233,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec233 = false;
    end
    
    % compute a calibration using all points
    % [M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2, P3 );
    %[R, t] = p2l(P3, zeros(size(P3)), normc(inv(GT_Mint)* normc([P2;e])), 1e-9);
    [R, t] = hand_eye_p2l(P3, P2, GT_Mint, 1e-9);
    M_ext_est = eye(4); M_ext_est(1:3,1:3) = R; M_ext_est(1:3,4) = t;
    TRE9(1,i) = calProjErr(GT_Mint*M_ext_est(1:3,:), P2D, P3D);
end
mean9 = mean(TRE9)

TRE10 = zeros(1,nTest);
e = ones(1,10);
for i=1:nTest
    P3=[];
    P2= [];

    if size(G311,2) > 0
        idx = randi(size(G311,2));
        P3 = [P3 G311(:, idx)];
        P2 = [P2 p311(:, idx)];
    else
        Sec311 = false;
    end
    
    if size(G312,2) > 0
        idx = randi(size(G312,2));
        P3 = [P3 G312(:, idx)];
        P2 = [P2 p312(:, idx)];
    else
        Sec312 = false;
    end        
    
    if size(G313,2) > 0
        idx = randi(size(G313,2));
        P3 = [P3 G313(:, idx)];
        P2 = [P2 p313(:, idx)];
    else
        Sec313 = false;
    end

    if size(G332,2) > 0
        idx = randi(size(G332,2));
        P3 = [P3 G332(:, idx)];
        P2 = [P2 p332(:, idx)];
    else
        Sec332 = false;
    end
    
    if size(G211,2) > 0
        idx = randi(size(G211,2));
        P3 = [P3 G211(:, idx)];
        P2 = [P2 p211(:, idx)];
    else
        Sec211 = false;
    end;
    
    % new
    if size(G212,2) > 0
        idx = randi(size(G212,2));
        P3 = [P3 G212(:, idx)];
        P2 = [P2 p212(:, idx)];
    else
        Sec212 = false;
    end;        
    
    if size(G213,2) > 0
        idx = randi(size(G213,2));
        P3 = [P3 G213(:, idx)];
        P2 = [P2 p213(:, idx)];
    else
        Sec213 = false;
    end

    if size(G231,2) > 0
        idx = randi(size(G231,2));
        P3 = [P3 G231(:, idx)];
        P2 = [P2 p231(:, idx)];
    else
        Sec231 = false;
    end
    
    % new
    if size(G232,2) > 0
        idx = randi(size(G232,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec232 = false;
    end
    
    if size(G233,2) > 0
        idx = randi(size(G233,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec233 = false;
    end
    
    % compute a calibration using all points
    % [M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2, P3 );
    %[R, t] = p2l(P3, zeros(size(P3)), normc(inv(GT_Mint)* normc([P2;e])), 1e-9);
    [R, t] = hand_eye_p2l(P3, P2, GT_Mint, 1e-9);
    M_ext_est = eye(4); M_ext_est(1:3,1:3) = R; M_ext_est(1:3,4) = t;
    TRE10(1,i) = calProjErr(GT_Mint*M_ext_est(1:3,:), P2D, P3D);
end
mean10 = mean(TRE10)

TRE11 = zeros(1,nTest);
e = ones(1,11);
for i=1:nTest
    P3=[];
    P2= [];

    if size(G311,2) > 0
        idx = randi(size(G311,2));
        P3 = [P3 G311(:, idx)];
        P2 = [P2 p311(:, idx)];
    else
        Sec311 = false;
    end
    
    if size(G312,2) > 0
        idx = randi(size(G312,2));
        P3 = [P3 G312(:, idx)];
        P2 = [P2 p312(:, idx)];
    else
        Sec312 = false;
    end        
    
    if size(G313,2) > 0
        idx = randi(size(G313,2));
        P3 = [P3 G313(:, idx)];
        P2 = [P2 p313(:, idx)];
    else
        Sec313 = false;
    end

    if size(G332,2) > 0
        idx = randi(size(G332,2));
        P3 = [P3 G332(:, idx)];
        P2 = [P2 p332(:, idx)];
    else
        Sec332 = false;
    end
    
    if size(G211,2) > 0
        idx = randi(size(G211,2));
        P3 = [P3 G211(:, idx)];
        P2 = [P2 p211(:, idx)];
    else
        Sec211 = false;
    end;
    
    if size(G212,2) > 0
        idx = randi(size(G212,2));
        P3 = [P3 G212(:, idx)];
        P2 = [P2 p212(:, idx)];
    else
        Sec212 = false;
    end;        
    
    if size(G213,2) > 0
        idx = randi(size(G213,2));
        P3 = [P3 G213(:, idx)];
        P2 = [P2 p213(:, idx)];
    else
        Sec213 = false;
    end
    
    % new
    if size(G221,2) > 0
        idx = randi(size(G221,2));
        P3 = [P3 G221(:, idx)];
        P2 = [P2 p221(:, idx)];
    else
        Sec221 = false;
    end        
    
    if size(G231,2) > 0
        idx = randi(size(G231,2));
        P3 = [P3 G231(:, idx)];
        P2 = [P2 p231(:, idx)];
    else
        Sec231 = false;
    end
    
    if size(G232,2) > 0
        idx = randi(size(G232,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec232 = false;
    end
    
    if size(G233,2) > 0
        idx = randi(size(G233,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec233 = false;
    end
    
    % compute a calibration using all points
    % [M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2, P3 );
    %[R, t] = p2l(P3, zeros(size(P3)), normc(inv(GT_Mint)* normc([P2;e])), 1e-9);
    [R, t] = hand_eye_p2l(P3, P2, GT_Mint, 1e-9);
    M_ext_est = eye(4); M_ext_est(1:3,1:3) = R; M_ext_est(1:3,4) = t;
    TRE11(1,i) = calProjErr(GT_Mint*M_ext_est(1:3,:), P2D, P3D);
end
mean11 = mean(TRE11)

TRE12 = zeros(1,nTest);
e = ones(1,12);
for i=1:nTest
    P3=[];
    P2= [];

    if size(G311,2) > 0
        idx = randi(size(G311,2));
        P3 = [P3 G311(:, idx)];
        P2 = [P2 p311(:, idx)];
    else
        Sec311 = false;
    end
    
    if size(G312,2) > 0
        idx = randi(size(G312,2));
        P3 = [P3 G312(:, idx)];
        P2 = [P2 p312(:, idx)];
    else
        Sec312 = false;
    end        
    
    if size(G313,2) > 0
        idx = randi(size(G313,2));
        P3 = [P3 G313(:, idx)];
        P2 = [P2 p313(:, idx)];
    else
        Sec313 = false;
    end

    if size(G332,2) > 0
        idx = randi(size(G332,2));
        P3 = [P3 G332(:, idx)];
        P2 = [P2 p332(:, idx)];
    else
        Sec332 = false;
    end
    
    if size(G211,2) > 0
        idx = randi(size(G211,2));
        P3 = [P3 G211(:, idx)];
        P2 = [P2 p211(:, idx)];
    else
        Sec211 = false;
    end;
    
    if size(G212,2) > 0
        idx = randi(size(G212,2));
        P3 = [P3 G212(:, idx)];
        P2 = [P2 p212(:, idx)];
    else
        Sec212 = false;
    end;        
    
    if size(G213,2) > 0
        idx = randi(size(G213,2));
        P3 = [P3 G213(:, idx)];
        P2 = [P2 p213(:, idx)];
    else
        Sec213 = false;
    end
    
    if size(G221,2) > 0
        idx = randi(size(G221,2));
        P3 = [P3 G221(:, idx)];
        P2 = [P2 p221(:, idx)];
    else
        Sec221 = false;
    end
    
    % new
    if size(G223,2) > 0
        idx = randi(size(G223,2));
        P3 = [P3 G223(:, idx)];
        P2 = [P2 p223(:, idx)];
    else
        Sec223 = false;
    end    
    
    if size(G231,2) > 0
        idx = randi(size(G231,2));
        P3 = [P3 G231(:, idx)];
        P2 = [P2 p231(:, idx)];
    else
        Sec231 = false;
    end
    
    if size(G232,2) > 0
        idx = randi(size(G232,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec232 = false;
    end
    
    if size(G233,2) > 0
        idx = randi(size(G233,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec233 = false;
    end
    
    
    % compute a calibration using all points
    % [M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2, P3 );
    %[R, t] = p2l(P3, zeros(size(P3)), normc(inv(GT_Mint)* normc([P2;e])), 1e-9);
    [R, t] = hand_eye_p2l(P3, P2, GT_Mint, 1e-9);
    M_ext_est = eye(4); M_ext_est(1:3,1:3) = R; M_ext_est(1:3,4) = t;
    TRE12(1,i) = calProjErr(GT_Mint*M_ext_est(1:3,:), P2D, P3D);
end
mean12 = mean(TRE12)

TRE13 = zeros(1,nTest);
e = ones(1,13);
for i=1:nTest
    P3=[];
    P2= [];

    if size(G311,2) > 0
        idx = randi(size(G311,2));
        P3 = [P3 G311(:, idx)];
        P2 = [P2 p311(:, idx)];
    else
        Sec311 = false;
    end
    
    if size(G312,2) > 0
        idx = randi(size(G312,2));
        P3 = [P3 G312(:, idx)];
        P2 = [P2 p312(:, idx)];
    else
        Sec312 = false;
    end        
    
    if size(G313,2) > 0
        idx = randi(size(G313,2));
        P3 = [P3 G313(:, idx)];
        P2 = [P2 p313(:, idx)];
    else
        Sec313 = false;
    end

    if size(G332,2) > 0
        idx = randi(size(G332,2));
        P3 = [P3 G332(:, idx)];
        P2 = [P2 p332(:, idx)];
    else
        Sec332 = false;
    end
    
    if size(G211,2) > 0
        idx = randi(size(G211,2));
        P3 = [P3 G211(:, idx)];
        P2 = [P2 p211(:, idx)];
    else
        Sec211 = false;
    end;
    
    if size(G212,2) > 0
        idx = randi(size(G212,2));
        P3 = [P3 G212(:, idx)];
        P2 = [P2 p212(:, idx)];
    else
        Sec212 = false;
    end;        
    
    if size(G213,2) > 0
        idx = randi(size(G213,2));
        P3 = [P3 G213(:, idx)];
        P2 = [P2 p213(:, idx)];
    else
        Sec213 = false;
    end
    
    if size(G221,2) > 0
        idx = randi(size(G221,2));
        P3 = [P3 G221(:, idx)];
        P2 = [P2 p221(:, idx)];
    else
        Sec221 = false;
    end
    
    if size(G223,2) > 0
        idx = randi(size(G223,2));
        P3 = [P3 G223(:, idx)];
        P2 = [P2 p223(:, idx)];
    else
        Sec223 = false;
    end    
    
    if size(G231,2) > 0
        idx = randi(size(G231,2));
        P3 = [P3 G231(:, idx)];
        P2 = [P2 p231(:, idx)];
    else
        Sec231 = false;
    end
    
    if size(G232,2) > 0
        idx = randi(size(G232,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec232 = false;
    end
    
    if size(G233,2) > 0
        idx = randi(size(G233,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec233 = false;
    end
    
    %new
    if size(G333,2) > 0
        idx = randi(size(G333,2));
        P3 = [P3 G333(:, idx)];
        P2 = [P2 p333(:, idx)];
    else
        Sec333 = false;
    end    
    
    % compute a calibration using all points
    % [M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2, P3 );
    %[R, t] = p2l(P3, zeros(size(P3)), normc(inv(GT_Mint)* normc([P2;e])), 1e-9);
    [R, t] = hand_eye_p2l(P3, P2, GT_Mint, 1e-9);
    M_ext_est = eye(4); M_ext_est(1:3,1:3) = R; M_ext_est(1:3,4) = t;
    TRE13(1,i) = calProjErr(GT_Mint*M_ext_est(1:3,:), P2D, P3D);
end
mean13 = mean(TRE13)

TRE14 = zeros(1,nTest);
e = ones(1,14);
for i=1:nTest
    P3=[];
    P2= [];

    if size(G311,2) > 0
        idx = randi(size(G311,2));
        P3 = [P3 G311(:, idx)];
        P2 = [P2 p311(:, idx)];
    else
        Sec311 = false;
    end
    
    if size(G312,2) > 0
        idx = randi(size(G312,2));
        P3 = [P3 G312(:, idx)];
        P2 = [P2 p312(:, idx)];
    else
        Sec312 = false;
    end        
    
    if size(G313,2) > 0
        idx = randi(size(G313,2));
        P3 = [P3 G313(:, idx)];
        P2 = [P2 p313(:, idx)];
    else
        Sec313 = false;
    end

    if size(G332,2) > 0
        idx = randi(size(G332,2));
        P3 = [P3 G332(:, idx)];
        P2 = [P2 p332(:, idx)];
    else
        Sec332 = false;
    end
    
    if size(G211,2) > 0
        idx = randi(size(G211,2));
        P3 = [P3 G211(:, idx)];
        P2 = [P2 p211(:, idx)];
    else
        Sec211 = false;
    end;
    
    if size(G212,2) > 0
        idx = randi(size(G212,2));
        P3 = [P3 G212(:, idx)];
        P2 = [P2 p212(:, idx)];
    else
        Sec212 = false;
    end;        
    
    if size(G213,2) > 0
        idx = randi(size(G213,2));
        P3 = [P3 G213(:, idx)];
        P2 = [P2 p213(:, idx)];
    else
        Sec213 = false;
    end
    
    if size(G221,2) > 0
        idx = randi(size(G221,2));
        P3 = [P3 G221(:, idx)];
        P2 = [P2 p221(:, idx)];
    else
        Sec221 = false;
    end
    
    if size(G223,2) > 0
        idx = randi(size(G223,2));
        P3 = [P3 G223(:, idx)];
        P2 = [P2 p223(:, idx)];
    else
        Sec223 = false;
    end    
    
    if size(G231,2) > 0
        idx = randi(size(G231,2));
        P3 = [P3 G231(:, idx)];
        P2 = [P2 p231(:, idx)];
    else
        Sec231 = false;
    end
    
    if size(G232,2) > 0
        idx = randi(size(G232,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec232 = false;
    end
    
    if size(G233,2) > 0
        idx = randi(size(G233,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec233 = false;
    end
    
    if size(G333,2) > 0
        idx = randi(size(G333,2));
        P3 = [P3 G333(:, idx)];
        P2 = [P2 p333(:, idx)];
    else
        Sec333 = false;
    end
    
    %new
    if size(G331,2) > 0
        idx = randi(size(G331,2));
        P3 = [P3 G331(:, idx)];
        P2 = [P2 p331(:, idx)];
    else
        Sec331 = false;
    end        
    
    % compute a calibration using all points
    % [M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2, P3 );
    %[R, t] = p2l(P3, zeros(size(P3)), normc(inv(GT_Mint)* normc([P2;e])), 1e-9);
    [R, t] = hand_eye_p2l(P3, P2, GT_Mint, 1e-9);
    M_ext_est = eye(4); M_ext_est(1:3,1:3) = R; M_ext_est(1:3,4) = t;
    TRE14(1,i) = calProjErr(GT_Mint*M_ext_est(1:3,:), P2D, P3D);
end
mean14 = mean(TRE14)

TRE15 = zeros(1,nTest);
e = ones(1,15);
for i=1:nTest
    P3=[];
    P2= [];

    if size(G311,2) > 0
        idx = randi(size(G311,2));
        P3 = [P3 G311(:, idx)];
        P2 = [P2 p311(:, idx)];
    else
        Sec311 = false;
    end
    
    if size(G312,2) > 0
        idx = randi(size(G312,2));
        P3 = [P3 G312(:, idx)];
        P2 = [P2 p312(:, idx)];
    else
        Sec312 = false;
    end        
    
    if size(G313,2) > 0
        idx = randi(size(G313,2));
        P3 = [P3 G313(:, idx)];
        P2 = [P2 p313(:, idx)];
    else
        Sec313 = false;
    end

    if size(G332,2) > 0
        idx = randi(size(G332,2));
        P3 = [P3 G332(:, idx)];
        P2 = [P2 p332(:, idx)];
    else
        Sec332 = false;
    end
    
    if size(G211,2) > 0
        idx = randi(size(G211,2));
        P3 = [P3 G211(:, idx)];
        P2 = [P2 p211(:, idx)];
    else
        Sec211 = false;
    end;
    
    if size(G212,2) > 0
        idx = randi(size(G212,2));
        P3 = [P3 G212(:, idx)];
        P2 = [P2 p212(:, idx)];
    else
        Sec212 = false;
    end;        
    
    if size(G213,2) > 0
        idx = randi(size(G213,2));
        P3 = [P3 G213(:, idx)];
        P2 = [P2 p213(:, idx)];
    else
        Sec213 = false;
    end
    
    if size(G221,2) > 0
        idx = randi(size(G221,2));
        P3 = [P3 G221(:, idx)];
        P2 = [P2 p221(:, idx)];
    else
        Sec221 = false;
    end
    
    if size(G223,2) > 0
        idx = randi(size(G223,2));
        P3 = [P3 G223(:, idx)];
        P2 = [P2 p223(:, idx)];
    else
        Sec223 = false;
    end    
    
    if size(G231,2) > 0
        idx = randi(size(G231,2));
        P3 = [P3 G231(:, idx)];
        P2 = [P2 p231(:, idx)];
    else
        Sec231 = false;
    end
    
    if size(G232,2) > 0
        idx = randi(size(G232,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec232 = false;
    end
    
    if size(G233,2) > 0
        idx = randi(size(G233,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec233 = false;
    end
    
    if size(G333,2) > 0
        idx = randi(size(G333,2));
        P3 = [P3 G333(:, idx)];
        P2 = [P2 p333(:, idx)];
    else
        Sec333 = false;
    end
    
    if size(G331,2) > 0
        idx = randi(size(G331,2));
        P3 = [P3 G331(:, idx)];
        P2 = [P2 p331(:, idx)];
    else
        Sec331 = false;
    end
    
    %new
    if size(G323,2) > 0
        idx = randi(size(G323,2));
        P3 = [P3 G323(:, idx)];
        P2 = [P2 p323(:, idx)];
    else
        Sec323 = false;
    end    
    
    % compute a calibration using all points
    % [M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2, P3 );
    %[R, t] = p2l(P3, zeros(size(P3)), normc(inv(GT_Mint)* normc([P2;e])), 1e-9);
    [R, t] = hand_eye_p2l(P3, P2, GT_Mint, 1e-9);
    M_ext_est = eye(4); M_ext_est(1:3,1:3) = R; M_ext_est(1:3,4) = t;
    TRE15(1,i) = calProjErr(GT_Mint*M_ext_est(1:3,:), P2D, P3D);
end
mean15 = mean(TRE15)

TRE16 = zeros(1,nTest);
e = ones(1,16);
for i=1:nTest
    P3=[];
    P2= [];

    if size(G311,2) > 0
        idx = randi(size(G311,2));
        P3 = [P3 G311(:, idx)];
        P2 = [P2 p311(:, idx)];
    else
        Sec311 = false;
    end
    
    if size(G312,2) > 0
        idx = randi(size(G312,2));
        P3 = [P3 G312(:, idx)];
        P2 = [P2 p312(:, idx)];
    else
        Sec312 = false;
    end        
    
    if size(G313,2) > 0
        idx = randi(size(G313,2));
        P3 = [P3 G313(:, idx)];
        P2 = [P2 p313(:, idx)];
    else
        Sec313 = false;
    end

    if size(G332,2) > 0
        idx = randi(size(G332,2));
        P3 = [P3 G332(:, idx)];
        P2 = [P2 p332(:, idx)];
    else
        Sec332 = false;
    end
    
    if size(G211,2) > 0
        idx = randi(size(G211,2));
        P3 = [P3 G211(:, idx)];
        P2 = [P2 p211(:, idx)];
    else
        Sec211 = false;
    end;
    
    if size(G212,2) > 0
        idx = randi(size(G212,2));
        P3 = [P3 G212(:, idx)];
        P2 = [P2 p212(:, idx)];
    else
        Sec212 = false;
    end;        
    
    if size(G213,2) > 0
        idx = randi(size(G213,2));
        P3 = [P3 G213(:, idx)];
        P2 = [P2 p213(:, idx)];
    else
        Sec213 = false;
    end
    
    if size(G221,2) > 0
        idx = randi(size(G221,2));
        P3 = [P3 G221(:, idx)];
        P2 = [P2 p221(:, idx)];
    else
        Sec221 = false;
    end
    
    if size(G223,2) > 0
        idx = randi(size(G223,2));
        P3 = [P3 G223(:, idx)];
        P2 = [P2 p223(:, idx)];
    else
        Sec223 = false;
    end    
    
    if size(G231,2) > 0
        idx = randi(size(G231,2));
        P3 = [P3 G231(:, idx)];
        P2 = [P2 p231(:, idx)];
    else
        Sec231 = false;
    end
    
    if size(G232,2) > 0
        idx = randi(size(G232,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec232 = false;
    end
    
    if size(G233,2) > 0
        idx = randi(size(G233,2));
        P3 = [P3 G233(:, idx)];
        P2 = [P2 p233(:, idx)];
    else
        Sec233 = false;
    end
    
    if size(G333,2) > 0
        idx = randi(size(G333,2));
        P3 = [P3 G333(:, idx)];
        P2 = [P2 p333(:, idx)];
    else
        Sec333 = false;
    end
    
    if size(G331,2) > 0
        idx = randi(size(G331,2));
        P3 = [P3 G331(:, idx)];
        P2 = [P2 p331(:, idx)];
    else
        Sec331 = false;
    end
    
    if size(G323,2) > 0
        idx = randi(size(G323,2));
        P3 = [P3 G323(:, idx)];
        P2 = [P2 p323(:, idx)];
    else
        Sec323 = false;
    end
    
    if size(G321,2) > 0
        idx = randi(size(G321,2));
        P3 = [P3 G321(:, idx)];
        P2 = [P2 p321(:, idx)];
    else
        Sec321 = false;
    end  
    
    % compute a calibration using all points
    % [M_int_est, M_ext_est, M_proj, fre] = cameraCombinedCalibration2(P2, P3 );
    %[R, t] = p2l(P3, zeros(size(P3)), normc(inv(GT_Mint)* normc([P2;e])), 1e-9);
    [R, t] = hand_eye_p2l(P3, P2, GT_Mint, 1e-9);
    M_ext_est = eye(4); M_ext_est(1:3,1:3) = R; M_ext_est(1:3,4) = t;
    TRE16(1,i) = calProjErr(GT_Mint*M_ext_est(1:3,:), P2D, P3D);
end
mean16 = mean(TRE16)

%add eroor message if flase print out

if Sec111 == false
    disp('no data is section 111')
end
if Sec112 == false
    disp('no data is section 112')
end
if Sec113 == false
    disp('no data is section 113')
end
if Sec121 == false
    disp('no data is section 121')
end
if Sec122 == false
    disp('no data is section 122')
end
if Sec123 == false
    disp('no data is section 123')
end
if Sec131 == false
    disp('no data is section 131')
end
if Sec132 == false
    disp('no data is section 132')
end
if Sec133 == false
    disp('no data is section 133')
end
if Sec211 == false
    disp('no data is section 211')
end
if Sec212 == false
    disp('no data is section 212')
end
if Sec213 == false
    disp('no data is section 213')
end
if Sec211 == false
    disp('no data is section 221')
end
if Sec222 == false
    disp('no data is section 222')
end
if Sec223 == false
    disp('no data is section 223')
end
if Sec231 == false
    disp('no data is section 231')
end
if Sec232 == false
    disp('no data is section 232')
end
if Sec233 == false
    disp('no data is section 233')
end
if Sec311 == false
    disp('no data is section 311')
end
if Sec312 == false
    disp('no data is section 312')
end
if Sec313 == false
    disp('no data is section 313')
end
if Sec321 == false
    disp('no data is section 321')
end
if Sec322 == false
    disp('no data is section 322')
end
if Sec323 == false
    disp('no data is section 323')
end
if Sec331 == false
    disp('no data is section 331')
end
if Sec332 == false
    disp('no data is section 332')
end
if Sec333 == false
    disp('no data is section 333')
end




TRE = [TRE6' TRE7' TRE8' TRE9' TRE10' TRE11' TRE12' TRE13' TRE14' TRE15' TRE16'];

% Replace with location of the downloaded folder
%pth = 'E:\chene\OneDrive - The University of Western Ontario\src\matlab\boxplot2\';
%addpath(fullfile(pth, 'boxplot2')); 
%addpath(fullfile(pth, 'minmax')); 
%boxplot2(TRE)
boxplot(TRE)
fn = 42;
ylabel('mean projective TRE (pixel)')
xlabel('number of registration fiducial points')
xticks(1:11)
xticklabels({'6', '7', '8', '9', '10', '11', '12' '13' '14' '15' '16'})

