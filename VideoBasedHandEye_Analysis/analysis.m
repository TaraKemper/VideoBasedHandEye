clear all;
%load('calibData.mat');

% add correct filepath for your computer to the CircleCentersOutput.txt
% and StylusTipCoordsOutput.txt files for you desired dataset which are 
% output to the OutputImages folder upon running the module
P2D = dlmread('\\imagingsrv.robarts.ca\Peters_Users$\tkemper\Documents\Hand Eye Calibration\Slicer Module\Test Data\27 Regions Data\Test 5\CircleCentersOutput.txt'); 
P3D = dlmread('\\imagingsrv.robarts.ca\Peters_Users$\tkemper\Documents\Hand Eye Calibration\Slicer Module\Test Data\27 Regions Data\Test 5\StylusTipCoordsOutput.txt');

% replace with output hand eye matrix for the desired dataset which will
% print in slicer's python interactor window upon running the module
GT_HE = [0.0661233505110463	-0.995315679492888	0.0705294312525108	1.19174967610230
0.941976674699558	0.0389538756202925	-0.333410467586311	-51.9636618100234
0.329101271403130	0.0884832963319526	0.940139914816446	7.89678881768355
0	0	0	1];

% replace with output intrinsic matrix for the desired dataset which will
% print in slicer's python interactor window upon running the module
GT_Mint = [629.40692139, 0, 322.69825503; 0, 626.10284424, 239.43214333; 0, 0, 1];

% image size, change according to the rezolution of your camera
sizex = 640;
sizey = 480;

% We assume that in the 'calibData.mat', we have a set of 2D/3D points
[m,n] = size(P3D);
e = ones(1,n);
P3D_transformed = GT_HE * [P3D;e]; 
P3D_transformed = P3D_transformed(1:3,:);

% divide the frustum into a 3x3x3 grid and assign points to them
minZ = min(P3D_transformed(3,:));
maxZ = max(P3D_transformed(3,:));

% x, and y are image size, z is the physical distance from camera
secX = (sizex)/3;
secY = (sizey)/3;
secZ = (maxZ-minZ)/3;

%create the sections
boolx(1,:) = P2D(1,:) < secX;
boolx(2,:) = P2D(1,:) >= secX & P2D(1,:) < 2*secX;
boolx(3,:) = P2D(1,:) >= 2*secX;
booly(1,:) = P2D(2,:) < secY;
booly(2,:) = P2D(2,:) >= secY & P2D(2,:) < 2*secY;
booly(3,:) = P2D(2,:) >= 2*secY;
boolz(1,:) = P3D_transformed(3,:) < minZ + secZ;
boolz(2,:) = P3D_transformed(3,:) >= minZ + secZ & P3D_transformed(3,:) < minZ + 2*secZ;
boolz(3,:) = P3D_transformed(3,:) >= minZ + 2*secZ;

G111 = []; G112 = []; G113 = []; G121 = []; G122 = []; G123 = []; G131 = []; G132 = []; G133 = []; 
G211 = []; G212 = []; G213 = []; G221 = []; G222 = []; G223 = []; G231 = []; G232 = []; G233 = []; 
G311 = []; G312 = []; G313 = []; G321 = []; G322 = []; G323 = []; G331 = []; G332 = []; G333 = []; 

p111 = []; p112 = []; p113 = []; p121 = []; p122 = []; p123 = []; p131 = []; p132 = []; p133 = []; 
p211 = []; p212 = []; p213 = []; p221 = []; p222 = []; p223 = []; p231 = []; p232 = []; p233 = []; 
p311 = []; p312 = []; p313 = []; p321 = []; p322 = []; p323 = []; p331 = []; p332 = []; p333 = []; 


%loop through and save data to each appropriate section
for i = 1:n
    if boolx(1,i) & booly(1,i) & boolz(1,i)
        G111 = [ G111 P3D(:,i) ];
        p111 = [ p111 P2D(:,i) ];
    end
    if boolx(2,i) & booly(1,i) & boolz(1,i)
        G121 = [ G121 P3D(:,i) ];
        p121 = [ p121 P2D(:,i) ];
    end
    if boolx(3,i) & booly(1,i) & boolz(1,i)
        G131 = [ G131 P3D(:,i) ];
        p131 = [ p131 P2D(:,i) ];
    end
    
    if boolx(1,i) & booly(2,i) & boolz(1,i)
        G112 = [ G112 P3D(:,i) ];
        p112 = [ p112 P2D(:,i) ];
    end
    if boolx(2,i) & booly(2,i) & boolz(1,i)
        G122 = [ G122 P3D(:,i) ];
        p122 = [ p122 P2D(:,i) ];
    end
    if boolx(3,i) & booly(2,i) & boolz(1,i)
        G132 = [ G132 P3D(:,i) ];
        p132 = [ p132 P2D(:,i) ];
    end
    
    if boolx(1,i) & booly(3,i) & boolz(1,i)
        G113 = [ G113 P3D(:,i) ];
        p113 = [ p113 P2D(:,i) ];
    end
    if boolx(2,i) & booly(3,i) & boolz(1,i)
        G123 = [ G123 P3D(:,i) ];
        p123 = [ p123 P2D(:,i) ];
    end
    if boolx(3,i) & booly(3,i) & boolz(1,i)
        G133 = [ G133 P3D(:,i) ];
        p133 = [ p133 P2D(:,i) ];
    end
    
    %------------------------------------------------
    if boolx(1,i) & booly(1,i) & boolz(2,i)
        G211 = [ G211 P3D(:,i) ];
        p211 = [ p211 P2D(:,i) ];
    end
    if boolx(2,i) & booly(1,i) & boolz(2,i)
        G221 = [ G221 P3D(:,i) ];
        p221 = [ p221 P2D(:,i) ];
    end
    if boolx(3,i) & booly(1,i) & boolz(2,i)
        G231 = [ G231 P3D(:,i) ];
        p231 = [ p231 P2D(:,i) ];
    end    
    if boolx(1,i) & booly(2,i) & boolz(2,i)
        G212 = [ G212 P3D(:,i) ];
        p212 = [ p212 P2D(:,i) ];
    end
    if boolx(2,i) & booly(2,i) & boolz(2,i)
        G222 = [ G222 P3D(:,i) ];
        p222 = [ p222 P2D(:,i) ];
    end
    if boolx(3,i) & booly(2,i) & boolz(2,i)
        G232 = [ G232 P3D(:,i) ];
        p232 = [ p232 P2D(:,i) ];
    end    
    if boolx(1,i) & booly(3,i) & boolz(2,i)
        G213 = [ G213 P3D(:,i) ];
        p213 = [ p213 P2D(:,i) ];
    end
    if boolx(2,i) & booly(3,i) & boolz(2,i)
        G223 = [ G223 P3D(:,i) ];
        p223 = [ p223 P2D(:,i) ];
    end
    if boolx(3,i) & booly(3,i) & boolz(2,i)
        G233 = [ G233 P3D(:,i) ];
        p233 = [ p233 P2D(:,i) ];
    end
    
    %------------------------------------------------
    if boolx(1,i) & booly(1,i) & boolz(3,i)
        G311 = [ G311 P3D(:,i) ];
        p311 = [ p311 P2D(:,i) ];
    end
    if boolx(2,i) & booly(1,i) & boolz(3,i)
        G321 = [ G321 P3D(:,i) ];
        p321 = [ p321 P2D(:,i) ];
    end
    if boolx(3,i) & booly(1,i) & boolz(3,i)
        G331 = [ G331 P3D(:,i) ];
        p331 = [ p331 P2D(:,i) ];
    end    
    if boolx(1,i) & booly(2,i) & boolz(3,i)
        G312 = [ G312 P3D(:,i) ];
        p312 = [ p312 P2D(:,i) ];
    end
    if boolx(2,i) & booly(2,i) & boolz(3,i)
        G322 = [ G322 P3D(:,i) ];
        p322 = [ p322 P2D(:,i) ];
    end
    if boolx(3,i) & booly(2,i) & boolz(3,i)
        G332 = [ G332 P3D(:,i) ];
        p332 = [ p332 P2D(:,i) ];
    end    
    if boolx(1,i) & booly(3,i) & boolz(3,i)
        G313 = [ G313 P3D(:,i) ];
        p313 = [ p313 P2D(:,i) ];
    end
    if boolx(2,i) & booly(3,i) & boolz(3,i)
        G323 = [ G323 P3D(:,i) ];
        p323 = [ p323 P2D(:,i) ];
    end
    if boolx(3,i) & booly(3,i) & boolz(3,i)
        G333 = [ G333 P3D(:,i) ];
        p333 = [ p333 P2D(:,i) ];
    end
    
    %------------------------------------------------
end

%print out final size
size(G111) + size(G112) + size(G113) + ...
    size(G121) + size(G122) + size(G123) + ...
    size(G131) + size(G132) + size(G133) + ...
    size(G211) + size(G212) + size(G213) + ...
    size(G221) + size(G222) + size(G223) + ...
    size(G231) + size(G232) + size(G233) + ...
    size(G311) + size(G312) + size(G313) + ...
    size(G321) + size(G322) + size(G323) + ...
    size(G331) + size(G332) + size(G333)  