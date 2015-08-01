function [eigenR, dataeigenR, sim] = PPC_one_culture(Y, samples, howmanysamples)

warning('off', 'optim:fminunc:SwitchingMethod');
warning('off', 'optim:fminunc:SwitchingMethod');
gY  = find(isnan(Y));
nri = size(Y,1);
nrk = size(Y,2);
nms = fieldnames(samples);

samples = structfun(@(x)x',samples, 'uni', 0);
nrofchains = size(samples.g_1,1);
origiter   = size(samples.g_1,2);
origpostss = nrofchains*origiter;

if origpostss > howmanysamples
    samples    = structfun(@(x)x(:,ceil(linspace(1,origiter,ceil(howmanysamples/nrofchains))),:),samples,'uni',0);
    nrofiter   = size(samples.g_1,1);
    postss     = nrofchains*nrofiter;
else
    postss = origpostss;
end

samples = structfun(@(x)x(:),samples, 'uni', 0); % make a vector from chains*iterations

for ctr = 1:howmanysamples
for i = 1 : nri
    for k = 1 : nrk
        theta =  samples.(sprintf('theta_%i', i))(ctr);
        b =  samples.(sprintf('b_%i', i))(ctr);
        g =  samples.(sprintf('g_%i', i))(ctr);
        delta =  samples.(sprintf('delta_%i', k))(ctr);
        Z =  samples.(sprintf('Z_%i', k))(ctr);
        sim.D(i,k, ctr) = invlogitT(theta-delta);
        p(1,i,k) = b*g + sim.D(i,k, ctr)*Z - sim.D(i,k, ctr)*b*g;
        p(2,i,k) = sim.D(i,k, ctr)-sim.D(i,k, ctr)*Z + b*(sim.D(i,k, ctr)-1)*(g-1);
        p(3,i,k) = 1-(p(1,i,k)+p(2,i,k));
        sim.PPY(i,k,ctr)   = mnrnd(1,p(:,i,k))*[1;2;3];
    end
end
    eigenR(ctr,:) = eigenratio3(sim.PPY(:,:,ctr))';
   
end
 dataeigenR = eigenratio3(Y);

close all
figure(1)
for v = 1:postss
    plot(eigenR(v,:),'color',[0.5 0.5 0.5])
    hold on
end
plot(dataeigenR,'k','LineWidth',2)
ylabel('value')
xlabel('eigenvalue')

function eigenR = eigenratio3(Y)

% Y(isnan(Y(:))) = 1;

% options = optimset('MaxFunEvals',5000,...
%     'Display','off',...
%     'Largescale','off',...
%     'Algorithm','active-set');
options = optimset('MaxFunEvals',15000,...
    'MaxIter',10000,...
    'TolFun',eps,...
    'TolX',eps,...
    'Largescale','off');
corrM = sparsecorr(Y');

corrM(isnan(corrM))=0;
% h = find(all(isnan(corrM),2));
% if numel(h) > 0
%     newY = Y;
%     for g = 1: numel(h)
%         newY(h(g),1) = 0.5;
%     end
%     corrM = corr(newY');
% end

N     = length(corrM);

corrM1 = corrM;
a1    = fminunc(@(x)mrmlsloss(x,corrM),rand(N,1),  options);
ap1   = a1*a1';
repl1 = diag(ap1);
for i=1:N, corrM1(i,i)= repl1(i); end,
eigen1 = max(eig(corrM1));

corrM2 = corrM1-ap1;
a2     = fminunc(@(x)mrmlsloss(x,corrM2),rand(N,1), options);
ap2    = a2*a2';
repl2  = diag(ap2);
for i=1:N, corrM2(i,i)= repl2(i); end,
eigen2 = max(eig(corrM2));
if eigen2>eigen1
    eigen2 = eigen1;
    corrM2 = corrM1;
    disp('Warning: Bad eigen')
end

corrM3 = corrM2-ap2;
a3     = fminunc(@(x)mrmlsloss(x,corrM3),rand(N,1), options);
ap3    = a3*a3';
repl3  = diag(ap3);
for i=1:N, corrM3(i,i)= repl3(i); end,
eigen3 = max(eig(corrM3));
if eigen3>eigen1
    eigen3 = eigen2;
    corrM3 = corrM2;
    disp('Warning: Bad eigen')
end
corrM4 = corrM3-ap3;
a4     = fminunc(@(x)mrmlsloss(x,corrM4),rand(N,1), options);
ap4    = a4*a4';
repl4  = diag(ap4);
for i=1:N, corrM4(i,i)= repl4(i); end,
eigen4 = max(eig(corrM4));
if eigen4>eigen1
    eigen4 = eigen3;
    corrM4 = corrM3;
    disp('Warning:Bad eigen')
end
corrM5 = corrM4-ap4;
a5     = fminunc(@(x)mrmlsloss(x,corrM5),rand(N,1), options);
ap5    = a5*a5';
repl5  = diag(ap5);
for i=1:N, corrM5(i,i)= repl5(i); end,
eigen5 = max(eig(corrM5));
if eigen5>eigen1
    eigen5 = eigen4;
    corrM5 = corrM4;
    disp('Warning: Bad eigen')
end

corrM6 = corrM5-ap5;
a6     = fminunc(@(x)mrmlsloss(x,corrM6),rand(N,1), options);
ap6    = a6*a6';
repl6  = diag(ap6);
for i=1:N, corrM6(i,i)= repl6(i); end,
eigen6 = max(eig(corrM6));
if eigen6>eigen1
    eigen6 = eigen5;
    corrM6 = corrM5;
    disp('Warning: Bad eigen')
end

eigenR = [eigen1 eigen2 eigen3 eigen4 eigen5 eigen6];


function l = mrmlsloss(x,A)
%minimum residual method least squares

r = (x*x'-A).^2;
n = size(r,1);
r(1:n+1:n*n)=0;
l = sum(r(:));

function Rho = sparsecorr(X,nanval)
% SPARSECORR  Correlation matrix with pairwise exclusion of some value

if nargin>1
    X(X==nanval) = nan ;
end
X = X' ;

R = size( X ,1 ) ;
Rho = ones( R ) ;

for r1 = 1:R
    for r2 = (r1+1):R
        x = X([r1 r2],:) ;
        censor = isnan( x(1,:) ) | isnan( x(2,:) ) ;
        x(:,censor) = [] ;
        cr = corr( x' ) ;
        Rho(r1,r2) = cr(2) ;
        Rho(r2,r1) = cr(2) ;
    end
end


function invl = invlogitT(p)

invl = exp(p)./(1+exp(p));
