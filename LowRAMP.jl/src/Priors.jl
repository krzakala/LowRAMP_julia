#= List of priors *********************************************** =#

#= Gaussian with var_gauss variance =#
function f_gauss(A,B,var_gauss=1)
   VAR=inv((1./var_gauss)*eye(A)+A);
   MEAN=B*VAR;
   logZ=-0.5*log(det(var_gauss*eye(A)+A))*size(B,1)+trace(0.5*B'*B*VAR);   
   MEAN,VAR,logZ;
end

#= Clustering Prior =#
function f_clust(A,B)
    RANK=size(A,1);n=size(B,1);
    AA=repmat(diag(A),1,n);
    Prob=-0.5*AA+B';
    KeepMax=maximum(Prob,1);
    Prob=exp(Prob-repmat(KeepMax,RANK,1));
    Norm=repmat(sum(Prob,1),RANK,1);
    Tokeep=sum(log(sum(Prob/RANK,1)));
    Prob=Prob./Norm;
    MEAN=Prob';
    VAR=diagm(vec(sum(MEAN,1)/n))-Prob*Prob'/n;
    logZ=sum(KeepMax)+Tokeep;
    MEAN,VAR,logZ;
end

#= Rank 1 1/0 prior, rho is the fraction of 1 =#
function f_Rank1Binary(A,B,rho=0.5)
    Weight=-0.5*A[1]+B;
    pos=find(Weight.>0);
    neg=setdiff([1:size(B,1);],pos);
    MEAN=zeros(size(B));
    MEAN[neg]=rho*exp(-0.5*A[1]+B[neg])./(1-rho+rho*exp(-0.5*A[1]+B[neg]));
    MEAN[pos]= rho./(rho+(1-rho)*exp(0.5*A[1]-B[pos]));
    VAR=mean(MEAN.*(1-MEAN));
    logZ=sum(log(1-rho+rho*exp(-0.5*A[1]+B[neg])));   
    logZ=logZ+sum(-0.5*A[1]+B[pos]+log(rho+(1-rho)*exp(0.5*A[1]-B[pos])));
    MEAN,VAR,logZ;
end

