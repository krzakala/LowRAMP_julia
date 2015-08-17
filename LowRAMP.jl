#=
Author: Florent Krzakala
Date: 07-27-2015 -- 08-17-2015
Where: Santa Fe Institute, Praha, Plzen
Description: AMP for Low Rank Estimation LowRAMP for UV' decomposition
=#

module LowRAMP

export LowRAMP_UV,LowRAMP_UV_new,demo_LowRAMP_UV

function  demo_LowRAMP_UV(n,m,RANK,Delta)
    @printf("Creating a problem of rank %d \n", RANK);
    V = zeros(m,RANK);
    for i=1:m
        V[i,ceil(rand()*RANK)]=1;
    end
    U = randn(n,RANK);
    
    #Adding noise!
    Y=U*V'/sqrt(n)+sqrt(Delta)*randn(n,m);
    
    #Computing the score and the inverse Fischer information
    S=Y/Delta;Iinv=Delta;
    
    #Calling the code
    @printf("Running AMP \n");
    damp=0.5;
    init_sol=1;
    tic()
    u_ample,v_ample = LowRAMP_UV_new(S,Iinv,RANK,f_gauss,f_clust,damp,1e-6,100,U,V,init_sol);
    toc()
end

function LowRAMP_UV_new( S, Delta ,RANK, Fun_u=f_gauss,Fun_v=f_clust, damp=0.5,conv_criterion=1e-6,max_iter=100,u_truth=[],v_truth=[],init_sol=0)
    # Usage AMP Lowrank Estimation for UV decomposition
    # LowRAMP_UV( S , Delta, u_truth ,v_truth , RANK,damp,init_sol)    
    m,n=size(S);

    #Initialization
    u=zeros(m,RANK);
    v=zeros(n,RANK);
    if init_sol==0
        @printf("Zeros initial conditions \n");
    elseif init_sol==1 #Init in the solution
        @printf("Random Gaussian initial conditions \n");
        u=randn(m,RANK);
        v=randn(n,RANK);
    elseif init_sol==2
        @printf("Use SVD as an initial condition \n");
        U,SS,V = svds(S,RANK);
        u=U[:,1:RANK];
        v=V[:,1:RANK];
    elseif init_sol==3
        @printf("Use solution as an initial condition \n");
        u=u_truth+1e-4*randn(m,RANK);
        v=v_truth+1e-4*randn(n,RANK);
    elseif init_sol==4
        @printf("Use prior as an initial condition \n");
        u,u_var,log_u = Fun_u(eye(RANK,RANK),zeros(m,RANK));
        v,v_var,log_v = Fun_v(eye(RANK,RANK),zeros(n,RANK));
    elseif init_sol==5
        @printf("Use ones as an initial condition \n");
        u=ones(m,RANK)/m;
        v=ones(n,RANK)/n;    
    end

    u_old=zeros(m,RANK);
    v_old=zeros(n,RANK);
    u_var=zeros(RANK,RANK);
    v_var=zeros(RANK,RANK);

    u_new=zeros(m,RANK);   
    v_new=zeros(n,RANK);
    
    u_V=ones(RANK,RANK);
    v_V=ones(RANK,RANK);
    
    diff=1;
    t=0;

    if ((u_truth==[])&&(v_truth==[]))
        @printf("T  Delta diff Free_Entropy damp \n");
    else
        @printf("T  Delta diff Free_Entropy damp Error_u Error_ v \n");
    end
    old_free_nrg=-realmax();delta_free_nrg=0;
    free_nrg=0;

    B_u=zeros(n,RANK);
    A_u=zeros(RANK,RANK);
    B_v=zeros(m,RANK);
    A_v=zeros(RANK,RANK);

    while ((diff>conv_criterion)&&(t<max_iter))    
        #AMP Part
        B_u=(S*v)/sqrt(n)-u_old*v_V/(Delta);
        A_u=v'*v/(n*Delta);
        B_v=(S'*u)/sqrt(n)-v_old*u_V/(Delta);
        A_v=u'*u/(n*Delta);
        #Keep old variables
        u_old=u;
        v_old=v;

        u_new,u_var,logu = Fun_u(A_u,B_u);#Gaussian Prior
        v_new,v_var,logv = Fun_v(A_v,B_v);#Community prior

        #Compute the Free Entropy
        minusDKL_u=logu+0.5*m*trace(A_u*u_var)+trace(0.5*A_u*u'*u)-trace(u'*B_u);   
        minusDKL_v=logv+0.5*n*trace(A_v*v_var)+trace(0.5*A_v*v'*v)-trace(v'*B_v);   
        term_u=-trace((u'*u)*v_var)/(2*Delta);
        term_v=-(m/n)*trace((v'*v)*u_var)/(2*Delta);#this is such that A_u and B_u gets a factor m/n
        term_uv=sum(u*v'.*S)/(sqrt(n))-trace((u'*u)*(v'*v))/(2*n*Delta); 
        free_nrg=(minusDKL_u+minusDKL_v+term_u+term_v+term_uv)/n;

        diff=mean(abs(v_new-v_old))+mean(abs(u_new-u_old));
        u=(1-damp)*u_new+damp*u_old;#damping
        v=(1-damp)*v_new+damp*v_old;#damping
          
        u_V=m*u_var/n;
        v_V=v_var;

        if ((u_truth==[])&&(v_truth==[]))
            @printf("%d %f %e %e %f\n",t,Delta,diff,free_nrg,damp);
        else
            @printf("%d %f %e %e %f %e %e \n",t,Delta,diff,free_nrg,damp,min(mean((u-u_truth).^2),mean((-u-u_truth).^2)),min(mean((v-v_truth).^2),mean((-v-v_truth).^2)));
        end

        t=t+1;
    end

    
    u,v    ;
end


#= List of priors ********************************************************************************* =#

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
   Weight=-0.5*A+B;
   pos=find(Weight>0);
   neg=setdiff([1:size(B,1)],pos);
   MEAN=zeros(size(B));
   MEAN(neg)=rho*exp(-0.5*A+B(neg))./(1-rho+rho*exp(-0.5*A+B(neg)));
   MEAN(pos)= rho./(rho+(1-rho)*exp(0.5*A-B(pos)));
   VAR=mean2(MEAN.*(1-MEAN));
   logZ=sum(log(1-rho+rho*exp(-0.5*A+B(neg))),1);   
   logZ=logZ+sum(-0.5*A+B(pos)+log(rho+(1-rho)*exp(0.5*A-B(pos))),1);
   MEAN,VAR,logZ;
end



end
