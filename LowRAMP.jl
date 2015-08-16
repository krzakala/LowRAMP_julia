#=
Author: Florent Krzakala
Date: 07-27-2015
Where: Santa Fe Institute
Description: AMP for Low Rank Estimation AMPLE for UV' decomposition
=#

module AMPLE

export AMPLE_UV,demo_AMPLE_UV

function  demo_AMPLE_UV(n,m,RANK,Delta)
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
    damp=0.8;
    init_sol=0;
    tic()
    u_ample,v_ample = AMPLE_UV(S,Iinv,U,V,RANK,damp,init_sol,Y)    ;
    toc()
    u_ample,v_ample 
end


function AMPLE_UV( S, Delta , u_truth ,v_truth , RANK,damp,init_sol,Y)
    # Usage AMP Lowrank Estimation for UV decomposition
    # AMPLE_UV( S , Delta, u_truth ,v_truth , RANK,damp,init_sol)    
    m,n=size(S);
    u_new=zeros(m,RANK);   
    v_new=zeros(n,RANK);
    u=randn(m,RANK);
    v=rand(n,RANK);    
    if init_sol==1 #Init in the solution
        u=u_truth+1e-4*randn(m,RANK);
        v=v_truth+1e-4*randn(n,RANK);
    elseif init_sol==2 #Init with SVD
        @printf("Use SVD as an initial condition")
        U,SS,V = svds(S,RANK);
        u=U[:,1:RANK];
        v=V[:,1:RANK];
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
    
    @printf("T  Delta diff Error_u Error_ v Error_Y \n");

    B_u=zeros(n,RANK);
    A_u=zeros(RANK,RANK);
    B_v=zeros(m,RANK);
    A_v=zeros(RANK,RANK);
    
    while (diff>1e-6)
        #AMP Part
        B_u=(S*v)/sqrt(n)-u_old*v_V/(Delta);
        A_u=v'*v/(n*Delta);
        B_v=(S'*u)/sqrt(n)-v_old*u_V/(Delta);
        A_v=u'*u/(n*Delta);
        #Keep old variables
        u_old=u;
        v_old=v;

        u_new,u_var = f_gauss(A_u,B_u);#Gaussian Prior
        v_new,v_var = f_clust(A_v,B_v,RANK,n);#Community prior

        diff=mean(abs(v_new-v_old))+mean(abs(u_new-u_old));
        u=(1-damp)*u_new+damp*u_old;#damping
        v=(1-damp)*v_new+damp*v_old;#damping
          
        u_V=m*u_var/n;
        v_V=v_var;

        err=mean((Y-u*v'/sqrt(n)).^2);
        @printf("%d %f %f %e %e %e \n",t,Delta,diff,min(mean((u-u_truth).^2),mean((-u-u_truth).^2)),min(mean((v-v_truth).^2),mean((-v-v_truth).^2)),err);         
        t=t+1;
        if t>1000 break;
        end
        
    end

    
    u,v    ;
end

function f_gauss(A,B)
    s1,s2=size(A);
    VAR=inv(eye(s1,s2)+A);
    MEAN=B*VAR;
    MEAN,VAR;
end

function f_clust(A,B,RANK,n)
    AA=repmat(diag(A),1,n);
    Prob=-0.5*AA+B';
    Prob=exp(Prob-repmat(maximum(Prob,1),RANK,1));
    Norm=repmat(sum(Prob,1),RANK,1);
    Prob=Prob./Norm;
    MEAN=Prob';
    VAR=diagm(vec(sum(MEAN,1)/n))-Prob*Prob'/n;
    MEAN,VAR;
end



end
