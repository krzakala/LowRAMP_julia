#=
Author: Florent Krzakala
Date: 07-27-2015 -- 08-17-2015
Where: Santa Fe Institute (New Mexico USA), Plzen (Czech republic), A330 (AF Paris-Boston)
Description: AMP for Low Rank Estimation LowRAMP for UV' and XX' decomposition
=#

module LowRAMP

export LowRAMP_UV,demo_LowRAMP_UV,demo_submatrix #demos
export LowRAMP_XX,demo_LowRAMP_XX #main functions
export f_Rank1Binary,f_gauss,f_clust #priors

function  demo_submatrix(m=5000,n=5000,m_m=250,n_n=250,Delta=1e-3)
    RANK=1;

    @printf("Creating a %dx%d signal with a %dx%d submatrix hidden \n",m,n,m_m,n_n);
    Y=[zeros(m-m_m,n); ones(m_m,n_n) zeros(m_m,n-n_n)];
    
    @printf("adding a noise with std %f \n",sqrt(n)*sqrt(Delta));
    W=Y/sqrt(n)+sqrt(Delta)*randn(m,n);

    #Computing the score and the inverse Fischer information
    S=W/Delta;Iinv=Delta;

    #Calling the code
    @printf("Running AMP \n");
    damp=0.5;    init_sol=4;
    prior_u(x,y)=f_Rank1Binary(x,y,m_m/m);
    prior_v(x,y)=f_Rank1Binary(x,y,m_m/m);
    tic();
    u_ample,v_ample = LowRAMP_UV(S,Iinv,RANK,prior_u,prior_v,damp,1e-6,100,[],[],init_sol);
    toc()
    #rounding to nearest integer
    u_hat=round(u_ample);
    v_hat=round(v_ample);
    @printf("Done! The Squared Reconstruction error on the matrix reads %e \n",mean((u_hat*v_hat'/sqrt(n)-W).^2));
end

function  demo_LowRAMP_XX(n=2500,RANK=3,p=0.5,Deltaeff=0.05)
    @printf("Creating a problem of rank %d with a %dx%d matrix \n", RANK,n,n);

    Delta=sqrt(p*(1-p)/Deltaeff);
    pout = p - Delta/(RANK*sqrt(n));
    pin = p + (1-1/RANK)*Delta/sqrt(n);

    X = zeros(n,RANK);
    for i=1:n
        X[i,ceil(rand()*RANK)]=1;
    end

    #creating the adjacency matrix
    random1=triu(int(rand(n,n).<pin),1);
    random1=random1 +random1';
    random2=triu(int(rand(n,n).<pout),1);
    random2=random2 +random2';

    #creating problem
    A=X*X'.*random1+(1-X*X').*random2;
    S=(Delta/pout)*A - (1-A)*Delta/(1-pout);
    mu=(pin-pout)*sqrt(n);
    Iinv=(mu*mu/(pout*(1-pout)))^-1;

    #Calling the code
    @printf("Running AMP \n");
    damp=-1;    init_sol=1;
    tic()  
    x_ample = LowRAMP_XX(S,Iinv,RANK,f_clust,damp,1e-6,100,X,init_sol);
    toc()
    @printf("Done! The Squared Reconstruction error on the matrix reads %e \n",mean((x_ample*x_ample'-X*X').^2)/sqrt(n));
end

function  demo_LowRAMP_UV(m=250,n=1000,RANK=3,Delta=1e-2)
    @printf("Creating a problem of rank %d with a %dx%d matrix \n", RANK,m,n);
    V = zeros(n,RANK);
    for i=1:n
        V[i,ceil(rand()*RANK)]=1;
    end
    U = randn(m,RANK);
    
    #Adding noise!
    Y=U*V'/sqrt(n)+sqrt(Delta)*randn(m,n);
    
    #Computing the score and the inverse Fischer information
    S=Y/Delta;Iinv=Delta;
    
    #Calling the code
    @printf("Running AMP \n");
    damp=-1;    init_sol=1;
    tic()
    u_ample,v_ample = LowRAMP_UV(S,Iinv,RANK,f_gauss,f_clust,damp,1e-6,100,U,V,init_sol);
    toc()
    @printf("Done! The Squared Reconstruction error on the matrix reads %e \n",mean((u_ample*v_ample'/sqrt(n)-Y).^2));

end

#= Core functions ********************************************************************************* =#

function LowRAMP_XX( S, Delta ,RANK, Fun_a=f_clust, damping=0.5,conv_criterion=1e-6,max_iter=100,x_truth=[],init_sol=1)
    # Usage AMP Lowrank Estimation for XX' decomposition
    # LowRAMP_XX(S,Delta,RANK,prior for x [f_clust], damping[0.5],conv_criterion[1e-6],max_iter[1000],x_truth=[],init_sol=1)

    n=size(S,1);

    #Initialization
    x=zeros(n,RANK);
    if init_sol==0
        @printf("Zeros initial conditions \n");
    elseif init_sol==1 #Init in the solution
        @printf("Random Gaussian initial conditions \n");
        x=randn(n,RANK);
    elseif init_sol==2
        @printf("Use SVD as an initial condition \n");
        V,D= eigs(S,RANK);
        x=V[:,1:RANK];
    elseif init_sol==3
        @printf("Use solution as an initial condition \n");
        x=x_truth+1e-4*randn(n,RANK);
    elseif init_sol==4
        @printf("Use prior as an initial condition \n");
        x,x_var,log_u = Fun_a(eye(RANK,RANK),zeros(n,RANK));
    elseif init_sol==5
        @printf("Use ones as an initial condition \n");
        x=ones(n,RANK)/n;    
    end

    x_old=zeros(n,RANK);
    x_V=zeros(RANK,RANK);

    diff=1;
    t=0;

    if (x_truth==[])
        @printf("T  Delta diff Free_Entropy damp \n");
    else
        @printf("T  Delta diff Free_Entropy damp Error_x \n");
    end
    old_free_nrg=-realmax();delta_free_nrg=0;
    free_nrg=0;
    
    B=zeros(n,RANK);
    A=zeros(RANK,RANK);

    while ((diff>conv_criterion)&&(t<max_iter))
        #Keep old variable
        A_old=A; 
        B_old=B;
        
        #AMP Part
        B_new=(S*x)/sqrt(n)-x_old*x_V/(Delta);
        A_new=x'*x/(n*Delta);

        #Keep old variables
        x_old=x;

        #Iteration with fixed damping or learner one
        pass=0;
        if (damping==-1)
            damp=1;
        else
            damp=damping;
        end
        while (pass!=1) 
            if (t>0)
                A=(1-damp)*A_old+damp*A_new;
                B=(1-damp)*B_old+damp*B_new;
            else
                A=A_new;
                B=B_new;
            end

            x,x_V,logZ = Fun_a(A,B);

            #Compute the Free Entropy
            minusDKL=logZ+0.5*n*trace(A*x_V)+trace(0.5*A*x'*x)-trace(x'*B)   ;  
            term_x=-trace((x'*x)*x_V)/(2*Delta);
            term_xx=sum(x*x'.*S)/(2*sqrt(n))-trace((x'*x)*(x'*x))/(4*n*Delta);
            free_nrg=(minusDKL+term_x+term_xx)/n;

            #if t==0 accept
            if (t==0)  delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg; break; end
            if (damping>=0)  delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg; break;end
            #Otherwise adapative damping
            if (free_nrg>old_free_nrg)
                delta_free_nrg=old_free_nrg-free_nrg;
                old_free_nrg=free_nrg;
                pass=1;
            else
                damp=damp/2;
                if damp<1e-4;   delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg;   break;end;
            end                                 
        end
            
        diff=mean(abs(x-x_old));
        
        if (x_truth==[])
            @printf("%d %f %e %e %f\n",t,Delta,diff,free_nrg,damp);
        else
            @printf("%d %f %e %e %f %e \n",t,Delta,diff,free_nrg,damp,min(mean((x-x_truth).^2),mean((-x-x_truth).^2)));
        end

        if (abs(delta_free_nrg/free_nrg)<conv_criterion)        
            break;
        end
        t=t+1;
    end
    
    x    ;
end




function LowRAMP_UV( S, Delta ,RANK, Fun_u=f_gauss,Fun_v=f_clust, damping=0.5,conv_criterion=1e-6,max_iter=100,u_truth=[],v_truth=[],init_sol=1)
    # Usage AMP Lowrank Estimation for UV decomposition
    # LowRAMP_UV(S,Delta,RANK,prior for u [f_gauss],prior for v [f_clust], damping[0.5],conv_criterion[1e-6],max_iter[1000],u_truth=[],v_truth=[],init_sol=1)

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
        #Keep old variable
        A_u_old=A_u;        A_v_old=A_v;
        B_u_old=B_u;        B_v_old=B_v;
        
        #AMP Part
        B_u_new=(S*v)/sqrt(n)-u_old*v_var/(Delta);
        A_u_new=v'*v/(n*Delta);
        B_v_new=(S'*u)/sqrt(n)-v_old*(m*u_var/n)/(Delta);
        A_v_new=u'*u/(n*Delta);

        #Keep old variables
        u_old=u;
        v_old=v;

        #Iteration with fixed damping or learner one
        pass=0;
        if (damping==-1)
            damp=1;
        else
            damp=damping;
        end
        while (pass!=1) 
            if (t>0)
                A_u=(1-damp)*A_u_old+damp*A_u_new;
                A_v=(1-damp)*A_v_old+damp*A_v_new;
                B_u=(1-damp)*B_u_old+damp*B_u_new;
                B_v=(1-damp)*B_v_old+damp*B_v_new;
            else
                A_u=A_u_new;                A_v=A_v_new;
                B_u=B_u_new;                B_v=B_v_new;
            end

            u,u_var,logu = Fun_u(A_u,B_u);
            v,v_var,logv = Fun_v(A_v,B_v);

            #Compute the Free Entropy
            minusDKL_u=logu+0.5*m*trace(A_u*u_var)+trace(0.5*A_u*u'*u)-trace(u'*B_u);   
            minusDKL_v=logv+0.5*n*trace(A_v*v_var)+trace(0.5*A_v*v'*v)-trace(v'*B_v);   
            term_u=-trace((u'*u)*v_var)/(2*Delta);
            term_v=-(m/n)*trace((v'*v)*u_var)/(2*Delta);#this is such that A_u and B_u gets a factor m/n
            term_uv=sum(u*v'.*S)/(sqrt(n))-trace((u'*u)*(v'*v))/(2*n*Delta); 
            free_nrg=(minusDKL_u+minusDKL_v+term_u+term_v+term_uv)/n;

            #if t==0 accept
            if (t==0)  delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg; break; end
            if (damping>=0)  delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg; break;end
            #Otherwise adapative damping
            if (free_nrg>old_free_nrg)
                delta_free_nrg=old_free_nrg-free_nrg;
                old_free_nrg=free_nrg;
                pass=1;
            else
                damp=damp/2;
                if damp<1e-4;   delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg;   break;end;
            end                                 
        end
            
        diff=mean(abs(v-v_old))+mean(abs(u-u_old));
        
        if ((u_truth==[])&&(v_truth==[]))
            @printf("%d %f %e %e %f\n",t,Delta,diff,free_nrg,damp);
        else
            @printf("%d %f %e %e %f %e %e \n",t,Delta,diff,free_nrg,damp,min(mean((u-u_truth).^2),mean((-u-u_truth).^2)),min(mean((v-v_truth).^2),mean((-v-v_truth).^2)));
        end

        if (abs(delta_free_nrg/free_nrg)<conv_criterion)        
            break;
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
    Weight=-0.5*A[1]+B;
    pos=find(Weight.>0);
    neg=setdiff([1:size(B,1)],pos);
    MEAN=zeros(size(B));
    MEAN[neg]=rho*exp(-0.5*A[1]+B[neg])./(1-rho+rho*exp(-0.5*A[1]+B[neg]));
    MEAN[pos]= rho./(rho+(1-rho)*exp(0.5*A[1]-B[pos]));
    VAR=mean(MEAN.*(1-MEAN));
    logZ=sum(log(1-rho+rho*exp(-0.5*A[1]+B[neg])));   
    logZ=logZ+sum(-0.5*A[1]+B[pos]+log(rho+(1-rho)*exp(0.5*A[1]-B[pos])));
    MEAN,VAR,logZ;
end



end
