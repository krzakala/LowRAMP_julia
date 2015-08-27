#= Core functions for UV' factorization **************************************** =#

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
