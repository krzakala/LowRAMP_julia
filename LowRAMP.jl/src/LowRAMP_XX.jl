#= Core function for XX factorization ***************************************** =#

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



