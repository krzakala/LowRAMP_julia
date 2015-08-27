#= List of demo =#

function  demo_completion(m=5000,n=5000,RANK=3,Delta=1e-4)

   @printf("Creating a %dx%d matrix of rank %d ...\n",m,n,RANK);

    U = randn(m,RANK);
    V = randn(n,RANK);

    @printf(" ...and adding a Gaussian noise with sigma  %f \n",sqrt(Delta));

    #Adding noise!
    Y=U*V'/sqrt(n)+sqrt(Delta)*randn(m,n);

    #Computing the score and the inverse Fischer information
    S=Y/Delta;Iinv=Delta;

    @printf("Let us now hide 90 percent of all entries \n");
    fraction=0.1;
    Support=round(Int,(rand(size(Y)).<fraction));
    
    #Calling the code
    @printf("Running AMP \n");
    damp=0.5;init_sol=1;
    tic
    u_ample,v_ample = LowRAMP_UV(S.*Support,Iinv/fraction,RANK,f_gauss,f_gauss,damp,1e-6,100,[],[],init_sol);
    toc;
    @printf("Done! The Squared Reconstruction error on the matrix reads %e \n",mean((u_ample*v_ample'/sqrt(n)-Y).^2));
end

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
        X[i,round(Int,ceil(rand()*RANK))]=1;
    end

    #creating the adjacency matrix
    random1=triu(round(Int,(rand(n,n).<pin)),1);
    random1=random1 +random1';
    random2=triu(round(Int,(rand(n,n).<pout)),1);
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
        V[i,round(Int,ceil(rand()*RANK))]=1;
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
