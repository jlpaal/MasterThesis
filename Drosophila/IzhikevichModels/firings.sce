//
// Izhikevich model for excitatory and inhibitory neurons. 
// Modified to reproduce the model of Wessnitzer et al. for STDP
// (spike timing dependent plasticity)                            15. Sep. 2022
//
// PART 1 of the program prepares the pattern to be learned. After
// normalization they are stored in the variables: Jnpat(:,1:5), as well as
// Anor, Enor, Inor, Onor, Unor.
//
// PART 2 contains the core part of the program, the definition of the many
// dynamical parameters of the program and the three main for-loops which
// compute the evolution of the system over time:
//    "ir=1:nr" loop : repetitions of the same evolution in time, but with
//       different randomly chosen connections between PN and KC neurons; the
//       noise added to the evolution equation is different (statistically 
//       independent) for each run
//     "t=1:??" loop : time-loop (discrete t measures time in units of ms)
//    "tf=1:nfts" loop : loop over fractional time-steps which are used for the
//       discretization of the differential equation (Euler scheme)
//
// The evolution consists of 6 phases: (i) Presentation of the pattern to be
// learned; in this phase the EN only some samples (individuals) will fire 
// their EN, but only those have a chance to actually learn because only those
// can receive the reinforcement signal. (ii) Presentation of the reinforcement
// signal (amine spillage). (iii) Presentation of an alternative pattern which
// should not been learned. (iv) Repetition of the first pattern, to see if it
// has been learned; the EN should fire more often. (v) Repetition of the 
// alternative pattern. (vi) Repetition of the first pattern.

// --- The principal input parameters for the program

// fhan2= mopen("PNKCEN-ENinputs-v3.ou","wt");

 nr= 3;                 // number of samples

 nfts= 10;              // number of fractional time steps
 Delt= 1.0/nfts;

 Np=  49;               // Number of projection neurons
 Nk= 360;               // Number of Kenyon cell neurons
                        // We have one extrinsic neuron, EN
// isee= 1327;
// grand('setsd',isee);
 grand('setsd',getdate('s'));

// --- PART 1 ----------------------------------------------------------------

 Jmin= [180,180,180,180,180];   // Dynamic range for the patterns to achieve
 Jmax= [260,260,260,260,260];   // spiking PN;

 Amat= read("NewPattern/A.dat",7,7)';   // pattern presented to the network
 Emat= read("NewPattern/E.dat",7,7)';
 Imat= read("NewPattern/I.dat",7,7)';
 Omat= read("NewPattern/O.dat",7,7)';
 Umat= read("NewPattern/U.dat",7,7)';

 Jpat= zeros(Np,5);                     // new normalization
 Jpat(:,1)= matrix(Amat,Np,1);          // first normalize, then map
 Jpat(:,2)= matrix(Emat,Np,1);
 Jpat(:,3)= matrix(Imat,Np,1);
 Jpat(:,4)= matrix(Omat,Np,1);
 Jpat(:,5)= matrix(Umat,Np,1);

 snorm= sqrt(sum(Jpat.^2,1))/Np;   // total brightness of every pattern
 sav= sqrt(sum(snorm.^2))/5.0;     // average brightness of all patterns
 Jnpat= zeros(Jpat);               // linear map to sensitive range
 for i=1:5
   Jnpat(:,i)= Jmin(i) + (Jmax(i) - Jmin(i))/255.0 * Jpat(:,i)*sav/snorm(i);
 end

 Anor= Jnpat(1:Np,1);       // For output of the normalized pattern
 Enor= Jnpat(1:Np,2);
 Inor= Jnpat(1:Np,3);
 Onor= Jnpat(1:Np,4);
 Unor= Jnpat(1:Np,5);

 if ~isfile("NewPattern/Anor.dat") then
   write("NewPattern/Anor.dat",matrix(Anor,7,7),'(7i7)');
 end
 if ~isfile("NewPattern/Enor.dat") then
   write("NewPattern/Enor.dat",matrix(Enor,7,7),'(7i7)');
 end
 if ~isfile("NewPattern/Inor.dat") then
   write("NewPattern/Inor.dat",matrix(Inor,7,7),'(7i7)');
 end
 if ~isfile("NewPattern/Onor.dat") then
   write("NewPattern/Onor.dat",matrix(Onor,7,7),'(7i7)');
 end
 if ~isfile("NewPattern/Unor.dat") then
   write("NewPattern/Unor.dat",matrix(Unor,7,7),'(7i7)');
 end

// --- PART 2 ----------------------------------------------------------------

 gPNKC= 0.8;   // fixed synaptic conductance between PN and KC. It should be
               // such that 3-6 PN have to fire together to make a KC neuron 
               // fire as well. In [Wessnitzer (2012)] it is varied between 
               // 0.13 - 0.45.

 tauPNKC= 1.6     // decay time for the dynamical synapse strength in ms
                  // In [Wessnitzer (2012)] it is "1.0".
 phiPNKC= 0.75;   // synapse strength increase due to pre-synaptic PN firing
                  // In [Wessnitzer (2012)] it is "0.5".

 tauKCEN= 5.0;   // decay time for the dynamical synapse strength in ms 
                  // (en Wess: 2.0)
 phiKCEN= 0.5;    // synapse strength increase due to pre-synaptic KC firing
 gmax= 2.0;       // gKCEN max. In [Wessnitzer (2012)] it is "1.0".

 tauc= 2000;      // estimate from Fig.5
 taud= 200;       // these are the values from Luz' program

 C= [100  *ones(Np,1);   4    *ones(Nk,1); 100  ];  // ProjN ;  KCs ;  ExtN
 a= [  0.3*ones(Np,1);   0.01 *ones(Nk,1);   0.3];
 b= [ -0.2*ones(Np,1);  -0.3  *ones(Nk,1);  -0.2];
 c= [-65  *ones(Np,1); -65    *ones(Nk,1); -65  ];
 d= [  8  *ones(Np,1);   8    *ones(Nk,1);   8  ];
 kk=[  2  *ones(Np,1);   0.015*ones(Nk,1);   2  ];
 vr=[-60  *ones(Np,1); -85    *ones(Nk,1); -60  ];
 vt=[-40  *ones(Np,1); -25    *ones(Nk,1); -40  ];

 for ir=1:nr    // repeat the simulation for different samples

   Sind= samwr(10,360,[1:49]);  // Each KC gets input from 10 randomly chosen
   S= zeros(Np+Nk+1,Np+Nk+1);   // PN's; "samwr" (sampling without replacement)
   for k= 1:Nk                  // Sind(is,k) : PN index, connected to KC k
     for is= 1:10
       S(Np+k, Sind(is,k))= 1;  // S(i,j) = 1 means that output of neuron j is 
     end                        // connected to input of neuron i
     S(Np+Nk+1,Np+k)= 1;
   end

   kclastfire= -10.*tauc*ones(Nk,1);
   enlastfire= -10*tauc;

   SKCdyn= zeros(Nk,1);
   SKCfire= zeros(Nk,1);
   SENdyn= zeros(Nk,1);           // conductance of the synapses themselves

   gdyn= 2.0*grand(Nk,1,'def');   // synaptic conductances (to each KC neuron)
   cdyn= zeros(Nk,1);             // synaptic tag
   ddyn= 0.0;                     // reinforcement signal
   taustdp= 25;

   v= vr;
   u= b.*(v - vr);
   firings=[];
   reforce=[];

   for t=1:95000

//   mfprintf(fhan2, " %10.5f  %10.5f  %10.5f  %10.5f  %10.5f\n", ..
//                     t, sum(cdyn), ddyn, sum(gdyn), sum(SENdyn));
//     for i=1:Nk
//       mfprintf(fhan2, " %10.5f  %d  %10.5f  %10.5f  %10.5f  %10.5f\n", ..
//                       t, i, cdyn(i), ddyn, gdyn(i), SENdyn(i));
//     end
//     mfprintf(fhan2, "\n");

     noise= [grand(Np,1,'nor',0,0.5); grand(Nk,1,'nor',0,0.5); ..
             grand(1,1,'nor',0,0.5)];
     for tf= 1:nfts
       fired= find(v >= 30)';

       kcfired= intersect([Np+1:Np+Nk],fired);
       if length(kcfired) > 0 then
         kcfired= kcfired - Np;      // keep array with the times of the last
       end                           // We count KC's only from 1 to Nk

       for l=1:length(kcfired)         // keep array with the times of the last
         kclastfire(kcfired(l))= t;    // firing for the KC-neurons
       end
       if or(410 == fired) then        // and of the last firing of the EN
         enlastfire= t;
       end

       if length(fired) > 0 then
         firings= [firings; t+0*fired, fired];
         v(fired)= c(fired);
         u(fired)= u(fired) + d(fired);
         SKCfire = phiPNKC * sum(S(Np+1:Np+Nk,fired),2);   // effecto de las
       else                                                // PN disparando 
         SKCfire = zeros(SKCfire);
       end

       I = zeros(Np+Nk,1);
       if (t >= 200 & t < 1200) then
         I(1:Np)= Inor; gPNKC= 0.765;   // Pattern to be reinforced
       end

       if (t >= 2200 & t < 3200) then
         I(1:Np)= Anor; gPNKC= 0.8;     // Pattern which should not be learned
       end

       if (t >= 4200 & t < 5200) then
         I(1:Np)= Inor; gPNKC= 0.765;   // For checking if the reinforced
       end                              // pattern has been learned
       if (t >= 6200 & t < 7200) then
         I(1:Np)= Anor; gPNKC= 0.8;     // Checking that the alternative
       end                              // pattern has not been learned
       if (t >= 8200 & t < 9200) then
         I(1:Np)= Inor; gPNKC= 0.765;   // For checking the reinforced pattern
       end

       ba= 0.0;
       if (t >= 1200 & t < 2000) then   // Reinforcement at the end of pattern
         ba= 1.0;                       // input
       end

       I(Np+1:Np+Nk)= - gPNKC * SKCdyn.*v(Np+1:Np+Nk);
       I(Np+Nk+1)= - sum(gdyn.*SENdyn)*v(Np+Nk+1);

       v= v + 0.25*Delt * ( kk.*(v - vr).*(v - vt) - u + I + noise)./C;
       v= v + 0.25*Delt * ( kk.*(v - vr).*(v - vt) - u + I + noise)./C;
       v= v + 0.25*Delt * ( kk.*(v - vr).*(v - vt) - u + I + noise)./C;
       v= v + 0.25*Delt * ( kk.*(v - vr).*(v - vt) - u + I + noise)./C;

       SKCdyn= (1.0 - Delt/tauPNKC)*SKCdyn + SKCfire;

       if length(kcfired) > 0 then
         SENdyn(kcfired)= SENdyn(kcfired) + phiKCEN * S(Np+Nk+1,Np + kcfired)';
       end
       SENdyn= (1.0 - Delt/tauKCEN)*SENdyn;

       u= u + Delt * a.*(b.*(v - vr) - u);

       gdyn= max(0.0, min(gmax, gdyn + Delt * cdyn*ddyn)); 
       ddyn= min(1.0, ddyn - (ddyn/taud - ba)*Delt);
       cdyn= cdyn -cdyn*Delt/tauc;

       if and(410 == fired) then    // only if EN fires alone, t_post > t_pre
         for i=1:Nk
           tpre = kclastfire(i);
           cdyn(i)= cdyn(i) + exp(- (t-tpre)/taustdp);
         end
       end

       for i=1:length(kcfired)
         tpre= enlastfire;
         cdyn(kcfired(i)) = cdyn(kcfired(i)) - exp(- (t-tpre)/taustdp);
       end
     end                          // time sub-time, i.e. integration time loop
   end                            // time t-loop

   for j=1:size(firings,1)
     printf(" %d  %d\n", firings(j,1), firings(j,2));
   end
   printf("\n\n");
 end  // of the repetition loop for different samples

 quit

