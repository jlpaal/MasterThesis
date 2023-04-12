//
// We use the Metropolis (Glauber) algorithm to find nearby orthogonal vectors
// for a given set of arbitrary ones. For the moment, we will simply start
// with random vectors


   function rv= random_pattern(n,k)
     rv = 2*grand(n,k, "uin", 0,1) - 1;
   endfunction


   function rv= random_pattern_0mag(n,k)
     vec= ones(n,1);
     vec(1:n/2)= -1;
     rv= grand(k, "prm",vec);
   endfunction


   function rvp= apply_move(rv)   // change a configuration at random
     i= grand("uin", 1,N);
     j= grand("uin", 1,K);
     rvp= rv;
     rvp(i,j)= -rv(i,j);
   endfunction

   function rvp= apply_glob_move(rv)   // change a configuration at random
     osa= zeros(1,K);
     osn= zeros(1,K);
     wrong = zeros(rv);
     for i=1:N
       for j=1:K
         rvp= rv;
         rvp(i,j)= -rv(i,j);

         for l=1:K
           osa(l)= rv(:,l)'*rv(:,j);
           osn(l)= rvp(:,l)'*rvp(:,j);
           if abs(osn(l)) > abs(osa(l)) then
             wrong(i,j)= wrong(i,j) + 1;
           end
         end
       end
     end
     m= min(wrong);
     ida= find(wrong == m);
     id= ida(grand("uin",1,length(ida))); 
     rvp= rv;
     rvp(id)= -rv(id);
   endfunction


   function rvp= apply_exch_move(rv)   // change a configuration at random
     j= grand("uin", 2,K);             // conserves the magnetization

     while %T
       [i1,i2]= grand(1,2,"uin", 2,N);
       if rv(i1,j)*rv(i2,j) < 0 then
         break;
       end
     end
     rvp= rv;
     rvp(i1,j)= -rv(i1,j);
     rvp(i2,j)= -rv(i2,j);
   endfunction

                              // calculate the energy of a configuration
   function en= energy(rv)    // .. the quantity to be minimized
     en= sum(abs(rv'*rv - eye(K,K)*N));
   endfunction


   function rvp= apply_move_2all(rv)   // change a configuration at random
     i= grand("uin", 1,N);
     rvp= rv;
     rvp(i,:)= -rv(i,:);
   endfunction


   function rvp= maximize_trace(rv,rv0,nr)
     rvp= rv;
     en= trace(rv0'*rvp);
     for ir=1:nr
       rv1= apply_move_2all(rvp);
       en1= trace(rv0'*rv1);
       if en1 > en then
         rvp= rv1;
         en= en1;
       elseif en1 == en then
         if grand("uin",0,1) == 0 then
           rvp= rv1;
           en= en1;
         end
       end
     end
   endfunction


   function rvp= maximize_trace2(rv,rv0)
     rvp= rv;
     en= trace(rv0'*rvp);
     for ir=1:N
       rv1= rvp;
       rv1(ir,:)= -rv1(ir,:);
       en1= trace(rv0'*rv1);
       if en1 > en then
         rvp= rv1;
         en= en1;
       elseif en1 == en then
         if grand("uin",0,1) == 0 then
           rvp= rv1;
           en= en1;
         end
       end
     end
   endfunction


   function dm= hamming_dist_matrix(rv0,rv)
     dm= rv0'*rv;
     dm= (N - dm)/2;
   endfunction


   function hd= mean_hamming_dist(rv0,rv)
     hd= N/2 - trace(rv0'*rv)/(2*K);
   endfunction

// ------------------------------- Hopfield functions ------------------------

   function [xv]= recover_pattern(W, ks, errp, nt)
     xv = rv0(:,ks);                             // prepare one of the pattern
     iea = find(grand(N,1,'def') < errp);        // states with erronous bits
     xv(iea) = -xv(iea);                         // with probability errp

     it = 0;
     while (check_equilibr(W,xv) ~= 0 & it < nt)
       it = it + 1;
       yv= sign(W*xv);

       j = grand('uin',1,N);            // pattern state 
       ch= xv(j)*yv(j);

       if ch < 0 then
         xv(j) = -xv(j);
       elseif ch == 0 then
         if grand("uin",0,1) == 0 then
           xv(j) = -xv(j);
         end
       end
     end
   endfunction


   function [r]= check_equilibr(W, xv)
     n= length(xv);
     yv= sign(W*xv);
     r= sum(xv.*yv) - n;  // r=0 if and only if xv is an equilibrium state
   endfunction            //  = -1 if one component of yv is zero


   function [r]= hamming_dist(xv,yv)  // the number of different components 
     n = length(xv);                  // modulo one vectors complement
     r = sum(abs(xv-yv));
     r = min(r,2*n - r)/2;
   endfunction

// ---------------------------------------------------------------------------


 grand('setsd',getdate('s'));

 N=80;   // dimension of the vectors (should be a multiple of 4)
 K=17;    // number of vectors (to make orthogonal)

 nrep= 200;   // number of repetitions of the orthogonalization process
 nmc= 10*N;   // number of repetitions of the pattern recognition

 temperature = 2.3;   // (2.5,3.5) temperature in energy-units

 errp = 0.1;     // probability for one neuron to be in the wrong state

 cnt1 = 0;       // pattern recognition without orthogonalization
 cnt2 = 0;
 cnt3 = 0;
 for ir=1:nrep
   rv0 = random_pattern(N,K);
   X= rv0;
   W= X*X' - K*eye(N,N);

   xv= recover_pattern(W, 1, errp, nmc*N);
   d= zeros(1,K);
   for j=1:K
     d(j) = hamming_dist(xv,X(:,j));
   end
   if (min(d) == 0) then
     cnt1 = cnt1 + 1;                        // we found the right state
   elseif (check_equilibr(W,xv) == 0) then
     cnt2 = cnt2 + 1;                        // we reached an equilibrium
   else
     cnt3 = cnt3 + 1;                        // no equilibrium reached
   end
 end
 printf(" %g   %g   %g   %g\n", K, cnt1/nrep, cnt2/nrep, cnt3/nrep);

 rv= rv0;
 en= energy(rv);
 for ir=1:nrep
   rv1= apply_glob_move(rv);
   en1= energy(rv1);

   update = (en1 < en);
   if ~update then
     exupd= exp( -(en1-en)/temperature);
     update= (grand("def") < exupd/(1.0 + exupd));
   end

   if update then
     rv= rv1;
     en= en1;
     rv= maximize_trace2(rv,rv0);
     printf(" %10d  %10d  %10.3f\n", ir, (en+1), mean_hamming_dist(rv0,rv));
   end
 end
 printf(" %10d  %10d  %10.3f\n", ir, (en+1), mean_hamming_dist(rv0,rv));
 printf("\n");
 disp(rv0'*rv);

 cnt1 = 0;       // pattern recognition without orthogonalization
 cnt2 = 0;
 cnt3 = 0;
 for ir=1:nrep
   X= rv;
   W= X*X' - K*eye(N,N);

   xv= recover_pattern(W, 1, errp, nmc*N);
   d= zeros(1,K);
   for j=1:K
     d(j) = hamming_dist(xv,X(:,j));
   end
   if (min(d) == 0) then
     cnt1 = cnt1 + 1;                        // we found the right state
   elseif (check_equilibr(W,xv) == 0) then
     cnt2 = cnt2 + 1;                        // we reached an equilibrium
   else
     cnt3 = cnt3 + 1;                        // no equilibrium reached
   end
 end
 printf(" %g   %g   %g   %g\n", K, cnt1/nrep, cnt2/nrep, cnt3/nrep);
 quit

