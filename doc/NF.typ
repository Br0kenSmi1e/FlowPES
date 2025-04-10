#align(center)[
  = Global Optimization through Normalizing Flow
  Longli Zheng
]

Suppose that we are trying to solve the global optimization problem of the form
$
  bold(x)^* = "argmin"_(bold(x)) V(bold(x))\ 
  "s.t." bold(x) in RR^n
$
which is equivalent to sampling the distribution
$
  rho(bold(x)) = frac(e^(-beta V(bold(x))), Z),
$
where $Z$ is the normalization factor, which is usually intractable.

Following the spirit of normalizing flow,
we assume the complicated distribution can be derived from the Gaussian distribution by a bijection $f$, namely,
$
  rho^*(bold(x)) = cal(N) circle.small f(bold(x)) abs("det" frac(partial f, partial bold(x))),
$
where $cal(N)(bold(z))$ is the standard $n$-dimensional normal distribution
$
  cal(N)(bold(z)) = frac(1, sqrt(2 pi)) e^(-bold(z)^2\/2).
$
Also notice that the distribution $rho^*(bold(x))$ is already normalized.
$
  integral  rho^*(bold(x)) "d" bold(x) = integral cal(N) circle.small f(bold(x)) abs("det" frac(partial f, partial bold(x))) "d" bold(x) = integral cal(N)(bold(z)) "d" bold(z)
$

The loss function chosen to train the flow can be the KL divergence.
$
  cal(L) &= "KL"(rho^* || rho) = 
  integral d bold(x) rho^*(bold(x)) [log rho^*(bold(x)) - log rho(bold(x))]\ 
  &= integral d bold(x) rho^*(bold(x)) [beta V(bold(x)) + log Z - 1/2 f(bold(x))^2 + log abs(det frac(partial f, partial bold(z))) - 1/2 log 2pi]\ 
  &= EE_(tilde(bold(x)) tilde rho^*(bold(x)))[beta V(bold(x)) - 1/2 f(bold(x))^2 + log abs(det frac(partial f, partial bold(z)))] + "const."
$
and the sample $tilde(bold(x))$ is easy to get.
$
  tilde(bold(z)) tilde cal(N)(bold(z)), tilde(bold(x)) = f^(-1)(tilde(bold(z)))
$
