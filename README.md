# GenDistributions.jl

This package implements a lightweight compatibility layer for using
distributions from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) in 
[Gen.jl](https://github.com/probcomp/Gen).

```julia
using GenDistributions
using Distributions
using Gen

# Arguments are:
#   * A function taking parameters to Distributions.jl distributions
#   * A tuple of Bools indicating whether Zygote should differentiate the logpdf
#     function w.r.t. each parameter
#   * A Bool indicating whether Zygote should differentiate the logpdf w.r.t.
#     the sampled value
#   * The output Julia type of the distribution.
# E.g.:
const dirichlet = DistributionsBacked(alpha -> Dirichlet(alpha), (true,), true, Vector{Float64})
const flip      = DistributionsBacked(p -> Bernoulli(p), (true,), false, Bool) 

# Using within Gen:
@gen function pick()
  x ~ dirichlet(ones(10))
  y ~ categorical(x)
  return (x, y)
end
```
