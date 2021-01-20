module GenDistributions

using Distributions
using DistributionsAD
using Zygote
using Gen

"""
    DistributionsBacked(dist::F, has_arg_grads::Tuple, has_output_grad::Tuple, return_type::Type)

Create a Gen.jl distribution based on a Distributions.jl distribution.
"""
struct DistributionsBacked{F, T} <: Gen.Distribution{T}
    to_dist :: F
    has_arg_grads :: Tuple
    has_output_grad :: Bool
    return_type :: Type{T}
end

Gen.random(d::DistributionsBacked, args...) = rand(d.to_dist(args...))
Gen.logpdf(d::DistributionsBacked{F, T}, v::T, args...) where {F, T} = Distributions.logpdf(d.to_dist(args...), v)
Gen.logpdf_grad(d::DistributionsBacked{F, T}, v::T, args...) where {F, T} = Zygote.gradient((v, args...) -> Distributions.logpdf(d.to_dist(args...), v), v, args...)
Gen.has_argument_grads(d::DistributionsBacked) = d.has_arg_grads
Gen.has_output_grad(d::DistributionsBacked) = d.has_output_grad


export DistributionsBacked

end # module
