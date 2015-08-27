#=
Author: Florent Krzakala
Date: 07-27-2015 -- 08-17-2015
Where: Santa Fe Institute (New Mexico USA), Plzen (Czech republic), A330 (AF Paris-Boston) and Harvard for the 0.4 version
Description: AMP for Low Rank Estimation LowRAMP for UV' and XX' decomposition
This is a very preliminary module, next iteration should be more stable
=#

module LowRAMP

export LowRAMP_UV,demo_LowRAMP_UV,demo_submatrix,demo_completion #demos
export LowRAMP_XX,demo_LowRAMP_XX #main functions
export f_Rank1Binary,f_gauss,f_clust #priors

include("LowRAMP_UV.jl")
include("LowRAMP_XX.jl")
include("Priors.jl")
include("demo.jl")

end
