push!(LOAD_PATH, ".")
using LowRAMP

println("Demo of UV' factorization: ")
demo_LowRAMP_UV()
println("Press enter to continue: ")
readline(STDIN)

println("Demo of UV' completion: ")
demo_completion() 
println("Press enter to continue: ")
readline(STDIN)

println("Demo of XX' factorization: ")
demo_LowRAMP_XX() 
println("Press enter to continue: ")
readline(STDIN)

println("Demo of submatrix factorization: ")
demo_submatrix()
println("Press enter to continue: ")
readline(STDIN)
