library(poweRlaw)


degrees_csv <- read.csv("./degrees_neighbors_200_0_0.5.txt")

data_pl <- displ$new(degrees_csv[,1])

data_pl$setXmin(2) 

est <- estimate_pars(data_pl)
data_pl$xmin <- est$xmin
data_pl$pars <- est$pars

bs <- bootstrap_p(data_pl, no_of_sims = 1000, seed=1, threads=4)

p_val <- bs$p

print("POWERLAW FIT FOR TEMPERATURE = 0.5 (STRUCTURE-BASED)")
print(data_pl$xmin)
print(data_pl$pars)
print(p_val)

degrees_csv <- read.csv("./degrees_neighbors_200_0_1.0.txt")

data_pl <- displ$new(degrees_csv[,1])

data_pl$setXmin(3) 

est <- estimate_pars(data_pl)
data_pl$xmin <- est$xmin
data_pl$pars <- est$pars

bs <- bootstrap_p(data_pl, no_of_sims = 1000, seed=1, threads=4)

p_val <- bs$p

print("POWERLAW FIT FOR TEMPERATURE = 1.0 (STRUCTURE-BASED)")
print(data_pl$xmin)
print(data_pl$pars)
print(p_val)


degrees_csv <- read.csv("./degrees_neighbors_200_0_1.5.txt")

data_pl <- displ$new(degrees_csv[,1])

data_pl$setXmin(4) 

est <- estimate_pars(data_pl)
data_pl$xmin <- est$xmin
data_pl$pars <- est$pars

bs <- bootstrap_p(data_pl, no_of_sims = 1000, seed=1, threads=4)

p_val <- bs$p

print("POWERLAW FIT FOR TEMPERATURE = 1.5 (STRUCTURE-BASED)")
print(data_pl$xmin)
print(data_pl$pars)
print(p_val)

# DEGREE BASED

degrees_csv <- read.csv("./degrees_200_0_0.5.txt")

data_pl <- displ$new(degrees_csv[,1])

data_pl$setXmin(1) 

est <- estimate_pars(data_pl)
data_pl$xmin <- est$xmin
data_pl$pars <- est$pars

bs <- bootstrap_p(data_pl, no_of_sims = 100, seed=1, threads=4)

p_val <- bs$p

print("POWERLAW FIT FOR TEMPERATURE = 0.5 (DEGREE-BASED)")
print(data_pl$xmin)
print(data_pl$pars)
print(p_val)

degrees_csv <- read.csv("./degrees_200_0_1.0.txt")

data_pl <- displ$new(degrees_csv[,1])

data_pl$setXmin(2) 

est <- estimate_pars(data_pl)
data_pl$xmin <- est$xmin
data_pl$pars <- est$pars

bs <- bootstrap_p(data_pl, no_of_sims = 100, seed=1, threads=4)

p_val <- bs$p

print("POWERLAW FIT FOR TEMPERATURE = 1.0 (DEGREE-BASED)")
print(data_pl$xmin)
print(data_pl$pars)
print(p_val)


degrees_csv <- read.csv("./degrees_200_0_1.5.txt")

data_pl <- displ$new(degrees_csv[,1])

data_pl$setXmin(2) 

est <- estimate_pars(data_pl)
data_pl$xmin <- est$xmin
data_pl$pars <- est$pars

bs <- bootstrap_p(data_pl, no_of_sims = 100, seed=1, threads=4)

p_val <- bs$p

print("POWERLAW FIT FOR TEMPERATURE = 1.5 (DEGREE-BASED)")
print(data_pl$xmin)
print(data_pl$pars)
print(p_val)

