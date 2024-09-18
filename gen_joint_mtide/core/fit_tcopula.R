# Copula package
library(copula)
library(dplyr)

id = c(1,1,1,1)
mt = c(2,2,2,2)
fit_t_copula = function(id,mt){
  cop_model = ellipCopula ("t", dim = 2)
  m = pobs(as.matrix(cbind(id,mt)))
  t_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent = TRUE)
  return (t_fit@estimate)
}

