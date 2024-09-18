library(copula)
library(ggplot2)
library(dplyr)
library(emg)

# from copula library mvdc.R
asCall <- function(fun, param)
{
  cc <-
    if (length(param) == 0)
      quote(FUN(x))
  else if(is.list(param)) {
    as.call(c(quote(FUN), c(quote(x), as.expression(param))))
  } else { ## assume that [dpq]<distrib>(x, param) will work
    as.call(c(quote(FUN), c(quote(x), substitute(param))))
  }
  cc[[1]] <- as.name(fun)
  cc
}

cond_sample_copula = function(block_levels, ntrials, copula){
  U =c()
  for (i in 1:length(block_levels)){
    b = block_levels[i]
    b_U = cCopula(cbind(b, runif(ntrials)), copula=copula, inverse=TRUE)
    U = rbind(U, b_U)
  }
  return(U)
  
}

blockrMvdc <- function(block_levels, ntrials, mvdc) {
  dim <- dim(mvdc@copula)
  u <- cond_sample_copula(block_levels, ntrials, mvdc@copula)
  x <- u
  for (i in 1:dim) {
    qdf.expr <- asCall(paste0("q", mvdc@margins[i]), mvdc@paramMargins[[i]])
    x[,i] <- eval(qdf.expr, list(x = u[,i]))
  }
  x
}

gen_block_t = function(rho1, df, id_params, mt_params, block_levels, ntrials, seed){
  set.seed(seed)
  t_copula = tCopula(rho1, df=df, dim=2)
  mymvd = mvdc(copula = t_copula, margins = c(id_params$distribution, mt_params$distribution), paramMargins = list(id_params$params, mt_params$params))
  sim_data= blockrMvdc(block_levels, ntrials, mymvd)
  return (sim_data)
}

# ========= Usage =======
# rho1 = .75
# df = 3.75
# id_params = list('distribution' = 'unif', 'params' = list(min = 1.58, max = 4.34))
# mt_params = list('distribution' = 'emg', 'params' = list(mu=1, sigma=.36, lambda=14.5))
# block_levels = c(0.001,.1,.2, .99)
# ntrials =20
# seed = 123
# sim_data = gen_block_t(rho1, df, id_params, mt_params, block_levels, ntrials, seed)
