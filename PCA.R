## Assignment 2 
## Question 1
## Importing .dat files and Constructing a Dataframe
USJ <-
  data.frame(CONT = c(5.7, 6.8, 7.2, 6.8, 7.3, 6.2, 10.6, 7, 7.3, 8.2,
                      7, 6.5, 6.7, 7, 6.5, 7.3, 8, 7.7, 8.3, 9.6, 7.1, 7.6, 6.6, 6.2, 7.5, 7.8,
                      7.1, 7.5, 7.5, 7.1, 6.6, 8.4, 6.9, 7.3, 7.7, 8.5, 6.9, 6.5, 8.3, 8.3, 9,
                      7.1, 8.6), INTG = c(7.9, 8.9, 8.1, 8.8, 6.4, 8.8, 9, 5.9, 8.9, 7.9, 8,
                                          8, 8.6, 7.5, 8.1, 8, 7.6, 7.7, 8.2, 6.9, 8.2, 7.3, 7.4, 8.3, 8.7, 8.9,
                                          8.5, 9, 8.1, 9.2, 7.4, 8, 8.5, 8.9, 6.2, 8.3, 8.3, 8.2, 7.3, 8.2, 7, 8.4,
                                          7.4), DMNR = c(7.7, 8.8, 7.8, 8.5, 4.3, 8.7, 8.9, 4.9, 8.9, 6.7, 7.6, 7.6,
                                                         8.2, 6.4, 8, 7.4, 6.6, 6.7, 7.4, 5.7, 7.7, 6.9, 6.2, 8.1, 8.5, 8.7, 8.3,
                                                         8.9, 7.7, 9, 6.9, 7.9, 7.8, 8.8, 5.1, 8.1, 8, 7.7, 7, 7.8, 5.9, 8.4, 7),
             DILG = c(7.3, 8.5, 7.8, 8.8, 6.5, 8.5, 8.7, 5.1, 8.7, 8.1, 7.4, 7.2, 6.8,
                      6.8, 8, 7.7, 7.2, 7.5, 7.8, 6.6, 7.1, 6.8, 6.2, 7.7, 8.6, 8.9, 8, 8.7,
                      8.2, 9, 8.4, 7.9, 8.5, 8.7, 5.6, 8.3, 8.1, 7.8, 6.8, 8.3, 7, 7.7, 7.5),
             CFMG = c(7.1, 7.8, 7.5, 8.3, 6, 7.9, 8.5, 5.4, 8.6, 7.9, 7.3, 7,
                      6.9, 6.5, 7.9, 7.3, 6.5, 7.4, 7.7, 6.9, 6.6, 6.7, 5.4, 7.4, 8.5,
                      8.7, 7.9, 8.4, 8, 8.4, 8, 7.8, 8.1, 8.4, 5.6, 8.4, 7.9, 7.6, 7,
                      8.4, 7, 7.5, 7.5), DECI = c(7.4, 8.1, 7.6, 8.5, 6.2, 8, 8.5, 5.9,
                                                  8.5, 8, 7.5, 7.1, 6.6, 7, 8, 7.3, 6.5, 7.5, 7.7, 6.6, 6.6, 6.8,
                                                  5.7, 7.3, 8.4, 8.8, 7.9, 8.5, 8.1, 8.6, 7.9, 7.8, 8.2, 8.5, 5.9,
                                                  8.2, 7.9, 7.7, 7.1, 8.3, 7.2, 7.7, 7.7), PREP = c(7.1, 8, 7.5, 8.7,
                                                                                                    5.7, 8.1, 8.5, 4.8, 8.4, 7.9, 7.1, 6.9, 7.1, 6.6, 7.9, 7.3, 6.8,
                                                                                                    7.1, 7.7, 6.2, 6.7, 6.4, 5.8, 7.3, 8.5, 8.9, 7.8, 8.4, 8.2, 9.1,
                                                                                                    8.2, 7.6, 8.4, 8.5, 5.6, 8.2, 7.9, 7.7, 6.7, 7.7, 6.9, 7.8, 7.4),
             FAMI = c(7.1, 8, 7.5, 8.7, 5.7, 8, 8.5, 5.1, 8.4, 8.1, 7.2, 7, 7.3,
                      6.8, 7.8, 7.2, 6.7, 7.3, 7.8, 6, 6.7, 6.3, 5.9, 7.3, 8.5, 9, 7.8,
                      8.3, 8.4, 9.1, 8.4, 7.4, 8.5, 8.5, 5.6, 8.1, 7.7, 7.7, 6.7, 7.6,
                      6.9, 8.2, 7.2), ORAL = c(7.1, 7.8, 7.3, 8.4, 5.1, 8, 8.6, 4.7, 8.4,
                                               7.7, 7.1, 7, 7.2, 6.3, 7.8, 7.1, 6.4, 7.1, 7.5, 5.8, 6.8, 6.3, 5.2,
                                               7.2, 8.4, 8.8, 7.8, 8.3, 8, 8.9, 7.7, 7.4, 8.1, 8.4, 5.3, 7.9, 7.6,
                                               7.5, 6.7, 7.5, 6.5, 8, 6.9), WRIT = c(7, 7.9, 7.4, 8.5, 5.3, 8,
                                                                                     8.4, 4.9, 8.5, 7.8, 7.2, 7.1, 7.2, 6.6, 7.8, 7.2, 6.5, 7.3, 7.6,
                                                                                     5.8, 6.8, 6.3, 5.8, 7.3, 8.4, 8.9, 7.7, 8.3, 8.1, 9, 7.9, 7.4, 8.3,
                                                                                     8.4, 5.5, 8, 7.7, 7.6, 6.7, 7.7, 6.6, 8.1, 7), PHYS = c(8.3, 8.5,
                                                                                                                                             7.9, 8.8, 5.5, 8.6, 9.1, 6.8, 8.8, 8.5, 8.4, 6.9, 8.1, 6.2, 8.4,
                                                                                                                                             8, 6.9, 8.1, 8, 7.2, 7.5, 7.4, 4.7, 7.8, 8.7, 9, 8.3, 8.8, 8.4,
                                                                                                                                             8.9, 8.4, 8.1, 8.7, 8.8, 6.3, 8, 8.1, 8.5, 8, 8.1, 7.6, 8.3, 7.8),
             RTEN = c(7.8, 8.7, 7.8, 8.7, 4.8, 8.6, 9, 5, 8.8, 7.9, 7.7, 7.2,
                      7.7, 6.5, 8, 7.6, 6.7, 7.4, 8, 6, 7.3, 6.6, 5.2, 7.6, 8.7, 9, 8.2,
                      8.7, 8.1, 9.2, 7.5, 7.9, 8.3, 8.8, 5.3, 8.2, 8, 7.7, 7, 7.9, 6.6,
                      8.1, 7.1), row.names = c("AARONSON,L.H.", "ALEXANDER,J.M.",
                                               "ARMENTANO,A.J.", "BERDON,R.I.", "BRACKEN,J.J.", "BURNS,E.B.",
                                               "CALLAHAN,R.J.", "COHEN,S.S.", "DALY,J.J.", "DANNEHY,J.F.",
                                               "DEAN,H.H.", "DEVITA,H.J.", "DRISCOLL,P.J.", "GRILLO,A.E.",
                                               "HADDEN,W.L.JR.", "HAMILL,E.C.", "HEALEY.A.H.", "HULL,T.C.",
                                               "LEVINE,I.", "LEVISTER,R.L.", "MARTIN,L.F.", "MCGRATH,J.F.",
                                               "MIGNONE,A.F.", "MISSAL,H.M.", "MULVEY,H.M.", "NARUK,H.J.",
                                               "O\'BRIEN,F.J.", "O\'SULLIVAN,T.J.", "PASKEY,L.", "RUBINOW,J.E.",
                                               "SADEN.G.A.", "SATANIELLO,A.G.", "SHEA,D.M.", "SHEA,J.F.JR.",
                                               "SIDOR,W.J.", "SPEZIALE,J.A.", "SPONZO,M.J.", "STAPLETON,J.F.",
                                               "TESTO,R.J.", "TIERNEY,W.L.JR.", "WALL,R.A.", "WRIGHT,D.B.",
                                               "ZARRILLI,K.J."))
##Calling library Psych
library(psych)
fa.parallel(USJ)
## This shows that we need to extract One principal component.
pc<-principal(USJ,nfactors=1,score=TRUE)
pc
## This shows that the principal component can effectively summarize all the components of the data other than CONT
pcr<-principal(USJ,nfactors=1,rotate = "varimax")
pcr
## This shows that the principal component is in the direction of maximun varience.
head(pc$scores)
##Question 2
gid<-read.csv("Glass Identification Data.csv")
##importing Glass Identification Data file
##Calling library Psych
library(psych)
fa.parallel(gid)
## This shows that we need to extract four principal component.
pcg<-principal(gid,nfactors=4,rotate="none",score=TRUE)
pcg
## This shows that the principal component can not effectively coorelate all the components of the data using rotation will result in better components
pcgr<-principal(gid,nfactors=4,rotate = "varimax")
pcgr
## This gives us the componets that are aligned in the direction of maximum variance.
head(pcg$scores)
## This gives the scores i.e. the linear coefficients for every data entry.
