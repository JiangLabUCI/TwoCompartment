import sys
from staph.analysis.landscape import igate as b2d1_igate
from staph.analysis.igate_brute import igate as bru_igate
from staph.analysis.igate_ntest import igate as igate_ntest
from staph.analysis.igate_multi import igate as igate_multi
from staph.analysis.compute_chisq import compute_chisq

option1 = int(sys.argv[1])

if option1 == 1:
    option2 = int(sys.argv[2])
    fnames = []
    for ind in range(100):
        fnames.append(
            "results/opt/6021324_1000rep0se100ite_Powell_"
            + str(ind)
            + "to"
            + str(ind)
            + "b2d1_1o3_cpu.npz"
        )
    b2d1_igate(filenames=fnames, choice=option2)
elif option1 == 2:
    fnames = []
    for ind in range(20):
        fnames.append(
            "results/6021324_1000rep0se_"
            + str(ind)
            + "to"
            + str(ind)
            + "b2d1_1o4_cpu.npz"
        )
    bru_igate(filenames=fnames)
elif option1 == 3:
    fnames = []
    flist = [9]
    for ind in flist:
        fnames.append("results/ntest.o7721941." + str(ind))
    igate_ntest(filenames=fnames)
elif option1 == 4:
    igate_multi(fname="results/pred_1000rep200000nstr1hypF6_multi.npz")
elif option1 == 5:
    compute_chisq()
