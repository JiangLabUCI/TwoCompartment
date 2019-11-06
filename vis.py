import sys
from staph.analysis.landscape import igate as b2d1_igate
from staph.analysis.igate_brute import igate as bru_igate
from staph.analysis.igate_ntest import igate as igate_ntest
from staph.analysis.igate_thresh import igate as igate_thresh
from staph.analysis.compute_chisq import compute_chisq
from staph.analysis.igate_demo import igate as igate_demo


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
    # fnames = ["ntest.o8451930.3"]
    fnames = []
    flist = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for ind in range(21, 101):
        fnames.append("ntest.o8523481." + str(ind))
    igate_ntest(filenames=fnames, option1=1)
elif option1 == 4:
    fnames = []
    option2 = int(sys.argv[2])
    for ind1 in range(100):
        fnames.append(
            "results/opt/6021324_1000rep0se_"
            + str(ind1)
            + "to"
            + str(ind1)
            + "b2d1_1o5_cpu.npz"
        )
    igate_thresh(filenames=fnames, option1=option2)
elif option1 == 5:
    compute_chisq([6.34, 8.67])
elif option1 == 6:
    igate_demo(option1=2)
