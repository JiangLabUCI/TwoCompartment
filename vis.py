import sys
from staph.analysis.landscape import igate as b2d1_igate
from staph.analysis.igate_brute import igate as bru_igate

option1 = int(sys.argv[1])

if option1 == 1:
    option2 = int(sys.argv[2])
    fnames = []
    for ind in range(100):
        fnames.append(
            # "results/6021324_1000rep0se30ite_Powell_"
            "results/6021324_1000rep0se100ite_Powell_"
            + str(ind)
            + "to"
            + str(ind)
            + "b2d1_1o3_cpu.npz"
        )
    b2d1_igate(filenames=fnames, choice=option2)
elif option1 == 2:
    filename = "results/6021324_1000rep0se_0to0b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_1to1b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_2to2b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_3to3b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_4to4b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_5to5b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_6to6b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_7to7b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_8to8b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_9to9b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_10to10b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_11to11b2d1_1o4_cpu.npz"
    # filename = "results/6021324_1000rep0se_12to12b2d1_1o4_cpu.npz"
    bru_igate(filename=filename)
