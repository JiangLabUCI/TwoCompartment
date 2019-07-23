import sys
from staph.analysis.landscape import igate as b2d1_igate
from staph.analysis.igate_brute import igate as bru_igate

option1 = int(sys.argv[1])

if option1 == 1:
    option2 = int(sys.argv[2])
    fnames = [
        "results/6021324_1000rep0se70ite_Powell_0to0b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_10to10b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_11to11b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_12to12b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_13to13b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_14to14b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_15to15b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_16to16b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_17to17b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_18to18b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_19to19b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_1to1b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_20to20b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_21to21b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_22to22b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_23to23b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_24to24b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_2to2b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_3to3b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_4to4b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_5to5b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_6to6b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_7to7b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_8to8b2d1_1o3_cpu.npz",
        "results/6021324_1000rep0se70ite_Powell_9to9b2d1_1o3_cpu.npz",
    ]
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
