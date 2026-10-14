import argparse
import subprocess

PYTHON_PATH = "python"




def programcall(cmd: str, timeout: float=10000):
    print(cmd)
    ret = subprocess.check_output(cmd, shell=True, timeout=timeout)
    ret = str(ret, encoding="utf-8")
    return ret.splitlines()


def filtertestacc(outputlines):
    ret = list(filter(lambda line: "Test Accuracy of the model on the CIFAR-10 test dataset" in line, outputlines))
    return float(ret[0].split(":")[-1].strip())


def scorer(acc, standardacc, zeroacc=0.1, fullacc=1):
    '''
    stardardacc代表70分时的成绩
    zeroacc是0分线
    fullacc是满分线
    '''
    if acc < standardacc:
        a = (70-0)/(standardacc-zeroacc)
        b = 0 - zeroacc * a
    else:
        a = (100-70)/(fullacc - standardacc)
        b = 100 - fullacc * a
    y = a*acc + b
    return max(min(y, 100), 0)


import os

def q1():
    print("q1")
    arg_path = "./model_ckpt/cifar10_SimpleCNN_arg.pickle"
    sd_path = "./model_ckpt/cifar10_SimpleCNN.pth"
    assert os.path.exists(arg_path) and os.path.exists(sd_path), f"Please ensure you have correctly trained the model, \n and saved the hyper-parameters as well as the model state dicts in {arg_path} and {sd_path}."
    
    output = programcall(f"{PYTHON_PATH} scripts/test.py --model SimpleCNN")
    acc = filtertestacc(output)
    print(f"Test acc {acc:.4f}")
    return scorer(acc, 67, 60, 73)

def q2():
    print("q2")
    arg_path = "./model_ckpt/cifar10_SimpleComposition_arg.pickle"
    sd_path = "./model_ckpt/cifar10_SimpleComposition.pth"
    assert os.path.exists(arg_path) and os.path.exists(sd_path), f"Please ensure you have correctly trained the model, \n and saved the hyper-parameters as well as the model state dicts in {arg_path} and {sd_path}."
    
    output = programcall(f"{PYTHON_PATH} scripts/test.py --model SimpleComposition")
    acc = filtertestacc(output)
    print(f"Test acc {acc:.4f}")
    return scorer(acc, 70, 65, 78)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", choices=["q1", "q2", "all"], default="all")
    args = parser.parse_args()
    
    if args.q == "all":
        for q in [q1, q2]:
            print(f"score {q():.0f}")
    else:
        print(f"score {eval(args.q)():.0f}")