import sys
import os

if __name__=="__main__":
    data = []

    possibilities = [0,8,16]
    for p1 in possibilities:
        for p2 in possibilities:
            for p3 in possibilities:
                for p4 in possibilities:
                    for p5 in possibilities:
                        print(f"{p1}_{p2}_{p3}_{p4}_{p5}/data.txt")
                        sys.stdout.flush()
                        try:
                            with open(f"{p1}_{p2}_{p3}_{p4}_{p5}/data.txt",'r') as f:
                                d = f.read()
                                data.append((float(d.split('\n')[0].split(';')[0]),float(d.split('\n')[1].split(';')[0]),float(d.split('\n')[2].split(';')[0]),float(d.split('\n')[3].split(';')[0]),f"{p1}_{p2}_{p3}_{p4}_{p5}"))
                                if data[-1][0]<0.001:
                                    job_name = f'{p1}_{p2}_{p3}_{p4}_{p5}'
                                    with open('run2.py','a') as f:
                                        f.write(f"cd {job_name}\n")
                                        f.write(f"sbatch -n 1 --mem 40G -t 4-00:00 -J {job_name}_redo --export=ALL,P1={p1},P2={p2},P3={p3},P4={p4},P5={p5} ../../scripts/job.sh\n")
                                        f.write(f"cd ../\n\n")
                        except:
                            pass
                                    
    newList = sorted(data, key = lambda x : x[3],reverse=True)

    for i in newList:
        print(i)
