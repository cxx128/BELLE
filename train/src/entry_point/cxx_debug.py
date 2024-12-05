import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import time
import subprocess
#import GPUtil



def init_slurm_env():  # 初始化slurm，国超步骤？？？
    if 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        if proc_id==0:
            print('Init dist using slurm!')
            print("Job Id is {} on {} ".format(os.environ["SLURM_JOBID"], os.environ['SLURM_NODELIST']))

        ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = os.environ['SLURM_NODELIST']
        node_list = os.environ['SLURM_STEP_NODELIST']
        # node_list = os.environ['SLURM_STEP_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        jobid = os.environ["SLURM_JOBID"]
        stepid = os.environ["SLURM_STEP_ID"]
       

        tcp_port = os.environ.get('MASTER_PORT', 9904)


        os.environ['MASTER_PORT'] =str(tcp_port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)

        print('rank: {} world size: {} addr: {}  port: {}'.format(proc_id, ntasks, addr, os.environ['MASTER_PORT']))

def get_gpu_memory_info(rank):
    try:
        # 执行hy-smi命令并获取输出
        result = subprocess.check_output(['hy-smi'])
        # 将输出转换为字符串
        result_str = result.decode('utf-8')
        print(f'rank : {rank}\n'+result_str)
    except Exception as e:
        print(f"Error executing hy-smi: {e}")



def main():
    init_slurm_env()   # 初始化国超
    # 参数设置
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank=int(os.environ['RANK'])
    # amd卡上面检测不到gpu
    #GPUs = GPUtil.getGPUs()
    #print(f'len(GPUs) : {len(GPUs)}')
    #for gpu in GPUs:
    #    print(f"rank : {rank} GPU ID: {gpu.id}, Name: {gpu.name}, VRAM: {gpu.memoryTotal}MB, Used: {gpu.memoryUsed}MB, Free: {gpu.memoryFree}MB, Utilization: {gpu.load*100}%")
    get_gpu_memory_info(rank)
    now_time=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    print(f'finish rank : {rank} now_time : {now_time}')    

        
if __name__=="__main__":
    main()