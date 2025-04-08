import os
import time
import psutil
import subprocess
import torch
import torch.distributed as dist

# 配置参数
TRAIN_SCRIPT = "/mnt/afs/chenxiaoxuan/BELLE/train/scripts/run_pt_test4090_qwen.sh"  # 训练脚本
TRAIN_RESUME_SCRIPT = TRAIN_SCRIPT # resume 脚本
#CHECKPOINT_DIR = "/mnt/afs/chenxiaoxuan/BELLE/train/work_dirs/pt/test"  # 检查点保存路径 ???
#CHECKPOINT_NAME = "checkpoint-24"  # 检查点文件名 ???
MAX_RETRIES = 10000000000  # 最大重试次数  #30天预估出错100次，最大值设成100次  # 无限大，无限重启，但是重启之间有时间间隔限制
SLEEP_INTERVAL = 600  # 检查间隔时间（秒）  # 每10分钟检测一次 600
WAIT_BETREEN_ERROR_AND_RESTART = 600  # 报错后等待一段时间再重启 

# 启动训练函数
def start_training():
    subprocess.Popen(["bash", TRAIN_SCRIPT])
    

# 检查训练进程是否仍在运行
def is_training_running():
    for proc in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent']):
        try :
            if proc.info['name'] == 'bash' and TRAIN_SCRIPT in ' '.join(proc.cmdline()):
                return True
        except Exception as e:
            print(e)
            proc.kill()
            continue

    return False

# 检查GPU状态是否正常
def check_gpu_status():
    try:
        # 获取当前的GPU使用情况，使用nvidia-smi命令
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gpu_util = result.stdout.decode().strip().split('\n')
        if any(int(util) == 0 for util in gpu_util):  # 如果GPU显存占用效益某个阈值，可能有问题
            return False
        return True
    except Exception as e:
        print(f"Error checking GPU status: {e}")
        return False

# 恢复训练
def resume_training():
    #checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
    #if os.path.exists(checkpoint_path):
    #    print(f"Resuming training from checkpoint {checkpoint_path}")
    #    # 假设你的训练脚本支持 --resume 参数来恢复训练
    #    subprocess.Popen(["bash", TRAIN_RESUME_SCRIPT, "--resume", checkpoint_path])
    #else:
    #    print("No checkpoint found, starting fresh.")
    #    start_training()
    subprocess.Popen(["bash", TRAIN_RESUME_SCRIPT])

# 监控函数
def monitor_training():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            # 监控训练进程是否在运行
            if not is_training_running():
                print(f"Training process not running, attempting to restart... (Attempt {retries + 1})")
                time.sleep(SLEEP_INTERVAL)  # 报错后等待一段时间重启程序
                resume_training()
                retries += 1
                print(f'now_retries : {retries}')
                time.sleep(SLEEP_INTERVAL)  # 等待一段时间后重新检查
            else:
                # 检查GPU状态
                if not check_gpu_status():
                    print(f"GPU error detected, attempting to restart training... (Attempt {retries + 1})")
                    time.sleep(SLEEP_INTERVAL)  # 报错后等待一段时间重启程序
                    resume_training()
                    retries += 1
                    print(f'now_retries : {retries}')
                    time.sleep(SLEEP_INTERVAL)
                else:
                    print("Training is running fine.")
                    time.sleep(SLEEP_INTERVAL)
        except:
            time.sleep(SLEEP_INTERVAL) # 看上去无限重启，实际上这个地方一直按ctrl+c就停止了整个程序
            continue


if __name__ == "__main__":
    monitor_training()
