import os
import datetime

class COLORS:
    RESET = '\033[0m'
    RED = '\033[00;31m'
    GREEN = '\033[00;32m'
    YELLOW = '\033[00;33m'
    BLUE = '\033[00;34m'
    MAGENTA = '\033[00;35m'
    PURPLE = '\033[00;35m'
    CYAN = '\033[00;36m'
    LIGHTGRAY = '\033[00;37m'
    LRED = '\033[01;31m'
    LGREEN = '\033[01;32m'
    LYELLOW = '\033[01;33m'
    LBLUE = '\033[01;34m'
    LMAGENTA = '\033[01;35m'
    LPURPLE = '\033[01;35m'
    LCYAN = '\033[01;36m'
    WHITE = '\033[01;37m'

def now_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def print_log():
    print()

class Logger:
    def __init__(self) -> None:
        pass
    def info(self,message:str,color=COLORS.LGREEN):
        print('%s[%s] INFO %s'%(color,now_time(),COLORS.RESET),message)
        
    def error(self,message:str,color=COLORS.LRED):
        print('%s[%s] ERROR %s%s'%(color,now_time(),chr(128561),COLORS.RESET),message)

    def warning(self,message:str,color=COLORS.LYELLOW):
        print('%s[%s] WARNING %s%s'%(color,now_time(),chr(128530),COLORS.RESET),message)