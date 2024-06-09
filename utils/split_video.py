import os
import subprocess

def main():
    # 视频绝对路径
    video_path = r"D:\\Embedded\\gaussian-splatting\\data\\ggbond\\input\\8.mp4"
    # 切分帧数，每秒多少帧
    fps = 4
    # 起始编号 #从1013开始是暗光条件
    start_number = 1508

    # 获取当前工作路径
    current_path = os.getcwd()
    # 上一级文件夹所在路径
    folder_path = os.path.dirname(video_path)
    # 图片保存路径
    images_path = os.path.join(folder_path, 'input')
    os.makedirs(images_path, exist_ok=True)

    ffmpeg_path = os.path.join(current_path, 'external', r'ffmpeg/bin/ffmpeg.exe')

    # 脚本运行
    # 视频切分脚本
    command = f'{ffmpeg_path} -i {video_path} -qscale:v 1 -qmin 1 -vf fps={fps} -start_number {start_number} {images_path}\\%04d.jpg'
    subprocess.run(command, shell=True)

if __name__=="__main__":
    main()