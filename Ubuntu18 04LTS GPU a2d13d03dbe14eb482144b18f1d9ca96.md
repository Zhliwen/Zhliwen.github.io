# Ubuntu18.04LTS配置GPU使用环境

版本号ubuntu18.04LTS+GTX2080+CUDA10.0+cUDNN7.4.2+tensorflow-gpu=1.13.1+gcc5.5.0+python3.6.9

# 在Tensorflow官网查看对应的版本

[从源代码编译 | TensorFlow](https://tensorflow.google.cn/install/source)

![Ubuntu18%2004LTS%20GPU%20a2d13d03dbe14eb482144b18f1d9ca96/Untitled.png](Ubuntu18%2004LTS%20GPU%20a2d13d03dbe14eb482144b18f1d9ca96/Untitled.png)

# 卸载旧版本驱动

1. 检查系统中含Nvidia相关文件的文件夹

    ```bash
    dpkg -l | grep -i nvidia
    ```

2. 检查gcc版本，本机5.5.0可用。

    ```bash
    gcc -V
    ```

3. 卸载之前安装过的所有驱动并重启

    ```bash
    sudo apt-get remove --purge nvidia-\*

    sudo apt autoremove 

    sudo reboot
    ```

# 屏蔽ubuntu自带的开源驱动nouveau

1. 使用gedit编辑blacklist.conf文件 

    ```bash
    sudo gedit /etc/modprobe.d/blacklist.conf
    ```

2. 在文件末尾中写入以下内容并保存退出

    ```bash
    blacklist nouveau

    options nouveau modeset=0
    ```

3. 使更改生效并重启

    ```bash
    sudo update-initramfs -u

    sudo reboot
    ```

4. 终端执行以下语句，若无内容输出，则说明nouveau已经屏蔽成功

    ```bash
    lsmod | grep nouveau
    ```

# run方式安装显卡驱动

ppa安装可能遇到版本不匹配，deb安装可能会无限循环登录，推荐run安装。

1. 查看显卡型号，显卡型号GTX2080，输出为 GTX1180。

    ```bash
    lspci |grep -i vga 
    ```

2. 在NVIDIA官网下载对应GPU的驱动文件，放在ubuntu系统Home下（主目录）。（本机440.36.run版本）

    [驱动程序下载](https://www.nvidia.com/Download/index.aspx?spm=a3c0i.o55240zh.a3.3.5b8c4b3acO7aPw&lang=cn)

3. 从图形化界面切换到命令行，输入用户名和密码。F1~6为不同终端。

    ```bash
    Ctrl + Alt + F1

    ```

4. 切回图形界面

    ```bash
    Ctrl + Alt + F7
    ```

5. 关闭图形界面

    ```bash
    sudo service lightdm stop
    ```

6. 修改权限

    ```bash
    sudo chmod a+x N.....run
    ```

7. 安装驱动，注意台式机运行此步骤不要加任何参数，否则会导致开机进入系统仍然使用nouveau驱动，在系统信息（设置-细节）里面是显示Gallium 0.4 on llvmpipe(LLVM 3.8, 128bits)。

    ```bash
    sudo ./NVIDA-Linux-x86_64-440.36.run
    ```

8. 根据提示安装，遇到DKMs？选项选择No，其余一律Yes继续。

    遇到 “The CC version check failed”问题：
    ubuntu18.04LTS内核版本gcc为7.4.0，而系统的gcc编译器为5.5.0，出现不匹配问题，此处忽略了。

9. 验证显卡是否安装上，输出GPU信息以及驱动型号。

    ```bash
    nvidia-smi

    nvidia-settings
    ```

10. 重启以防万一，若重启后遇到循环登录情况，卸载全部驱动重新启动。

    ```bash
    sudo reboot
    ```

# run方式安装CUDA10.0

装完显卡驱动之后，输出显示推荐安装CUDA10.2，但10.0是经过测试与tensorflow-gpu=1.13.1匹配的版本，所以装10.0，后续暂无问题。deb安装会自动安装NVIDIA旧版驱动，造成循环登录问题，故使用run方式安装。

1. CUDA官网下载匹配的run文件，放在Home下。

    [CUDA Toolkit 10.2 Download](https://developer.nvidia.com/cuda-downloads)

2. 修改权限

    ```bash
    sudo chmod 755 cuda_10.0.130_410.48_linux.run
    ```

3. 安装CUDA

    ```bash
    sudo ./cuda_10.0.130_410.48_linux.run
    ```

4. 按住空格跳过阅读步骤，询问是否安装NVIDIA驱动时选择No。
5. 开启图形化界面

    ```bash
    sudo service lightdm start
    ```

6. 验证CUDA是否安装成功，若输出PASS则安装成功。

    ```bash
    cd /usr/local/cuda-10.0/samples/1_Utilities/deviceQuery

    sudo make

    ./deviceQuery
    ```

7. 重启以防万一。

    ```bash
    sudo reboot
    ```

8. 配置CUDA环境变量，一般配置此步即可，若有问题再配置下一步。

    ```bash
    sudo gedit ~/.bashrc  

    # add the two lines following
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    # cuda can change as cuda-10.0 specifically

    source ~/.bashrc  # MUST source
    ```

9. 验证是否配置成功

    ```bash
    nvcc -V
    ```

10. 若有问题，配置此步。

    ```bash

    sudo gedit /etc/ld.so.conf.d/cuda.conf

    # add the two lines following
        /usr/local/cuda/lib64
        /lib

    sudo ldconfig -v

    sudo gedit /etc/profile  

    # add following two lines
        PATH=/usr/local/cuda/bin:$PATH  #  不能有空格！！ 
        export PATH  

    source /etc/profile
    ```

# run方式安装cuDNN

1. 官网注册登录，下载对应的cuDNN版本，本机7.4.2版本，下载cuDNN Library for Linux。

    [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

2. 下载后解压，得到cuda文件夹，里面有include和lib64两个子文件夹。
3. 进入include子文件夹，复制头文件到之前安装的cuda目录。

    ```bash
    sudo cp cudnn.h /usr/local/cuda/include/
    ```

4. 进入lib64子文件夹，记录版本号，方便后期修改。复制动态链接库到之前安装的cuda目录。

    ```bash
    sudo cp lib* /usr/local/cuda/lib64/
    ```

5. 重新生成软连接。

    ```bash
    cd /usr/local/cuda/lib64/sudo rm -rf libcudnn.so libcudnn.so.x  # 删除原有动态文件，版本号注意变化，可
    在cudnn的lib64文件夹中查看

    sudo ln -s libcudnn.so.7.4.2 libcudnn.so.7  # 生成软衔接，具体看cudnn的lib64中为什么版本，注意变化

    sudo ln -s libcudnn.so.7 libcudnn.so  # 生成软链接

    sudo ldconfig -v
    ```

6. 再次验证。

    ```bash
    nvcc -V
    ```

# 在虚拟环境中测试tensorflow的GPU和CPU计算效率

建立python虚拟环境，本机文件夹test_gpu/环境tesgpu /python3.6，激活环境之后再进行以下步骤。

1. 安装tensorflow-gpu和keras。

    ```bash
    pip install tensorflow-gpu==1.13.1

    pip install keras
    ```

2. 启动python，验证tensorflow和keras安装成功。

    ```bash
    import tensorflow
    import keras  #输出：using tensorflow backend
    ```

    keras默认后端是tensorflow，之后可在keras.json文件修改。

3. 简单程序验证，传入参数不能大于30000。

    ```bash
    python [test.py](http://test.py) gpu 2000
    python [test.py](http://test.py) cpu 2000

    #花费时间 
    gpu：2.247696
    cpu： 49.566694
    ```

    ```python
    import sys
    import numpy as np
    import tensorflow as tf
    from datetime import datetime

    device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
    shape = (int(sys.argv[2]), int(sys.argv[2]))
    if device_name == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

    with tf.device(device_name):
        random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
        dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
        sum_operation = tf.reduce_sum(dot_operation)

    startTime = datetime.now()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
            result = session.run(sum_operation)
            print(result)

    # It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
    print("\n" * 5)
    print("Shape:", shape, "Device:", device_name)
    print("Time taken:", datetime.now() - startTime)

    print("\n" * 5)
    ```

4. 监测GPU动态。

    ```bash
    watch -n 0.1 -d nvidia-smi     #每隔0.1秒刷新一次
    ```

5. 一些说明。

    在配置好GPU环境的TensorFlow中 ，如果操作没有明确地指定运行设备，那么TensorFlow会优先选择GPU。

    GPU：从0开始表示第一块GPU，CPU即使多核也只能显示CPU：0。

    虽然GPU可以加速TensorFlow的计算，但一般来说不会把所有的操作全部放在GPU上。一个比较好的实践是将计算密集型的运算放在GPU上，而把其他操作放到CPU上。GPU是机器中相对独立的资源，将计算放入或者转出GPU都需要额外的时间。而且GPU需要将计算时用到的数据从内存复制到GPU设备上，这也需要额外的时间。TensorFlow可以自动完成这些操作而不需要用户特别处理，但为了提高程序运行的速度，用户也需要尽量将相关的运算放在同一个设备上。如果需要将某些运算放到不同的GPU或者CPU上，就需要通过tf.device来手工指定。

    ---

    [GPU参数说明](Ubuntu18%2004LTS%20GPU%20a2d13d03dbe14eb482144b18f1d9ca96/GPU%20a688f473c48641058ac524ca4fc0ddcf.csv)