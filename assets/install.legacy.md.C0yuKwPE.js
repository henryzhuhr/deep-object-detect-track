import{_ as s,c as i,o as a,a1 as e}from"./chunks/framework.BMAahRQD.js";const F=JSON.parse('{"title":"安装环境","description":"","frontmatter":{"lastUpdated":true,"editLink":true,"footer":true,"outline":"deep"},"headers":[],"relativePath":"install.legacy.md","filePath":"install.legacy.md","lastUpdated":1717762479000}'),t={name:"install.legacy.md"},l=e(`<h1 id="安装环境" tabindex="-1">安装环境 <a class="header-anchor" href="#安装环境" aria-label="Permalink to &quot;安装环境&quot;">​</a></h1><h2 id="获取代码" tabindex="-1">获取代码 <a class="header-anchor" href="#获取代码" aria-label="Permalink to &quot;获取代码&quot;">​</a></h2><div class="vp-code-group vp-adaptive-theme"><div class="tabs"><input type="radio" name="group-d-9rl" id="tab-9RyANtv" checked="checked"><label for="tab-9RyANtv">SSH(Recommend)</label><input type="radio" name="group-d-9rl" id="tab-qoJ9OsR"><label for="tab-qoJ9OsR">HTTP</label></div><div class="blocks"><div class="language-shell vp-adaptive-theme active"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 需要配置 github 上的 SSH key</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> clone</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --recursive</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> git@github.com:HenryZhuHR/deep-object-detect-track.git</span></span></code></pre></div><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> clone</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --recursive</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https://github.com/HenryZhuHR/deep-object-detect-track.git</span></span></code></pre></div></div></div><p>进入项目目录</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cd</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> deep-object-detect-track</span></span></code></pre></div><blockquote><p>后续的脚本基于 <code>deep-object-detect-track</code> 目录下执行</p></blockquote><p>如果未能获取子模块，可以手动获取，如果 <code>git submodule</code> 无法获取，可以使用 <code>git clone</code> 获取</p><div class="vp-code-group vp-adaptive-theme"><div class="tabs"><input type="radio" name="group-sEsIl" id="tab-r59kRiX" checked="checked"><label for="tab-r59kRiX">git submodule</label><input type="radio" name="group-sEsIl" id="tab-Ophh_zw"><label for="tab-Ophh_zw">git clone</label></div><div class="blocks"><div class="language-shell vp-adaptive-theme active"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># in deep-object-detect-track directory</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> submodule</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> init</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> submodule</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> update</span></span></code></pre></div><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> clone</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https://github.com/ultralytics/yolov5.git</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> projects/yolov5</span></span></code></pre></div></div></div><h2 id="系统要求" tabindex="-1">系统要求 <a class="header-anchor" href="#系统要求" aria-label="Permalink to &quot;系统要求&quot;">​</a></h2><h3 id="操作系统" tabindex="-1">操作系统 <a class="header-anchor" href="#操作系统" aria-label="Permalink to &quot;操作系统&quot;">​</a></h3><p>项目在 Linux(Ubuntu) 和 MacOS 系统并经过测试 ，经过测试的系统：</p><ul><li>✅ Ubuntu 22.04 jammy (CPU &amp; GPU)</li><li>✅ MacOS (CPU)</li></ul><div class="warning custom-block"><p class="custom-block-title">WARNING</p><p>项目不支持 Windows 系统 ❌ ，如果需要在 Windows 系统上运行，可以使用 WSL2 或者根据提供的脚本手动执行；虽然已经测试通过，但是不保证所有功能都能正常运行，因此不接受 Windows 系统的问题反馈</p></div><h3 id="gpu" tabindex="-1">GPU <a class="header-anchor" href="#gpu" aria-label="Permalink to &quot;GPU&quot;">​</a></h3><p>如果需要使用 GPU 训练模型，需要安装 CUDA Toolkit，可以参考 <a href="https://developer.nvidia.com/cuda-toolkit-archive" target="_blank" rel="noreferrer">CUDA Toolkit Archive</a> 下载对应版本的 CUDA Toolkit，具体下载的版本需要参考 <a href="https://pytorch.org/get-started/locally/" target="_blank" rel="noreferrer"><em>INSTALL PYTORCH</em></a></p><p>例如 Pytorch 2.3.0 支持 CUDA 11.8/12.1，因此安装 CUDA 11.8/12.1 即可，而不需要过高的 CUDA 版本，安装后需要配置环境变量</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># ~/.bashrc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">export</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> CUDA_VERSION</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">12.1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">export</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> CUDA_HOME</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;/usr/local/cuda-\${</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">CUDA_VERSION</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">}&quot;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">export</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> PATH</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">$CUDA_HOME</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">/bin:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">$PATH</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">export</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LD_LIBRARY_PATH</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">$CUDA_HOME</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">/lib64:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">$LD_LIBRARY_PATH</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span></span></code></pre></div><blockquote><p>事实上，Pytorch 1.8 开始就会在安装的时候自动安装对应的 CUDA Toolkit，因此不需要手动安装 CUDA Toolkit，因此可以跳过这一步</p></blockquote><p>MacOS 系统不支持 CUDA Toolkit，可以使用 CPU 训练模型 (Yolov5 项目暂不支持 MPS 训练)，但是推理过程可以使用 Metal ，参考 <a href="https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/#getting-started" target="_blank" rel="noreferrer"><em>Introducing Accelerated PyTorch Training on Mac</em></a> 和 <a href="https://pytorch.org/docs/stable/notes/mps.html#mps-backend" target="_blank" rel="noreferrer"><em>MPS backend</em></a></p><h2 id="安装环境-1" tabindex="-1">安装环境 <a class="header-anchor" href="#安装环境-1" aria-label="Permalink to &quot;安装环境&quot;">​</a></h2><p>这里安装的环境指的是需要训练的环境，如果不需要训练而是直接部署，请转至 「<a href="./deploy.html">模型部署</a>」 文档</p><p>提供两种方式安装， venv 或 conda</p><ul><li><strong>venv</strong> : 嵌入式设备的部署建议使用这种方案，以确保链接到系统的库，如果没有安装，请安装</li></ul><div class="vp-code-group vp-adaptive-theme"><div class="tabs"><input type="radio" name="group-cilZu" id="tab-WwgehbS" checked="checked"><label for="tab-WwgehbS">Linux</label><input type="radio" name="group-cilZu" id="tab-QdW3gb4"><label for="tab-QdW3gb4">MacOS</label></div><div class="blocks"><div class="language-shell vp-adaptive-theme active"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">sudo</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> apt</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> install</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -y</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> python3-venv</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> python3-pip</span></span></code></pre></div><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Mac 貌似自带了 python3-venv</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># brew install python3-venv python3-pip</span></span></code></pre></div></div></div><ul><li><strong>conda</strong> : 如果没有安装，请从 <a href="https://docs.anaconda.com/free/miniconda/index.html" target="_blank" rel="noreferrer">Miniconda</a> 下载，或者快速安装</li></ul><div class="vp-code-group vp-adaptive-theme"><div class="tabs"><input type="radio" name="group-Rg117" id="tab-ruChhl6" checked="checked"><label for="tab-ruChhl6">linux x64</label><input type="radio" name="group-Rg117" id="tab-lQLLZcJ"><label for="tab-lQLLZcJ">MacOS arm64</label></div><div class="blocks"><div class="language-shell vp-adaptive-theme active"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">wget</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">bash</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Miniconda3-latest-Linux-x86_64.sh</span></span></code></pre></div><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">wget</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">zsh</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> Miniconda3-latest-MacOSX-arm64.sh</span></span></code></pre></div></div></div><h3 id="方法一-手动安装" tabindex="-1">方法一：手动安装 <a class="header-anchor" href="#方法一-手动安装" aria-label="Permalink to &quot;方法一：手动安装&quot;">​</a></h3><p>创建虚拟环境</p><div class="vp-code-group vp-adaptive-theme"><div class="tabs"><input type="radio" name="group-DjLnt" id="tab-FNEojDv" checked="checked"><label for="tab-FNEojDv">在项目内安装环境(推荐)</label><input type="radio" name="group-DjLnt" id="tab-ftbebkH"><label for="tab-ftbebkH">全局安装环境</label></div><div class="blocks"><div class="language-shell vp-adaptive-theme active"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">conda</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> create</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -p</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> .env/deep-object-detect-track</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> python=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3.10</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -y</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">conda</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> activate</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ./.env/deep-object-detect-track</span></span></code></pre></div><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">conda</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> create</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> deep-object-detect-track</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> python=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3.10</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -y</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">conda</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> activate</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> deep-object-detect-track</span></span></code></pre></div></div></div><blockquote><p>Python 版本选择 3.10 是因为 Ubuntu 22.04 默认安装的 Python 版本是 3.10</p></blockquote><ul><li>如果电脑有 NVIDIA GPU，可以直接安装 <a href="https://pytorch.org/get-started/locally/" target="_blank" rel="noreferrer">PyTorch</a> 和其他依赖</li></ul><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">pip</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> install</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -r</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> requirements.txt</span></span></code></pre></div><ul><li>如果电脑没有 NVIDIA GPU，可以安装 CPU 版本的 PyTorch</li></ul><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">pip</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> install</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -r</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> requirements/requirements-cpu.txt</span></span></code></pre></div><h3 id="方法二-使用提供的脚本" tabindex="-1">方法二：使用提供的脚本 <a class="header-anchor" href="#方法二-使用提供的脚本" aria-label="Permalink to &quot;方法二：使用提供的脚本&quot;">​</a></h3><p>提供的安装脚本依赖于基本环境变量 <code>scripts/variables.sh</code> ，可以复制一份到项目目录下进行自定义修改（推荐），如果不需要修改，可以直接执行</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">cp</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> scripts/variables.sh</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> scripts/variables.custom.sh</span></span></code></pre></div><ul><li><code>CACHE_DIR</code>: 用于存放一些缓存文件，例如 <code>yolov5/requirements.txt</code>，默认为项目目录下的 <code>.cache</code></li><li>安装过程会自动检测 <code>CUDA_VERSION</code> 以安装对应的 PyTorch 版本，否则默认安装 CPU 版本的 PyTorch；如果电脑有 NVIDIA GPU 但是不想安装 CUDA Toolkit 到全局系统（需要 sudo）可以取消注释 <code>export CUDA_VERSION=12.1</code> 以安装对应的 PyTorch 版本</li></ul><p>运行会自动检测是否存在用户自定义的环境变量 <code>scripts/variables.custom.sh</code> ，如果存在则使用自定义的环境变量，否则使用默认的环境变量 <code>scripts/variables.sh</code></p><p>执行命令自动创建并且激活虚拟环境，默认使用 <code>venv</code>，<strong>可以重复执行该脚本获取激活环境的提示信息或者安装依赖</strong></p><div class="vp-code-group vp-adaptive-theme"><div class="tabs"><input type="radio" name="group-cYXfJ" id="tab-PJExLmU" checked="checked"><label for="tab-PJExLmU">使用 venv 创建虚拟环境</label><input type="radio" name="group-cYXfJ" id="tab-ryyvxX1"><label for="tab-ryyvxX1">使用 conda 创建虚拟环境</label></div><div class="blocks"><div class="language-shell vp-adaptive-theme active"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">bash</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> scripts/create-python-env.sh</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -i</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> # -i 自动安装依赖</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">#zsh scripts/create-python-env.sh -i # zsh</span></span></code></pre></div><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">bash</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> scripts/create-python-env.sh</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -e</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> conda</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -i</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> # -i 自动安装依赖</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">#zsh scripts/create-python-env.sh -e conda -i # zsh</span></span></code></pre></div></div></div>`,41),h=[l];function n(p,d,k,c,o,r){return a(),i("div",null,h)}const u=s(t,[["render",n]]);export{F as __pageData,u as default};