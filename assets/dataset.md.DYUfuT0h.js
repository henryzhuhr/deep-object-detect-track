import{_ as s,a as i,b as a}from"./chunks/labelImg-rect-3.CpXWU20D.js";import{_ as t,c as l,o as e,a1 as n}from"./chunks/framework.BMAahRQD.js";const u=JSON.parse('{"title":"目标检测数据集制作","description":"","frontmatter":{"lastUpdated":true,"editLink":true,"footer":true,"outline":"deep"},"headers":[],"relativePath":"dataset.md","filePath":"dataset.md","lastUpdated":1717762479000}'),p={name:"dataset.md"},h=n(`<h1 id="目标检测数据集制作" tabindex="-1">目标检测数据集制作 <a class="header-anchor" href="#目标检测数据集制作" aria-label="Permalink to &quot;目标检测数据集制作&quot;">​</a></h1><h2 id="概要" tabindex="-1">概要 <a class="header-anchor" href="#概要" aria-label="Permalink to &quot;概要&quot;">​</a></h2><p>数据集的构建，应该使用“<strong>少量标注+半自动标注</strong>”的方式，即先标注少量数据，然后通过「<a href="#半自动标注">半自动标注</a>」的方式，快速标注其他数据，最后通过标注工具进行精细化调整。这样可以大大减少标注的工作量</p><p>因此建议先阅读完整个文档，再构建自己的数据集</p><h2 id="数据集采集和归档" tabindex="-1">数据集采集和归档 <a class="header-anchor" href="#数据集采集和归档" aria-label="Permalink to &quot;数据集采集和归档&quot;">​</a></h2><div class="tip custom-block"><p class="custom-block-title">TIP</p><p>数据是比代码更重要的资产，因此不要放置在项目内</p></div><p>将数据集放入如下目录</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">DATASET_DIR</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">/path/to/dataset</span></span></code></pre></div><blockquote><p>需要注意的是，数据集通常需要放置在项目外的路径，例如 <code>~/data</code> 或 <code>$HOME/data</code> （推荐）（win 下为 <code>$env:USERPROFILE/data</code>）。如果放置在项目内，导致编辑器对于项目的索引过大，会导致编辑器卡顿</p></blockquote><p>这里准备好了一个示例数据集，可以下载，并保存在 <code>~/data/drink</code> 目录下</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">wget</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -P</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ~/data</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> https://github.com/HenryZhuHR/deep-object-detect-track/releases/download/v1.0.0/drink.tar.bz2</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">tar</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -xf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ~/data/drink.tar.bz2</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -C</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ~/data</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">cp</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -r</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ~/data/drink</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ~/data/drink.unlabel</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">rm</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -rf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ~/data/drink.unlabel/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">**</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">*</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">.xml</span></span></code></pre></div><p>为了方便可以在项目内建立软链接，软链接不会影响编辑器进行索引，但是可以方便查看</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">ln</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -s</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ~/data</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> resource/data</span></span></code></pre></div><p>随后可以设置数据集目录为</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">DATASET_DIR</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">~/data/drink</span></span></code></pre></div><p>考虑到单张图像中可能出现不同类别的目标，因此数据集不一定需要按照类别进行划分，可以自定义划分，按照项目的需求任意归档数据集，但是请确保，每一张图像同级目录下有同名的<strong>标签文件</strong>。可以参考如下的几种目录结构构建自己的数据集：</p><div class="vp-code-group vp-adaptive-theme"><div class="tabs"><input type="radio" name="group-6IIDm" id="tab-Ug31TCX" checked="checked"><label for="tab-Ug31TCX">不划分子目录</label><input type="radio" name="group-6IIDm" id="tab-uC8WWaW"><label for="tab-uC8WWaW">按照类别划分的目录</label><input type="radio" name="group-6IIDm" id="tab-CoNA-S-"><label for="tab-CoNA-S-">自定义划分的目录结构</label></div><div class="blocks"><div class="language-shell vp-adaptive-theme active"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 这种目录结构适用于数据集不划分类别，直接存放在同一目录下</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">·</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">└──</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> /path/to/dataset</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    </span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    ├─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> file_1.jpg</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    ├─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> file_1.xml</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    └─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ...</span></span></code></pre></div><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 这种目录结构适用于数据集按照类别划分，每个类别一个目录</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">·</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">└──</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> /path/to/dataset</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    ├──</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> class_A</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">         </span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    │</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">   ├─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> file_A1.jpg</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    │</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">   ├─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> file_A1.xml</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    │</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">   └─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ...</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    └──</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> class_B</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       </span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">        ├─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> file_B1.jpg</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">        ├─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> file_B1.xml</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">        └─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ...</span></span></code></pre></div><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 这种目录结构可以根据项目的需求自定义划分，例如按照视频分割帧构建的数据集</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">·</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">└──</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> /path/to/dataset</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    ├──</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> video_A</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    │</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">   ├─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> file_A1.jpg</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    │</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">   ├─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> file_A1.xml</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    │</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">   └─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ...</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">    └──</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> video_B</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">        ├─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> file_B1.jpg</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">        ├─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> file_B1.xml</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">        └─</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ...</span></span></code></pre></div></div></div><h2 id="启动标注工具" tabindex="-1">启动标注工具 <a class="header-anchor" href="#启动标注工具" aria-label="Permalink to &quot;启动标注工具&quot;">​</a></h2><p>使用 labelImg 标注，安装并启动</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">pip</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> install</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> labelImg</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">labelImg</span></span></code></pre></div><p>在 Ubuntu 下启动后的界面如下（Windows 版本可能略有差异） <img src="`+s+'" alt="start"></p><ul><li>打开文件 : 标注单张图像（不推荐使用）</li><li><strong>打开目录</strong> : 打开数据集存放的目录，目录下应该是图像的位置</li><li><strong>改变存放目录</strong>: 标注文件 <code>.xml</code> 存放的目录</li><li>下一个图片:</li><li>上一个图像:</li><li><strong>验证图像</strong>: 验证标记无误，用于全部数据集标记完成后的检查工作</li><li><strong>保存</strong>: 保存标记结果，快捷键 <code>Ctrl+s</code></li><li><strong>数据集格式</strong>: 选择 <code>PascalVOC</code> ，后续再转化为 <code>YOLO</code></li></ul><p>点击 <code>创建区块</code> 创建一个矩形框，画出范围 <img src="'+i+'" alt="rect"></p><p>每个类别都有对应的颜色加以区分 <img src="'+a+`" alt="rect"></p><p>完成一张图片的标注后，点击 <code>下一个图片</code></p><ul><li>labelImg 快捷键</li></ul><table><thead><tr><th style="text-align:center;">快捷键</th><th style="text-align:center;">功能</th><th style="text-align:center;">快捷键</th><th style="text-align:center;">功能</th></tr></thead><tbody><tr><td style="text-align:center;">Ctrl+u</td><td style="text-align:center;">从目录加载所有图像</td><td style="text-align:center;">w</td><td style="text-align:center;">创建一个矩形框</td></tr><tr><td style="text-align:center;">Ctrl+R</td><td style="text-align:center;">更改默认注释目标目录</td><td style="text-align:center;">d</td><td style="text-align:center;">下一张图片</td></tr><tr><td style="text-align:center;">Ctrl+s</td><td style="text-align:center;">保存当前标注结果</td><td style="text-align:center;">a</td><td style="text-align:center;">上一张图片</td></tr><tr><td style="text-align:center;">Ctrl+d</td><td style="text-align:center;">复制当前标签和矩形框</td><td style="text-align:center;">del</td><td style="text-align:center;">删除选定的矩形框</td></tr><tr><td style="text-align:center;">space</td><td style="text-align:center;">将当前图像标记为已验证</td><td style="text-align:center;">Ctrl+</td><td style="text-align:center;">放大</td></tr><tr><td style="text-align:center;">↑→↓←</td><td style="text-align:center;">键盘箭头移动选定的矩形框</td><td style="text-align:center;">Ctrl–</td><td style="text-align:center;">缩小</td></tr></tbody></table><h2 id="数据处理" tabindex="-1">数据处理 <a class="header-anchor" href="#数据处理" aria-label="Permalink to &quot;数据处理&quot;">​</a></h2><p>运行脚本，生成同名目录，但是会带 <code>-organized</code> 后缀，例如</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">python</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> dataset-process.py</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -d</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ~/data/drink</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> # -d / --datadir</span></span></code></pre></div><blockquote><p>建议路径不要有最后的 <code>/</code> 或者 <code>\\</code>，否则可能会出现路径错误</p></blockquote><p>该脚本会自动递归地扫描目录 <code>~/data/drink</code> 下的所有 <code>.xml</code> 文件，并查看是否存在对应的 <code>.jpg</code> 文件</p><div class="tip custom-block"><p class="custom-block-title">TIP</p><p>因此，你可以不必担心目录结构，只需要确保每张图像有对应的标签文件即可，也不必担心没有标注完成的情况，脚本只处理以及标注完成的图像</p></div><p>生成的目录 <code>~/data/drink-organized</code> 用于数据集训练，并且该目录为 yolov5 中指定的数据集路径</p><h2 id="数据自定义处理" tabindex="-1">数据自定义处理 <a class="header-anchor" href="#数据自定义处理" aria-label="Permalink to &quot;数据自定义处理&quot;">​</a></h2><div class="tip custom-block"><p class="custom-block-title">TIP</p><p>通常来说不需要自定义处理，只需要遵循上述的规则即可快速创建数据集，但是如果需要，可以参考下面提供的接口</p></div><p>如果不需要完全遍历数据集、数据集自定义路径，则在 <code>get_all_label_files()</code> 函数中传入自定义的 <code>custom_get_all_files</code> 函数，以获取全部文件路径，该自定义函数可以参考 <code>default_get_all_files()</code></p><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> default_get_all_files</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(directory: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">str</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">):</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    file_paths: List[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">str</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> []</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> root, dirs, files </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> os.walk(directory):</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;"> file</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> files:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            file_paths.append(os.path.join(root, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">file</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> file_paths</span></span></code></pre></div><p>并且在调用的时候传入该参数</p><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark has-diff vp-code"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># -- get all label files, type: List[ImageLabel]</span></span>
<span class="line diff remove"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">label_file_list </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> get_all_label_files(args.datadir) </span></span>
<span class="line diff add"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">label_file_list </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> get_all_label_files(          </span></span>
<span class="line diff add"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    args.datadir,                               </span></span>
<span class="line diff add"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">    custom_get_all_files</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">default_get_all_files  </span></span>
<span class="line diff add"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)                                               </span></span></code></pre></div><h2 id="半自动标注" tabindex="-1">半自动标注 <a class="header-anchor" href="#半自动标注" aria-label="Permalink to &quot;半自动标注&quot;">​</a></h2><p>训练完少量数据集后可以使用 <code>auto_label.py</code> 脚本进行半自动标注</p><p>该脚本需要使用 OpenVINO 的模型进行推理，因此参考 <a href="./deploy.html#导出模型"><em>导出模型</em></a> 导出 openvino 模型，主要修改 <code>EXPORTED_MODEL_PATH</code> 为导出的模型路径和 数据集配置 <code>DATASET_CONFIG</code> ，然后注释导出命令 <code>python3 export.py ... --include onnx </code> 和 <code>python3 export.py ... --include engine </code></p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">bash</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> scripts/export-yolov5.sh</span></span></code></pre></div><p>查看 <code>auto_label.py</code> 参数设置，然后执行</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">python</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> auto_label.py</span></span></code></pre></div><p>如果已经被标注，会提示 <code>If you want to re-label, please delete it by &#39;rm ~/data/drink.unlabel/cola/cola_0000.xml&#39;</code> 的信息防止已经被标注的数据被覆盖，如果希望删除全部，可以用正则表达式</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">rm</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -rf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ~/data/drink.unlabel/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">**</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">*</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">.xml</span></span></code></pre></div><p>随后可以用 LabelImg 进行<strong>检查</strong>和精细化调整</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">labelImg</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ~/data/drink.unlabel</span></span></code></pre></div>`,50),d=[h];function k(r,c,o,g,F,y){return e(),l("div",null,d)}const b=t(p,[["render",k]]);export{u as __pageData,b as default};