# 如何贡献文档

我们非常欢迎用户向官网贡献优质的文档，PaddlePaddle的文档包括中英文两个部分。文档都是通过 cmake 驱动 sphinx 编译生成的。

PaddlePaddle.org工具是一个文档生成和查看器工具，用户可以使用它实现官网的中英文档编译过程，查看预览效果。

这篇教程将指导您如何在本地搭建与PaddlePaddle官网一致的网站，这样您就可以看到贡献的内容将如何显示在paddlepaddle官网上。

## 安装PaddlePaddle.org
### 1. Clone你希望更新或测试的相关仓库：
*如果您已经拥有了这些存储库的本地副本，请跳过此步骤*

可拉取的存储库有：

 - [Paddle](https://github.com/PaddlePaddle/Paddle)
 - [Book](https://github.com/PaddlePaddle/book)
 - [Models](https://github.com/PaddlePaddle/models)
 - [Mobile](https://github.com/PaddlePaddle/mobile)

您可以将这些本地副本放在电脑的任意目录下，稍后我们会在启动 PaddlePaddle.org时指定这些仓库的位置。

### 2. 在新目录下拉取 PaddlePaddle.org 并安装其依赖项
在此之前，请确认您的操作系统安装了python的依赖项

以ubuntu系统为例，运行：

``` 
sudo apt-get update && apt-get install -y python-dev build-essential
```

然后：
``` 
git clone https://github.com/PaddlePaddle/PaddlePaddle.org.git
cd PaddlePaddle.org/portal
# To install in a virtual environment.
# virtualenv venv; source venv/bin/activate
pip install -r requirements.txt
```
**可选项**：如果你希望实现中英网站转换，以改善PaddlePaddle.org，请安装[GNU gettext](https://www.gnu.org/software/gettext/)

### 3. 在本地运行 PaddlePaddle.org
添加您希望加载和构建内容的目录列表(选项包括：--paddle，--book，--models，--mobile)

运行：
``` 
./runserver --paddle <path_to_paddle_dir> --book <path_to_book_dir>
```

**注意：**  `<pathe_to_paddle_dir>`为第一步中paddle副本在您本机的存储地址，并且对于 --paddle目录，您可以指向特定的API版本目录（例如：`<path to Paddle>/doc/fluid` or `<path to Paddle>v2`)

然后：
	
打开浏览器并导航到http://localhost:8000。

*网站可能需要几秒钟才能成功加载，因为构建需要一定的时间。*

## 如何书写文档

PaddlePaddle文档使用 [sphinx](http://www.sphinx-doc.org/en/1.4.8/) 自动生成，用户可以参考sphinx教程进行书写。

## 贡献新的内容
所有存储库都支持[Markdown](https://guides.github.com/features/mastering-markdown/) (GitHub风格)格式的内容贡献。在Paddle仓库中，同时也支持[reStructured Text](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)格式。

在完成安装步骤后，您还需要完成下列操作：

 - 在你开始写作之前，我们建议你回顾一下这些关于贡献内容的指南
 - 创建一个新的` .md` 文件（在 paddle repo 中可以创建 `.rst` 文件）或者在您当前操作的仓库中修改已存在的文章
 - 查看浏览器中的更改，请单击右上角的Refresh Content
 - 将修改的文档添加到菜单或更改其在菜单上的位置，请单击页面左侧菜单顶部的Edit menu按钮，打开菜单编辑器。
 
##贡献或修改Python API
在build了新的pybind目标并测试了新的Python API之后，您可以继续测试文档字符串和注释的显示方式:

- 我们建议回顾这些API文档贡献指南
- 确保构建的Python目录(包含 Paddle )在您运行`./runserver`的Python路径中可用。
- 在要更新的特定“API”页面上，单击右上角的Refresh Content。
- 将修改的API添加到菜单或更改其在菜单上的位置，请单击页面左侧菜单顶部的Edit menu按钮，打开菜单编辑器。

## 帮助改进预览工具
我们非常欢迎您对平台和支持内容的各个方面做出贡献，以便更好地呈现这些内容。您可以fork或clone这个存储库，或者提出问题并提供反馈，以及在issues上提交bug信息。详细内容请参考[开发指南](https://github.com/PaddlePaddle/PaddlePaddle.org/blob/develop/DEVELOPING.md)。

## 版权和许可
PaddlePaddle.org在Apache-2.0的许可下提供。
