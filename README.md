# so-vits-svc4.0-tts-package
请至[bilibili@羽毛布団](https://www.bilibili.com/video/BV1H24y187Ko/?spm_id_from=333.337.top_right_bar_window_history.content.click&vd_source=da80b8c27bdce8110c01ddca1da17289)处下载[整合包](https://docs.qq.com/doc/DUWdxS1ZaV29vZnlV)

后下载[edge-tts](https://github.com/rany2/edge-tts)解压至你的整合包文件夹中的\workenv\Lib文件夹（看来行不通，建议跟着我的专栏走）

最后下载app.py文件替换原路径里的同名文件

## 小改动

&emsp;增加了对Svc_768l12的适配，可能是由于羽毛大佬的更新，现有的app.py不适用于768l12模型，这里我对照两个文件，改成app_768l12.py。使用的时候将app_768l12.py重命名为app.py替换原路径里的同名文件即可。
