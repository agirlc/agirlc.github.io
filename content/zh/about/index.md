---
title: "😀关于"
layout: "about"
url: "about"
summary: about
hidemeta: true
comments: true
---

## 关于本博客
```python
class Blog:
    def __init__(self):
        self.name = "QíBlog"
        self.domain = "blog.laoqi.cc"
        self.host = "github"

    @property
    def welcome():
        return f"欢迎来到{self.name}! 请收藏我的域名{self.domain}"

    @property
    def function():
        return "记录与分享一些学术上的见闻与心得"
```


## 关于作者
```python
class Me:
    def __init__(self):
        self.name = "QíQingguo"
        self.want = [
            "想要自由自在", 
            "想要无忧无虑", 
            "想要做一些很酷的事情",
            "想要论文能发表",
            "想要赚钱",
            "想要睡个好觉",
            "想要顺利毕业",
            "想要耕者有其田，居者有其屋",
            "想要世界和平"
        ]
    
    @property
    def greeting():
        return f"你好，我是{self.name}，很高兴能认识你！"

    @property
    def interests():
        return "人工智能与机器人"

    @property
    def dream():
        import random
        return random.choice(self.want)
    
    @property
    def contact():
        return "请通过邮件联系我 laoqi.cc at gmail.com"
```