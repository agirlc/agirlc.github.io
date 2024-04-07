---
title: "😀About"
layout: "about"
url: "about"
summary: about
hidemeta: true
comments: true
---


## About this blog
```python
class Blog:
    def __init__(self):
        self.name = "QíBlog"
        self.domain = "blog.laoqi.cc"
        self.host = "github"

    @property
    def welcome():
        return f"Welcome to {self.name}! Please bookmark my domain name{self.domain}"

    @property
    def function():
        return "Record and share some academic insights and experiences."
```


## About the author
```python
class Me:
    def __init__(self):
        self.name = "QíQingguo"
        self.want = [
            "Want to be free and easy", 
            "Want to live without worries", 
            "Want to do something cool",
            "Want to get my paper published",
            "Want to make money",
            "Want to have a good night's sleep",
            "Want to graduate smoothly",
            "Want farmers to have their land, and residents to have their homes",
            "Want world peace"
        ]
    
    @property
    def greeting():
        return f"Hello, I'm {self.name}, nice to meet you！"

    @property
    def interests():
        return "Artificial Intelligence and Robotics"

    @property
    def dream():
        import random
        return random.choice(self.want)

    @property
    def contact():
        return "Please contact me by email laoqi.cc at gmail.com"
```