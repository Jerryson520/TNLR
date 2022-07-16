### Question 1:（来自run_classifier.py的代码）
![image](https://raw.githubusercontent.com/Jerryson520/TNLR/main/Questions/run_classifier_part.png)  
这个地方好像说明的是如果不是single GPU，就只会output loss这个metric，因为其他metric (e.g. accuracy) 平均下来的结果没有那么好。
问题是这样的话output就不能用mnli-mm和mnli-m来表示了，因为它的metric是accuracy，那么我们应该用什么来表示它的performance?

### Question 2:
在classifier的代码中只有pretrained的model，没有看到classifier的代码，是不是tnlrv5包含了classifier的代码？

### Question 3:
![image](https://raw.githubusercontent.com/Jerryson520/TNLR/main/Questions/set_seed.png)  
run_classifier.py 中是不是需要加:
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


### Question 4:
sp.model用来干什么？