**背景：**

  基于用户对中文API的强烈需求，11月份启动了API翻译工作。翻译时发现部分英文API仍需优化，主要问题有：
  
  1） 文字描述较抽象，用户很难根据doc了解此API的所有信息
  2） 参数说明有误，或过于简略
  3） 缺少使用example
  4） 格式混乱
  
  其中，将需要研发老师帮助修复的API统计出来，记录如下：
  
## 需要 RD 老师帮忙补充内容

1. [_switch_scope](https://github.com/PaddlePaddle/Paddle/blob/0abfbd1c41e6d558f76252854d4d78bef581b720/python/paddle/fluid/executor.py#L39):
   
   没有文档内容，是否在官网上去掉显示
   ![]()
   
2. [staticRNN](https://github.com/PaddlePaddle/Paddle/blob/61b4812f2fe8c0591323f9d60db69231d8933322/python/paddle/fluid/layers/control_flow.py#L429)

    参数说明过于简略，缺少example
    
    ![](https://user-images.githubusercontent.com/31891223/48390389-bd9c6f00-e73c-11e8-988a-bc8a8d34307d.png)
    
3. [shuffle](https://github.com/PaddlePaddle/Paddle/blob/d3aed98d86af288a15047eee75ed8c939b0babb6/python/paddle/fluid/layers/io.py#L946)
    
    内容过于简略，缺少参数说明和example
    ![](https://user-images.githubusercontent.com/31891223/48536300-20326e00-e8e9-11e8-8a08-5f64fbe656c1.png)
    
4. [DynamicRNN](https://github.com/PaddlePaddle/Paddle/blob/61b4812f2fe8c0591323f9d60db69231d8933322/python/paddle/fluid/layers/control_flow.py#L1542)
 
    缺少部分参数说明，且函数中没有找到这几个参数
    ![](https://user-images.githubusercontent.com/31891223/48042193-7c93e000-e1bb-11e8-9a47-5780c7de4a48.png)
    
5. [ctc_greedy_decoder](https://github.com/PaddlePaddle/Paddle/blob/d3aed98d86af288a15047eee75ed8c939b0babb6/python/paddle/fluid/layers/nn.py#L4116)
    
    看了案例感觉不是很懂 output代表什么，能否优化文字描述呢？
    ![](https://user-images.githubusercontent.com/31891223/48394112-34416880-e74d-11e8-865b-d1959be37507.png)
    
6. [resize_bilinead](https://github.com/PaddlePaddle/Paddle/blob/d3aed98d86af288a15047eee75ed8c939b0babb6/python/paddle/fluid/layers/nn.py#L5801)
    
    缺少example
    ![](https://user-images.githubusercontent.com/31891223/48394360-3c4dd800-e74e-11e8-8932-a195de9eb012.png)
    
7. ['elu','relu6','pow','stanh','hard_sigmoid','swish', 'prelu','brelu','leaky_relu','soft_relu'](https://github.com/PaddlePaddle/Paddle/blob/d3aed98d86af288a15047eee75ed8c939b0babb6/python/paddle/fluid/layers/nn.py#L6659)

   这些API，缺少example
    
8. ['uniform_random_batch_size_like','gaussian_random','sampling_id','gaussian_random_batch_size_like','sum','slice', 'shape','logical_and','logical_or','logical_xor','logical_not','clip','clip_by_norm','mean','mul','sigmoid_cross_entropy_with_logits','maxout'](https://github.com/PaddlePaddle/Paddle/blob/d3aed98d86af288a15047eee75ed8c939b0babb6/python/paddle/fluid/layers/nn.py#L7215)
    
   这些API，缺少example
   
9. [append_LARS](https://github.com/PaddlePaddle/Paddle/blob/26200f2e420566cba3112ee725197a1c12c8682b/python/paddle/fluid/layers/learning_rate_scheduler.py#L310)

   目前内容有些混乱，需要修改一下
   ![](https://user-images.githubusercontent.com/31891223/48396492-91d9b300-e755-11e8-8e47-aba1bc2b804a.png)


10. [polygon_box_transform](https://github.com/PaddlePaddle/Paddle/blob/4a55fb5f5b8d177a61133afe7210561f796a7e32/python/paddle/fluid/layers/detection.py#L383)
    
    缺少example
    
11. [force_init_on_cpu](https://github.com/PaddlePaddle/Paddle/blob/cffad81c1a4e787d7495bf6e5e3d35f08ebae967/python/paddle/fluid/initializer.py#L32)

    文字说明需要补充，该API如何应用呢？
    ![](https://user-images.githubusercontent.com/31891223/48398529-f5ff7580-e75b-11e8-9bcb-516a166db446.png)
