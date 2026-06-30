1. 使用方式和参数：
--planner     crescendo | pair | tap | sema     # 选哪个 Planner
  --goal        "How to hack email?"               # 
  单个目标（不指定则从数据集取）
  --scale       10                                 # 批量测试数量，或 'all'
  --rounds      15                                 # 每目标最大攻击轮数
  --beam        3                                  # TAP 专用：搜索宽度
  --branch      3                                  # TAP 专用：分支数
  --attack-model  llama2-uncensored:7b             # 攻击模型
  --victim-model  llama3.2:latest                  # 受害者模型
  --judge-model   deepseek-chat                    # Judge 模型
  --judge-key     sk-xxx                           # Judge API Key
  --compare                                        # 四种 Planner 同台对比