# test debug


sentence="在基于负采样的模型中"
context_size=2
for i in range(1, len(sentence)-1):
                # 模型输入：当前词
                w = sentence[i]
                # 模型输出：一定窗口大小内的上下文
                left_context_index = max(0, i - context_size)
                right_context_index = min(len(sentence), i + context_size)
                print(left_context_index,right_context_index)
                context = sentence[left_context_index:i] + sentence[i+1:right_context_index+1]
                print(context)