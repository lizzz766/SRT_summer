import gym
import time
env = gym.make("BreakoutNoFrameskip-v4")

print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)


obs = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)
env.close()
'''
Note foe git
1.配置：默认编辑器(VIM,VSCode)

2.基础操作：
    Repository：一个根目录

    创建仓库：
    git init
    添加文件：
    git add
    不知道改啥了

3.文件分支
    



'''