import numpy as np
import matplotlib.pyplot as plt
import random

# 读取tsp文件
def read_tsp_file(file_path):
    city_coord=[]
    reading_coords = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # 跳过空行和注释
            if not line:
                continue

            # 开始读取坐标部分
            if line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue

            # 结束读取
            if line == "EOF":
                break

            # 读取坐标数据
            if reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    city_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    city_coord.append({"id": city_id, "x": x, "y": y})

    return city_coord

class GA:
    def __init__(self,city_coord,
                 pop_num=100,
                 generation_num=100,
                 greedy_init_percent=0.20,
                 mutation_rate=0.95,
                 selection_mod=0,
                 crossover_mod=0,
                 mutation_mod=0):
        # TSP问题参数
        self.city_num=len(city_coord)
        self.city_coord=city_coord
        # 预计算所有城市之间的距离，降低计算时间
        self.dist_matrix=self.cal_dist_matrix()

        # 遗传算法参数
        self.pop_num=pop_num  # 种群数量
        self.generation_num=generation_num  # 迭代次数
        self.mutation_rate=mutation_rate    # 最大变异概率

        self.greedy_init_percent=greedy_init_percent  # 用贪心算法生成初始解的占比

        # 方法选择
        self.selection_mod = selection_mod  # 选择的方法。默认值为0，表示锦标赛选择法
        self.crossover_mod = crossover_mod  # 交叉的方法。默认值为0，表示顺序交叉法
        self.mutation_mod = mutation_mod    # 变异的方法。默认值为0，表示混合变异法

        # 数据存储
        self.pops=np.zeros((self.pop_num,self.city_num),dtype=int)   # 种群列表
        self.total_dist = np.zeros(self.pop_num)  # 总路径长度列表

        # 每一代最短距离
        self.shortest_dist =[]
        # 每一代最优种群
        self.best_pops =[]

    def cal_distance(self,x1,y1,x2,y2):
        return np.sqrt((x1-x2)**2+(y1-y2)**2)

    def cal_dist_matrix(self):
        dist_matrix=np.zeros((self.city_num,self.city_num))
        city_coord=self.city_coord
        for i in range(self.city_num):
            for j in range(i+1,self.city_num):
                dist_matrix[i,j]=self.cal_distance(city_coord[i]['x'],city_coord[i]['y'],
                                            city_coord[j]['x'],city_coord[j]['y'])
                dist_matrix[j,i]=dist_matrix[i, j]
        return dist_matrix

    '''--------------初始化种群initialization-----------------'''
    def greedy_init(self, start_city=0):
        # 贪心算法初始化
        path = [start_city]
        unvisited = set(range(self.city_num)) - {start_city}
        while unvisited:
            current = path[-1]
            next_city = min(unvisited, key=lambda x: self.dist_matrix[current, x])
            path.append(next_city)
            unvisited.remove(next_city)
        return path

    def random_init(self):
        # 初始化种群
        chromosome = np.random.permutation(self.city_num)
        return chromosome

    def init_pops(self):
        # 贪心初始化解
        greedy_num = int(self.pop_num * self.greedy_init_percent)
        for i in range(greedy_num):
            # 随机选择一个开始城市
            start_city = np.random.randint(0, self.city_num)
            # 调用贪心算法
            self.pops[i, :] = self.greedy_init(start_city)

        # 剩余种群随机生成
        for i in range(greedy_num, self.pop_num):
            self.pops[i, :] = self.random_init()

    '''--------------目标函数（总距离）total_dist-----------------'''
    def cal_one_path_total_dist(self,one_path):
        # 计算一条路径的距离
        total_dist=0
        for i in range(self.city_num-1):
            total_dist+=self.dist_matrix[one_path[i],one_path[i+1]]
        total_dist+=self.dist_matrix[one_path[-1],one_path[0]]
        return total_dist

    def cal_total_dist(self):
        # 计算所有路径的总路径长度
        for i in range(self.pop_num):
            self.total_dist[i]=self.cal_one_path_total_dist(self.pops[i,:])

        # 记录最短路径和对应的最优种群
        shortest_dist_value=min(self.total_dist)
        self.shortest_dist.append(shortest_dist_value)

        shortest_dist_idx=np.argmin(self.total_dist)
        shortest_dist_path=self.pops[shortest_dist_idx,:]
        self.best_pops.append(shortest_dist_path)
        return shortest_dist_value,shortest_dist_path

    '''--------------适应度fitness-----------------'''
    def cal_fitness(self):
        '''计算适应度
        适应度=1/总路径长度
        '''
        fitness = np.zeros(self.pop_num)
        for i in range(self.pop_num):
            fitness[i] = 1 / self.total_dist[i]
        return fitness

    '''--------------选择selection-----------------'''
    def seletion_roulette(self,fitness):
        '''轮盘赌选择
        缺点：容易早熟,实验表明确实如此
        '''
        fitness_sum=np.sum(fitness)
        prob=[item/fitness_sum for item in fitness]
        prob=np.array(prob)
        prob_cumsum=np.cumsum(prob)

        # 生成两个0~1之间的随机数
        parents= []  # 存储父母
        for i in range(2):  # 生成随机数来进行轮盘赌，确定父母
            rand_num=random.uniform(0,1)

            # 看这个随机数落在哪里
            for j in range(len(prob_cumsum)):
                if j==0:
                    if rand_num<prob_cumsum[0]:
                        parents.append(self.pops[j,:])
                else:
                    if prob_cumsum[j-1]<=rand_num and rand_num<prob_cumsum[j]:
                        parents.append(self.pops[j])
        return parents

    def selection_tournament(self,k=3):
        '''锦标赛选择
        效果较好'''
        candidates=np.random.choice(self.pop_num,k)
        winner=candidates[np.argmin(self.total_dist[candidates])]
        return self.pops[winner]

    def selection(self, fitness):
        if self.selection_mod == 1:
            # 轮盘赌选择
            parents = self.seletion_roulette(fitness)
        else:
            # 锦标赛选择
            parents = []
            parents.append(self.selection_tournament())
            parents.append(self.selection_tournament())
        return parents

    '''--------------交叉crossover-----------------'''
    def crossover_OX(self, parents):
        # 顺序交叉（OX）
        parent1, parent2 = parents[0], parents[1]
        size = len(parent1)
        # 生成交叉点
        index1, index2 = sorted(np.random.choice(size, 2, replace=False))
        # 从 parent2 中截取片段
        sec2 = parent2[index1:index2]
        # 构建子代1：保留 parent1 中不在 sec2 的部分，插入 sec2
        mask = np.isin(parent1, sec2, invert=True)
        remaining = parent1[mask]
        son1 = np.concatenate([remaining[:index1], sec2, remaining[index1:]])
        # 构建子代2
        sec1 = parent1[index1:index2]
        mask = np.isin(parent2, sec1, invert=True)
        remaining = parent2[mask]
        son2 = np.concatenate([remaining[:index1], sec1, remaining[index1:]])
        return son1, son2


    def crossover_PMX(self,parents):
        # 部分映射交叉(PMX)
        parent1, parent2 = parents[0], parents[1]
        size = len(parent1)
        # 1. 随机选择两个交叉点
        index1, index2 = sorted(np.random.choice(size, 2, replace=False))

        # 2. 初始化子代
        son1 = np.full(size, -1, dtype=int)
        son2 = np.full(size, -1, dtype=int)

        # 3. 交换中间片段
        son1[index1:index2] = parent2[index1:index2]
        son2[index1:index2] = parent1[index1:index2]

        # 4. 建立映射关系
        mapping1 = {parent2[i]: parent1[i] for i in range(index1, index2)}
        mapping2 = {parent1[i]: parent2[i] for i in range(index1, index2)}

        # 5. 填充子代剩余位置
        for i in list(range(0, index1)) + list(range(index2, size)):
            # 处理 son1
            value = parent1[i]
            while value in son1[index1:index2]:  # 如果值已存在于中间片段
                value = mapping1[value]  # 根据映射替换
            son1[i] = value

            # 处理 son2
            value = parent2[i]
            while value in son2[index1:index2]:
                value = mapping2[value]
            son2[i] = value

        return son1, son2

    def crossover(self, parents):
        if self.crossover_mod==1:
            # 部分映射交叉
            son1, son2 = self.crossover_PMX(parents)
        else:
            # 顺序交叉
            son1, son2 = self.crossover_OX(parents)
        return son1, son2

    '''--------------变异mutation-----------------'''
    def mutation_swap(self):
        '''
        交换变异
        生成两个1~city_num之间的随机整数，分别代表城市的下标，
        且保证不相等
        然后交换这两个下标的城市编号
        :return:
        '''
        for i in range(self.pop_num):
            rate=random.uniform(0,1)
            if rate>self.mutation_rate:
                # 如果随机数大于变异概率，则不变异
                continue

            index1=random.randint(0,self.city_num-1) # 注意：右边的值也能取到，所以是要-1
            index2=random.randint(0,self.city_num-1)
            # 保证两个下标不相等
            while(index1 == index2):
                index2 = random.randint(0, self.city_num-1)

            # 交换
            self.pops[i,index1],self.pops[i, index2]=self.pops[i,index2],self.pops[i, index1]

    def mutation_inverse(self):
        '''倒置变异
        随机选择一段路径并反转顺序
        :return:
        '''
        for i in range(self.pop_num):
            rate=random.uniform(0,1)
            if rate>self.mutation_rate:
                # 如果随机数大于变异概率，则不变异
                continue

            # 生成两个不同的随机索引（范围0~city_num-1），并按升序排列
            index1,index2=sorted(np.random.choice(self.city_num,2,replace=False))
            self.pops[index1:index2]=self.pops[index1:index2][::-1]

    def two_opt(self, path):
        '''2-opt算法
        效果较好
        '''
        improved = True
        while improved:
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path)):
                    if j - i == 1:
                        # 已经相邻的两个城市不需要交换，因为即使交换了也没有变化
                        continue
                    # 计算交换后的距离变化
                    old_dist = (self.dist_matrix[path[i - 1], path[i]] +
                                self.dist_matrix[path[j - 1], path[j]])
                    new_dist = (self.dist_matrix[path[i - 1], path[j - 1]] +
                                self.dist_matrix[path[i], path[j]])
                    if new_dist < old_dist:
                        path[i:j] = path[j - 1:i - 1:-1]  # 反转片段
                        improved = True
        return path
    def mutation(self,t):
        if self.mutation_mod==1:
            self.mutation_inverse()
        elif self.mutation_mod==2:
            self.mutation_swap()
        else:
            self.mutation_inverse()
            if t % (self.generation_num // 20) == 0:
                self.mutation_swap()

        if t%(self.generation_num // 20)==0:
            # 每隔一段时间进行一次2-opt算法优化
            for i in range(self.pop_num):
                self.pops[i]=self.two_opt(self.pops[i])

    '''---------------其他辅助函数----------------'''
    def choose_best(self):
        shortest_dist_value=min(self.shortest_dist)
        shortest_dist_id=self.shortest_dist.index(shortest_dist_value)
        shortest_dist_path=self.best_pops[shortest_dist_id]
        return shortest_dist_value,shortest_dist_path

    # 画路径图
    def draw_path(self,one_path):
        x, y = [], []
        for i in range(self.city_num):
            one_city_coord=self.city_coord[one_path[i]]
            x.append( one_city_coord['x'] )
            y.append(one_city_coord['y'])
        x.append(x[0])
        y.append(y[0])
        plt.plot(x, y, 'r-', alpha=0.8, linewidth=2.2,
                 marker='o',markerfacecolor='blue',markeredgecolor='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def draw_shortest_dist_value_by_generation(self):
        generation=np.arange(0,self.generation_num+1)
        plt.plot(generation,self.shortest_dist)
        plt.ylabel("Shortest distance (m)")
        plt.xlabel("Generation")
        plt.show()

    def print_shortest_dist_path(self,shortest_dist_path):
        for city in shortest_dist_path:
            print(f"{city + 1}->",end="")
        print(shortest_dist_path[0]+1)

    '''-----------------选择和交叉----------------'''
    def selection_crossover(self):
        # 先计算适应度
        fitness = self.cal_fitness()
        new_pops=[]
        # 保留前 10% 的精英个体
        elite_num = int(self.pop_num * 0.10)
        elite_indices = np.argsort(self.total_dist)[:elite_num]
        for i in elite_indices:
            new_pops.append(self.pops[i])

        # 生成剩余个体
        for i in range((self.pop_num - elite_num) // 2):
            # 选择
            parents = self.selection(fitness)
            # 交叉
            son1, son2 = self.crossover(parents)

            new_pops.append(son1)
            new_pops.append(son2)

        self.pops = np.array(new_pops)

    '''------------------主函数--------------------'''
    def solve_GA(self):
        self.init_pops()  # 初始化种群
        print("--------初代--------")
        shortest_dist_value, shortest_dist_path = self.cal_total_dist()
        print("初代最短路径长度：", shortest_dist_value)
        print("初代最短路径:", shortest_dist_path)

        # 主循环，迭代
        for t in range(self.generation_num):
            print(f"----------第{t+1}次迭代----------")
            # 选择、交叉
            self.selection_crossover()
            # 变异
            self.mutation(t)

            shortest_dist_value,shortest_dist_path=self.cal_total_dist()
            print(f"第{t+1}代最短路径长度：", shortest_dist_value)
            print(f"第{t+1}代最短路径:", shortest_dist_path)

        shortest_dist_value,shortest_dist_path=self.choose_best()
        print("最短路径长度：",shortest_dist_value)
        print("最短路径：", shortest_dist_path)
        self.print_shortest_dist_path(shortest_dist_path)

        # 绘制优化过程图
        self.draw_shortest_dist_value_by_generation()
        # 绘制最短路径
        self.draw_path(shortest_dist_path)


if __name__=="__main__":
    city_coord = read_tsp_file("./tsp_data/wi29.tsp")
    # city_coord = read_tsp_file("./tsp_data/qa194.tsp")
    # city_coord=read_tsp_file("./tsp_data/dj38.tsp")
    # city_coord = read_tsp_file("tsp_data/zi929.tsp")
    ga=GA(city_coord,pop_num=100,generation_num=1500,greedy_init_percent=0.5,mutation_rate=0.95)
    ga.solve_GA()

    