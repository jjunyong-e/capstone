import random
import pandas as pd
# 각자 랜덤시드 입력
# 1 - 지현
# 2 - 찬규 (greedy)
# 3 - 승열
# 4 - 병현
# 5 - 채균
# 6 - 준용
# 7 - 찬규 (clustering)
seed_num = [1,2,3,4,5,6,7]

dist = []
for seed in seed_num:
    random.seed(seed)
    for i in range(20):
        arr = []
        for j in range(20):
            if i == j :
                arr.append(0) 
                # 출발지와 목적지가 같은경우 0 을 저장한다
            else:
                arr.append(random.randrange(1,50))
        dist.append(arr)
        # 가능한 거리는 1 에서 50 키로 까지로 설정하였다
        # 가는 방향 오는 방향에 따라 길이 다를수있으므로 dist[i][j] != dist [j][i]
    df = pd.DataFrame(dist)
    df.to_csv(f'거리배열_{seed}.csv')
# csv파일 의 index 와 header 의 숫자는 node에 붙인 번호를 의미함