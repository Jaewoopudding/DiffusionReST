import heapq
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, priority: str):
        """
        우선순위 기반 리플레이 버퍼 초기화
        :param capacity: 저장할 수 있는 최대 경험 개수 (top k 샘플)
        :param priority: 경험 dict 내 우선순위 값을 나타내는 키 이름
        """
        self.capacity = capacity
        self.priority = priority
        # heapq를 사용하여 (priority, experience) 튜플을 저장합니다.
        # min-heap이므로 가장 낮은 priority 값이 루트에 위치합니다.
        self.buffer = []

    def push(self, experience):
        """
        새로운 경험 혹은 경험 리스트를 버퍼에 저장합니다.
        각 경험은 반드시 self.priority 키를 포함해야 합니다.
        버퍼가 꽉 찬 경우, 새 경험의 우선순위가 현재 최소 우선순위보다 높으면 교체합니다.
        
        :param experience: 단일 경험(dict) 또는 경험들의 리스트
        """
        if isinstance(experience, list):
            for exp in experience:
                self._push_single(exp)
        elif isinstance(experience, dict):
            self._push_single(experience)
        else:
            raise TypeError("experience는 dict 또는 dict의 리스트여야 합니다.")

    def _push_single(self, exp: dict):
        if self.priority not in exp:
            raise ValueError(f"경험에 '{self.priority}' key가 필요합니다.")
        
        prio_value = exp[self.priority]
        # 버퍼가 아직 가득 차지 않은 경우, 그냥 추가합니다.
        if len(self.buffer) < self.capacity:
            heapq.heappush(self.buffer, (prio_value, exp))
        else:
            # 버퍼가 꽉 찬 경우, 현재 최소 우선순위와 비교하여 교체합니다.
            if prio_value > self.buffer[0][0]:
                heapq.heapreplace(self.buffer, (prio_value, exp))

    def sample(self, batch_size: int, target_threshold: dict):
        """
        버퍼에서 무작위로 미니배치 샘플을 반환합니다.
        추가적으로 target_threshold 조건을 만족하는 경험만 샘플합니다.

        :param batch_size: 샘플링할 경험의 개수
        :param target_threshold: {key: threshold} 조건을 만족해야 하는 추가 조건
        :return: 샘플된 경험 리스트 (각 경험은 dict)
        """
        # 조건을 만족하는 경험만 필터링합니다.
        filtered_experiences = [exp for _, exp in self.buffer
                                if all(exp.get(key, float('-inf')) >= threshold
                                    for key, threshold in target_threshold.items())]

        if batch_size > len(filtered_experiences):
            raise ValueError("배치 사이즈가 조건을 만족하는 경험 수보다 많습니다.")

        samples = random.sample(filtered_experiences, batch_size)
        return samples


    def cutoff(self, threshold: float):
        """
        버퍼에서 주어진 임계값 이하의 우선순위를 가진 경험을 제거합니다.
        :param threshold: 임계값 (해당 임계값 이하의 우선순위를 가진 경험은 제거됩니다)
        """
        self.buffer = [(p, exp) for p, exp in self.buffer if p > threshold]
        # 힙 속성을 유지하기 위해 heapify를 수행합니다.
        heapq.heapify(self.buffer)

    def __len__(self):
        """
        버퍼에 저장된 현재 경험의 개수를 반환합니다.
        """
        return len(self.buffer)