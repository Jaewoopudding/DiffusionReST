#!/usr/bin/env python3
import os
import re
import math
from collections import defaultdict

def extract_reward_from_filename(filename):
    """파일명에서 보상 점수를 추출합니다."""
    # 패턴: G{숫자}_rank{숫자}_idx{숫자}_{동물명}_{점수}.jpg
    pattern = r'G\d+_rank\d+_idx\d+_[a-zA-Z]+_(\d+\.\d+)\.jpg'
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    return None

def calculate_mean(values):
    """평균 계산"""
    return sum(values) / len(values) if values else 0

def calculate_std(values):
    """표준편차 계산"""
    if len(values) < 2:
        return 0
    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def calculate_eval_folder_means(base_path):
    """모든 eval 폴더의 보상 점수 평균을 계산합니다."""
    eval_folders = []
    
    # eval 폴더들을 찾기
    for item in os.listdir(base_path):
        if item.startswith('eval_') and item.endswith('-improve_4'):
            eval_folders.append(item)
    
    # 숫자 순으로 정렬
    eval_folders.sort(key=lambda x: int(x.split('_')[1].split('-')[0]))
    
    results = []
    
    for folder in eval_folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue
            
        rewards = []
        
        # 폴더 내 모든 jpg 파일 확인
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                reward = extract_reward_from_filename(filename)
                if reward is not None:
                    rewards.append(reward)
        
        if rewards:
            mean_reward = calculate_mean(rewards)
            results.append({
                'folder': folder,
                'mean_reward': mean_reward,
                'num_files': len(rewards),
                'min_reward': min(rewards),
                'max_reward': max(rewards),
                'std_reward': calculate_std(rewards)
            })
            print(f"{folder}: 평균 = {mean_reward:.4f}, 파일 수 = {len(rewards)}, 범위 = [{min(rewards):.4f}, {max(rewards):.4f}], 표준편차 = {calculate_std(rewards):.4f}")
        else:
            print(f"{folder}: 보상 점수를 가진 파일이 없습니다.")
    
    return results

if __name__ == "__main__":
    base_path = "/home/jaewoo/DiffusionReST/images/aesthetic_score_diff_B=64_M=4_KL=0.005_G=True:0.005_I=4_2025.08.07_4_improve_S=0"
    
    print("모든 eval 폴더의 보상 점수 평균 계산 중...")
    print("=" * 100)
    
    results = calculate_eval_folder_means(base_path)
    
    if results:
        print("\n" + "=" * 100)
        print("요약:")
        print(f"총 eval 폴더 수: {len(results)}")
        
        all_means = [r['mean_reward'] for r in results]
        print(f"전체 평균의 평균: {calculate_mean(all_means):.4f}")
        print(f"전체 평균의 표준편차: {calculate_std(all_means):.4f}")
        print(f"최고 평균: {max(all_means):.4f}")
        print(f"최저 평균: {min(all_means):.4f}")
        
        # 세대별 트렌드 확인
        print("\n세대별 보상 점수 트렌드:")
        for result in results:
            generation = int(result['folder'].split('_')[1].split('-')[0])
            print(f"세대 {generation:2d}: {result['mean_reward']:.4f}") 