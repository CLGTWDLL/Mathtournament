"""输入首次观测及算法配置，输出第二检测点、候选点排名或多步策略。"""
import argparse
import json
from pathlib import Path
from model import Problem,select_point


def main():
    parser = argparse.ArgumentParser(description='第二问：全向目标的第二检测点选择')
    parser.add_argument('input',type=Path,help='输入 JSON 文件')
    parser.add_argument('--output',type=Path,required=True,help='输出 JSON 文件')
    parser.add_argument('--mode',choices=['auto','hybrid','multistep','continuous','discrete'],default='auto',help='选点模式')
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding='utf-8-sig'))
    problem = Problem(**source['first_observation'])
    mode = args.mode
    if mode == 'auto':
        if 'continuous_region' in source:
            mode = 'continuous'
        elif source.get('mode') == 'discrete':
            mode = 'discrete'
        elif source.get('mode') == 'multistep':
            mode = 'multistep'
        else:
            mode = 'hybrid'
    if mode == 'hybrid':
        from hybrid_q2 import select_hybrid
        settings = source.get('hybrid_settings')
        if not settings:
            raise ValueError('融合模式必须在 hybrid_settings 中给出计算配置')
        result = select_hybrid(problem, **settings)
        result['interpretation'] = '不设性能硬阈值；在完整耗时与连续误差的帕累托前沿上选拐点。'
    elif mode == 'multistep':
        from rollout_q2 import select_second_point
        result = select_second_point(
            (problem.x, problem.y),
            problem.angle_deg,
            max_extra_measurements=source.get('multistep_settings', {}).get('max_extra_measurements', 5),
            weight=source.get('multistep_settings', {}).get('weight', 1.0),
        )
        result['problem'] = source['first_observation']
        result['interpretation'] = '多步 rollout 策略选点；后续动作仍应使用同一公开反馈策略逐步执行。'
    elif mode == 'continuous':
        from continuous_region import build_region,recommend_in_region
        result = build_region(problem,**source['continuous_region'])
        result['recommendation'] = recommend_in_region(result,**source.get('ranking_settings',{}))
    else:
        result = select_point(problem,**source.get('settings',{}))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if mode == 'hybrid':
        print('融合选点：',result['selected']['point'])
        print('完整rollout估计耗时：',result['selected'].get('rollout_mean_s'))
        print('连续直径证书上界：',result['selected']['continuous_metric'].get('diameter_upper_m'))
    elif mode == 'multistep':
        print('多步策略第二点：',result['world_point'])
        print('局部第二点：',result['local_point'])
        print('选点来源：',result['method'])
    elif mode == 'continuous':
        print('认证矩形数：',result['cell_count'],'；区域面积（平方米）：',result['area_m2'])
        print(json.dumps(result['recommendation']['best'] if result['recommendation'] else None,ensure_ascii=False,indent=2))
    else:
        print(json.dumps(result['best'],ensure_ascii=False,indent=2))
        print('已评估候选点数：',result['candidate_count'])


if __name__ == '__main__':
    main()
