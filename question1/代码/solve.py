"""用法：python solve.py 输入.json --output 输出.json"""
import argparse
import json
from pathlib import Path
from geometry import solve


def main():
    parser = argparse.ArgumentParser(description='第一问：交会定位区域直径与圆覆盖判定')
    parser.add_argument('input',type=Path,help='含 observations 数组的 JSON 文件')
    parser.add_argument('--output',type=Path,help='可选的结果 JSON 路径')
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8-sig'))
    result = solve(data['observations']).to_dict()
    payload = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload,encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
