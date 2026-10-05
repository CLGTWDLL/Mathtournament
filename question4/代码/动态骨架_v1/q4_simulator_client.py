"""第四问官方模拟器串行客户端：驱动 q4_dynamic_policy.Policy 跑一局。

用法（先在模拟器里启动一局，等接口就绪，再运行本脚本）：

  python -B q4_simulator_client.py --run --robot-id <参赛队号>

演练与正式测试使用同一套接口，本脚本不区分两者，由你在模拟器界面选择。
导入本模块不会连接任何东西；只有加 --run 才会发请求。
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
from pathlib import Path
import sys
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from q4_dynamic_policy import Policy

ENDPOINTS = ('/enter', '/exit', '/measure', '/clear')


class ProtocolError(RuntimeError):
    pass


def http_transport(url, data, timeout):
    request = Request(url, data=data,
                      headers={'Content-Type': 'application/json; charset=utf-8'},
                      method='POST')
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode('utf-8'))


class Client:
    def __init__(self, robot_id, base_url='http://127.0.0.1:2026',
                 transport=http_transport, log=None, timeout=5.,
                 retries=3, retry_delay=.3):
        if not 1 <= len(robot_id.encode('utf-8')) <= 64 or any(
                unicodedata.category(c) in ('Cc', 'Cf') for c in robot_id):
            raise ValueError('robot_id 必须是参赛队号，1-64 字节且无控制字符')
        parsed = urlsplit(base_url)
        if (parsed.scheme != 'http'
                or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment
                or parsed.username):
            raise ValueError('base_url 必须是本机回环地址，且不带路径或查询串')
        self.base = base_url.rstrip('/')
        self.robot_id = robot_id
        self.transport = transport
        self.log = log
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.deadline = None
        self.virtual_time_s = 0.
        self.accepted_actions = 0

    def remaining(self):
        if self.deadline is None:
            return math.inf
        return self.deadline - time.monotonic()

    def record(self, value):
        if self.log is not None:
            self.log.write(json.dumps(value, ensure_ascii=False,
                                      allow_nan=False) + '\n')
            self.log.flush()

    def post(self, path, action=None):
        if path not in ENDPOINTS:
            raise ValueError('未知接口')
        payload = dict(arena_id='default', robot_id=self.robot_id,
                       request_id=uuid.uuid4().hex)
        if path in ('/measure', '/clear'):
            if action is None:
                raise ValueError('缺少动作参数')
            channel = action['channel']
            q = action['position']
            if not isinstance(channel, int) or isinstance(channel, bool) \
                    or not 1 <= channel <= 20:
                raise ValueError('频道必须是 1..20 的整数')
            if len(q) != 2 or not all(math.isfinite(x) and abs(x) <= 2000000
                                      for x in q):
                raise ValueError('坐标非法')
            payload.update(channel=channel,
                           position=dict(x=float(q[0]), y=float(q[1])))
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')
        for attempt in range(self.retries + 1):
            available = self.remaining()
            if available <= 0:
                raise TimeoutError('本局现实时间已耗尽')
            self.record(dict(event='request', path=path, payload=payload,
                             attempt=attempt, timestamp=time.time()))
            try:
                status, response = self.transport(
                    self.base + path, data, min(self.timeout, available))
            except (URLError, TimeoutError, ConnectionError,
                    http.client.RemoteDisconnected,
                    http.client.IncompleteRead) as error:
                self.record(dict(event='transport_error',
                                 request_id=payload['request_id'],
                                 error=str(error)))
                if attempt == self.retries:
                    raise
                time.sleep(min(self.retry_delay, max(0., available)))
                continue
            self.record(dict(event='response', path=path,
                             request_id=payload['request_id'],
                             status=status, body=response))
            if status != 200 or response.get('accepted') is not True:
                raise ProtocolError(f'{path} 被拒绝：HTTP {status}, '
                                    f'accepted={response.get("accepted")!r}')
            vt = response.get('virtual_time_s')
            if not isinstance(vt, (int, float)) or not math.isfinite(vt) \
                    or vt < self.virtual_time_s - 1e-6:
                raise ProtocolError('虚拟时间缺失、非有限或倒退')
            self.virtual_time_s = float(vt)
            self.accepted_actions += 1
            return response
        raise AssertionError('不可达')


def public_feedback(action, response):
    kind = response.get('measure_result' if action['action'] == 'measure'
                        else 'clear_result')
    valid = (('direction', 'near', 'no_signal')
             if action['action'] == 'measure'
             else ('success', 'no_target_in_range'))
    if kind not in valid:
        raise ProtocolError(f'结果码异常：{kind!r}')
    result = dict(accepted=True, kind=kind,
                  virtual_time_s=response['virtual_time_s'])
    if kind == 'direction':
        angle = response.get('svd_deg')
        if not isinstance(angle, (int, float)) or not math.isfinite(angle) \
                or not 0 <= angle < 360:
            raise ProtocolError('示向度异常')
        result['angle'] = float(angle)
    return result


def state_of(policy):
    return dict(cleared=int(policy.cleared.sum()),
                confirmed=int(policy.confirmed.sum()),
                absent=int(policy.absent.sum()),
                unknown=int(len(policy.unknown())))


def run_session(client, policy, max_actions=5000, time_margin=20.,
                verbose=True):
    started = time.monotonic()
    entered = client.post('/enter')
    remaining = entered.get('remaining_real_duration_s')
    if not isinstance(remaining, (int, float)) or not math.isfinite(remaining) \
            or not 0 <= remaining <= 1200:
        raise ProtocolError('remaining_real_duration_s 异常')
    client.deadline = started + remaining
    if verbose:
        print(f'[enter] 本局可用现实时间 {remaining:.0f} 秒，虚拟时间起点 '
              f'{client.virtual_time_s:.1f} 秒', flush=True)

    def graceful_exit(reason, step):
        summary = dict(status=reason, **state_of(policy),
                       absent_channels=policy.absence_reason,
                       virtual_s=client.virtual_time_s,
                       local_wall_s=time.monotonic() - started,
                       actions=step)
        try:
            response = client.post('/exit')
            summary['exit_reason'] = response.get('exit_reason')
        except Exception as error:
            summary['exit_error'] = f'{type(error).__name__}: {error}'
        return summary

    for step in range(max_actions):
        if client.remaining() < time_margin:
            if verbose:
                print('[warn] 现实时间将尽，主动退出', flush=True)
            return graceful_exit('deadline_exit', step)
        action = policy.next_action()
        if action is None:
            if not policy.complete():
                raise ProtocolError('策略停止但没有完成证书')
            response = client.post('/exit')
            if response.get('exit_reason') != 'user_exit':
                raise ProtocolError('退出原因异常')
            state = state_of(policy)
            return dict(status='completed_with_certificate',
                        virtual_s=client.virtual_time_s,
                        average_per_cleared_s=(client.virtual_time_s
                                               / state['cleared']
                                               if state['cleared'] else None),
                        local_wall_s=time.monotonic() - started,
                        actions=step,
                        absence_channels=policy.absence_reason,
                        dynamic_stop_reason=getattr(policy,
                                                    'dynamic_stop_reason', None),
                        probability_stopped_channels=getattr(
                            policy, 'probability_stopped_channels', 0),
                        **state)
        response = client.post('/' + action['action'], action)
        policy.accept_feedback(public_feedback(action, response))
        if verbose and (step % 10 == 0 or step < 5):
            state = state_of(policy)
            print(f'[{(step + 1):>4}] {action["action"]:<7} '
                  f'ch{action["channel"]:<3} {str(action.get("reason")):<24} '
                  f'虚拟 {client.virtual_time_s:>8.1f}s '
                  f'现实剩 {client.remaining():>5.0f}s '
                  f'已清 {state["cleared"]:>2} 已确认 {state["confirmed"]:>2} '
                  f'未决 {state["unknown"]:>2}', flush=True)
    raise RuntimeError('达到动作上限仍未结束')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true',
                        help='确认接口就绪后才会真正连接')
    parser.add_argument('--robot-id', help='当前登录的参赛队号')
    parser.add_argument('--base-url', default='http://127.0.0.1:2026')
    parser.add_argument('--log-dir', default=str(HERE / 'q4_http_logs'))
    parser.add_argument('--particles', type=int, default=32)
    parser.add_argument('--seed', type=int, default=9411)
    parser.add_argument('--existence-threshold', type=float, default=0.01)
    parser.add_argument('--time-margin', type=float, default=20.)
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    if not args.robot_id:
        parser.error('--run 需要 --robot-id')
    folder = Path(args.log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    name = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    policy = Policy(particles=args.particles, seed=args.seed,
                    existence_stop_threshold=args.existence_threshold)
    with (folder / (name + '.jsonl')).open('x', encoding='utf-8') as log:
        client = Client(args.robot_id, args.base_url, log=log)
        try:
            summary = run_session(client, policy,
                                  time_margin=args.time_margin)
        except Exception as error:
            summary = dict(status='error_or_incomplete',
                           error=f'{type(error).__name__}: {error}',
                           last_accepted_virtual_s=client.virtual_time_s)
            client.record(dict(event='failure', **summary))
        summary['log_file'] = str(folder / (name + '.jsonl'))
        (folder / (name + '.summary.json')).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary['status'] != 'completed_with_certificate':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
