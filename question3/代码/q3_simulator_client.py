"""Serial HTTP adapter for the frozen Q3 policy, following attachments 1/2.

Importing this module never connects. CLI connections require --run and a team
ID. Tests use an injected local mock, never the official simulator. Retrying
an uncertain transport operation reuses its exact bytes and request ID.
"""
import argparse
import http.client
import json
import math
from pathlib import Path
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid
from q3_joint_replan import JointReplanPolicy


class ProtocolError(RuntimeError): pass


def http_transport(url, data, timeout):
    request = Request(url, data=data, headers={'Content-Type':'application/json; charset=utf-8'}, method='POST')
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode('utf-8'))


class Client:
    def __init__(self, robot_id, base_url='http://127.0.0.1:2026', transport=http_transport,
                 log=None, timeout=5., retries=3, retry_delay=.2):
        if not 1 <= len(robot_id.encode('utf-8')) <= 64 or any(unicodedata.category(c) in ('Cc','Cf') for c in robot_id):
            raise ValueError('robot_id must be the team ID, 1–64 UTF-8 bytes without hidden/control characters')
        parsed = urlsplit(base_url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or parsed.path not in ('','/') or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('Use the simulator loopback HTTP address without a path/query')
        self.base = base_url.rstrip('/'); self.robot_id = robot_id
        self.transport = transport; self.log = log; self.timeout = timeout
        self.retries = retries; self.retry_delay = retry_delay; self.deadline = None
        self.virtual_time_s = 0.; self.accepted_actions = 0

    def record(self, value):
        if self.log is not None:
            self.log.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n'); self.log.flush()

    def post(self, path, action=None):
        if path not in ('/enter','/exit','/measure','/clear'): raise ValueError('Unknown endpoint')
        payload = dict(arena_id='default', robot_id=self.robot_id, request_id=uuid.uuid4().hex)
        if path in ('/measure','/clear'):
            if action is None: raise ValueError('Action required')
            channel = action['channel']; q = action['position']
            if not isinstance(channel, int) or isinstance(channel, bool) or not 1 <= channel <= 20: raise ValueError('Invalid channel')
            if len(q) != 2 or not all(math.isfinite(x) and abs(x) <= 2000000 for x in q): raise ValueError('Invalid position')
            payload.update(channel=channel, position=dict(x=float(q[0]), y=float(q[1])))
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',',':')).encode('utf-8')
        for attempt in range(self.retries+1):
            available = math.inf if self.deadline is None else self.deadline-time.monotonic()
            if available <= 0: raise TimeoutError('Actual remaining test time exhausted')
            self.record(dict(event='request', path=path, payload=payload, attempt=attempt, timestamp=time.time()))
            try:
                status, response = self.transport(self.base+path, data, min(self.timeout, available))
            except (URLError, TimeoutError, ConnectionError, http.client.RemoteDisconnected, http.client.IncompleteRead) as error:
                self.record(dict(event='transport_error', request_id=payload['request_id'], error=str(error)))
                if attempt == self.retries: raise
                time.sleep(min(self.retry_delay, max(0., available)))
                continue
            self.record(dict(event='response', path=path, request_id=payload['request_id'], status=status, body=response))
            if status != 200 or response.get('accepted') is not True:
                raise ProtocolError(f'{path} rejected: HTTP {status}, accepted={response.get("accepted")!r}')
            vt = response.get('virtual_time_s')
            if not isinstance(vt, (int,float)) or not math.isfinite(vt) or vt < self.virtual_time_s-1e-6:
                raise ProtocolError('Missing, nonfinite or decreasing virtual time')
            self.virtual_time_s = float(vt); self.accepted_actions += 1
            return response
        raise AssertionError('Unreachable')


def public_feedback(action, response):
    kind = response.get('measure_result' if action['action'] == 'measure' else 'clear_result')
    valid = ('direction','near','no_signal') if action['action'] == 'measure' else ('success','no_target_in_range')
    if kind not in valid: raise ProtocolError('Unexpected result code')
    result = dict(accepted=True, kind=kind, virtual_time_s=response['virtual_time_s'])
    if kind == 'direction':
        angle = response.get('svd_deg')
        if not isinstance(angle, (int,float)) or not math.isfinite(angle) or not 0 <= angle < 360:
            raise ProtocolError('Invalid direction reading')
        result['angle'] = float(angle)
    return result


def run_session(client, policy=None, max_actions=5000):
    policy = policy if policy is not None else JointReplanPolicy(seed=9101, particles=48, scan_scale=.5)
    started = time.monotonic(); entered = client.post('/enter')
    remaining = entered.get('remaining_real_duration_s')
    if not isinstance(remaining, (int,float)) or not math.isfinite(remaining) or not 0 <= remaining <= 1200:
        raise ProtocolError('Invalid actual remaining real duration')
    # Counting from before the enter round-trip is conservative under latency.
    client.deadline = started+remaining
    for step in range(max_actions):
        if time.monotonic() >= client.deadline: raise TimeoutError('Actual test deadline reached')
        action = policy.next_action()
        if action is None:
            if not policy.complete(): raise ProtocolError('Planner stopped without a completion certificate')
            response = client.post('/exit')
            if response.get('exit_reason') != 'user_exit': raise ProtocolError('Unexpected exit reason')
            cleared = int(policy.cleared.sum())
            return dict(status='completed_with_certificate', cleared=cleared, source_count=None,
                clear_fraction=None, virtual_s=client.virtual_time_s,
                average_per_cleared_s=client.virtual_time_s/cleared if cleared else None,
                local_wall_s=time.monotonic()-started, actions=step,
                confirmed_original_sources=int(policy.confirmed.sum()), absent_channels=policy.absence_reason,
                note='Use the simulator case code and displayed program time for official Table 1; this is not its encrypted log')
        response = client.post('/'+action['action'], action)
        policy.accept_feedback(public_feedback(action, response))
        client.record(dict(event='policy_state', step=step+1, position=policy.position.tolist(),
                           channel=policy.channel, cleared=int(policy.cleared.sum()),
                           confirmed=int(policy.confirmed.sum()), reason=action.get('reason')))
    raise RuntimeError('Finite action guard reached without completion; no successful exit claimed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Connect only after the simulator says the interface is ready')
    parser.add_argument('--robot-id', help='Current logged-in team ID')
    parser.add_argument('--base-url', default='http://127.0.0.1:2026')
    parser.add_argument('--log-dir', default='q3_http_logs')
    args = parser.parse_args()
    if not args.run: parser.print_help(); return
    if not args.robot_id: parser.error('--run requires --robot-id')
    folder = Path(args.log_dir); folder.mkdir(parents=True, exist_ok=True)
    name = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    with (folder/(name+'.jsonl')).open('x', encoding='utf-8') as log:
        client = Client(args.robot_id, args.base_url, log=log)
        try: summary = run_session(client)
        except Exception as error:
            summary = dict(status='error_or_incomplete', error=f'{type(error).__name__}: {error}',
                           last_accepted_virtual_s=client.virtual_time_s)
            client.record(dict(event='failure', **summary))
        (folder/(name+'.summary.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary['status'] != 'completed_with_certificate': raise SystemExit(1)


if __name__ == '__main__': main()
